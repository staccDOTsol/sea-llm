use solana_program::{
    account_info::{next_account_info, AccountInfo},
    entrypoint, entrypoint::ProgramResult, msg,
    program_error::ProgramError, pubkey::Pubkey,
    system_instruction, program::{invoke, invoke_signed},
};

entrypoint!(process_instruction);

// ── Model constants ──
const DIM: usize = 64;
const N_HEADS: usize = 16;
const HEAD_DIM: usize = DIM / N_HEADS;
const N_LAYERS: usize = 8;
const INTERMEDIATE: usize = 256;
const VOCAB_SIZE: usize = 50257;
const MAX_SEQ: usize = 32;
const FFN_UP_CHUNK: usize = INTERMEDIATE / 4;
const FFN_DOWN_CHUNK: usize = DIM / 4;

// ── Game constants ──
const MAGIC_TOKEN: u16 = 5536; // " magic" in GPT-2 vocab
// All tokens containing "magic" — banned from prompts to prevent trivial wins
const BANNED_TOKENS: [u16; 9] = [5536, 6139, 10883, 22294, 22975, 32707, 34850, 42055, 44546];
const ENTRY_FEE: u64 = 1_000_000_000; // 1 SOL
const DEV_FEE: u64 = 200_000_000;     // 0.2 SOL (20% to dev)
const POT_FEE: u64 = 800_000_000;     // 0.8 SOL (80% to pot)
// Dev wallet: 89VB5UmvopuCFmp5Mf8YPX28fGvvqn79afCgouQuPyhY
const DEV_WALLET: [u8; 32] = [
    0x6a, 0x2e, 0x4f, 0xa5, 0x0e, 0xc4, 0x50, 0x2e,
    0x1a, 0x2e, 0xb9, 0x12, 0xc9, 0xf2, 0xf9, 0x4e,
    0x86, 0xfc, 0x10, 0x09, 0xf8, 0xcb, 0xee, 0x23,
    0x31, 0xea, 0xdc, 0x6f, 0xee, 0xbc, 0x8e, 0x1f,
];

// ── Shard0 layout (F16 embeddings + INT8 lm_head) ──
const TOK_EMB_OFFSET: usize = 0;
const POS_EMB_OFFSET: usize = VOCAB_SIZE * DIM * 2;
const LM_HEAD_OFFSET: usize = POS_EMB_OFFSET + MAX_SEQ * DIM * 2;
const LM_HEAD_SCALE_OFFSET: usize = LM_HEAD_OFFSET + VOCAB_SIZE * DIM;
const FINAL_LN_OFFSET: usize = LM_HEAD_SCALE_OFFSET + 4;

// ── Shard1 layout (F32 transformer layers) ──
const ATTN_PROJ_SIZE: usize = DIM * DIM * 4;
const O_BIAS_OFF: usize = 4 * ATTN_PROJ_SIZE;
const LN1_W_OFF: usize = O_BIAS_OFF + DIM * 4;
const LN1_B_OFF: usize = LN1_W_OFF + DIM * 4;
const FC_W_OFF: usize = LN1_B_OFF + DIM * 4;
const FC_B_OFF: usize = FC_W_OFF + INTERMEDIATE * DIM * 4;
const PROJ_W_OFF: usize = FC_B_OFF + INTERMEDIATE * 4;
const PROJ_B_OFF: usize = PROJ_W_OFF + DIM * INTERMEDIATE * 4;
const LN2_W_OFF: usize = PROJ_B_OFF + DIM * 4;
const LN2_B_OFF: usize = LN2_W_OFF + DIM * 4;
const LAYER_SIZE: usize = LN2_B_OFF + DIM * 4;

// ── State account layout ──
// Header: [0]=initialized, [1]=stage, [2]=layer, [3..5]=seq_len(u16),
//         [5..7]=gen_count(u16), [7]=done, [8..10]=pred_token(u16),
//         [10..12]=last_pos(u16), [12]=has_won, [13]=reserved, [14]=ln_done
//         [15]=reserved
const HEADER_SIZE: usize = 16;
const TOKEN_AREA: usize = HEADER_SIZE;
const HIDDEN_STATES: usize = TOKEN_AREA + MAX_SEQ * 2;
const SCRATCH_AREA: usize = HIDDEN_STATES + MAX_SEQ * DIM * 4;
const Q_AREA: usize = SCRATCH_AREA + DIM * 4;
const KV_CACHE: usize = Q_AREA + DIM * 4;
const KV_LAYER_SIZE: usize = MAX_SEQ * DIM * 4 * 2;
const KV_CACHE_TOTAL: usize = N_LAYERS * KV_LAYER_SIZE;
const ATTN_OUT_AREA: usize = KV_CACHE + KV_CACHE_TOTAL;
const FFN_INTER_AREA: usize = ATTN_OUT_AREA + DIM * 4;
const FFN_DOWN_AREA: usize = FFN_INTER_AREA + INTERMEDIATE * 4;
const LN_HIDDEN_OFFSET: usize = FFN_DOWN_AREA + DIM * 4;
const ARGMAX_Q_AREA: usize = LN_HIDDEN_OFFSET + DIM * 4;
pub const STATE_SIZE: usize = ARGMAX_Q_AREA + DIM + 4 + 64;

// ── KV cache helpers ──
#[inline(always)]
fn kv_k_off(layer: usize, pos: usize, d: usize) -> usize {
    KV_CACHE + layer * KV_LAYER_SIZE + pos * DIM * 4 + d * 4
}
#[inline(always)]
fn kv_v_off(layer: usize, pos: usize, d: usize) -> usize {
    KV_CACHE + layer * KV_LAYER_SIZE + MAX_SEQ * DIM * 4 + pos * DIM * 4 + d * 4
}

// ── Argmax workers ──
const N_WORKERS: usize = 4;
const TOKENS_PER_WORKER: usize = (VOCAB_SIZE + N_WORKERS - 1) / N_WORKERS;
const N_SUB_CHUNKS: usize = 16;
const TOKENS_PER_SUB_CHUNK: usize = (TOKENS_PER_WORKER + N_SUB_CHUNKS - 1) / N_SUB_CHUNKS;
pub const WORKER_SIZE: usize = 64;

// ── Pot PDA layout (72 bytes) ──
// [0..8]   total_pot (u64 lamports - excess above rent)
// [8..16]  total_plays (u64)
// [16..48] last_winner (32 bytes pubkey)
// [48..56] last_win_amount (u64)
// [56..64] reserved
// [64..72] total_won (u64)
pub const POT_SIZE: usize = 72;
const POT_TOTAL_POT: usize = 0;
const POT_TOTAL_PLAYS: usize = 8;
const POT_LAST_WINNER: usize = 16;
const POT_LAST_WIN_AMT: usize = 48;
const POT_TOTAL_WON: usize = 64;

// ── Instruction codes ──
const IX_INIT_POT: u8 = 1;
const IX_INIT: u8 = 2;
const IX_EMBED: u8 = 3;
const IX_LN1_Q: u8 = 40;
const IX_K_PROJ: u8 = 41;
const IX_V_PROJ: u8 = 42;
const IX_ATTN: u8 = 43;
const IX_O_PROJ: u8 = 44;
const IX_LN2: u8 = 45;
const IX_FFN_UP: u8 = 46;
const IX_FFN_DOWN: u8 = 48;
const IX_FFN_RESIDUAL: u8 = 50;
const IX_OUTPUT_LN: u8 = 9;
const IX_WORKER_ARGMAX: u8 = 10;
const IX_WORKER_MERGE: u8 = 11;
const IX_COPY_HIDDEN: u8 = 13;
const IX_NEXT_TOKEN: u8 = 14;
const IX_CLAIM_WIN: u8 = 20;
const IX_CLOSE_SESSION: u8 = 21;
const IX_WRITE_SHARD: u8 = 30;
const IX_ADMIN_CLOSE: u8 = 99;

// ── State field offsets (for close_session logging) ──
const STATE_SEQ_LEN: usize = 3;
const STATE_GEN_COUNT: usize = 5;
const STATE_PRED_TOKEN: usize = 8;
const STATE_HAS_WON: usize = 12;
const STATE_TOKEN_AREA: usize = TOKEN_AREA;

// ── Byte helpers ──
#[inline(always)]
fn read_u16_le(d: &[u8], o: usize) -> u16 { u16::from_le_bytes([d[o], d[o+1]]) }
#[inline(always)]
fn write_u16_le(d: &mut [u8], o: usize, v: u16) { let b = v.to_le_bytes(); d[o] = b[0]; d[o+1] = b[1]; }
#[inline(always)]
fn read_u64_le(d: &[u8], o: usize) -> u64 { u64::from_le_bytes([d[o],d[o+1],d[o+2],d[o+3],d[o+4],d[o+5],d[o+6],d[o+7]]) }
#[inline(always)]
fn write_u64_le(d: &mut [u8], o: usize, v: u64) { let b = v.to_le_bytes(); for i in 0..8 { d[o+i] = b[i]; } }
#[inline(always)]
fn read_f32_le(d: &[u8], o: usize) -> f32 { f32::from_le_bytes([d[o], d[o+1], d[o+2], d[o+3]]) }
#[inline(always)]
fn write_f32_le(d: &mut [u8], o: usize, v: f32) { let b = v.to_le_bytes(); d[o]=b[0]; d[o+1]=b[1]; d[o+2]=b[2]; d[o+3]=b[3]; }
#[inline(always)]
fn read_f16_le(d: &[u8], o: usize) -> f32 { f16_to_f32(u16::from_le_bytes([d[o], d[o+1]])) }

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    if exp == 0 {
        if mant == 0 { return f32::from_bits(sign << 31); }
        let mut m = mant; let mut e: i32 = -14;
        while (m & 0x400) == 0 { m <<= 1; e -= 1; }
        m &= 0x3FF;
        return f32::from_bits((sign << 31) | ((((e + 127) as u32) & 0xFF) << 23) | (m << 13));
    }
    if exp == 31 { return f32::from_bits((sign << 31) | (0xFF << 23) | if mant == 0 { 0 } else { mant << 13 }); }
    f32::from_bits((sign << 31) | (((exp as i32 - 15 + 127) as u32) << 23) | (mant << 13))
}

fn gelu_new(x: f32) -> f32 {
    let inner = 0.7978845608f32 * (x + 0.044715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

// ── Entrypoint ──
pub fn process_instruction(pid: &Pubkey, accounts: &[AccountInfo], ix_data: &[u8]) -> ProgramResult {
    if ix_data.is_empty() { return Err(ProgramError::InvalidInstructionData); }
    match ix_data[0] {
        IX_INIT_POT => init_pot(pid, accounts),
        IX_INIT => init(pid, accounts, &ix_data[1..]),
        IX_EMBED => embed(accounts),
        IX_LN1_Q => ln1_q(accounts, &ix_data[1..]),
        IX_K_PROJ => k_proj(accounts, &ix_data[1..]),
        IX_V_PROJ => v_proj(accounts, &ix_data[1..]),
        IX_ATTN => attn(accounts, &ix_data[1..]),
        IX_O_PROJ => o_proj(accounts, &ix_data[1..]),
        IX_LN2 => ln2(accounts, &ix_data[1..]),
        IX_FFN_UP => ffn_up(accounts, &ix_data[1..]),
        IX_FFN_DOWN => ffn_down(accounts, &ix_data[1..]),
        IX_FFN_RESIDUAL => ffn_residual(accounts, &ix_data[1..]),
        IX_OUTPUT_LN => output_ln(accounts),
        IX_WORKER_ARGMAX => worker_argmax(accounts, &ix_data[1..]),
        IX_WORKER_MERGE => worker_merge(accounts),
        IX_COPY_HIDDEN => copy_hidden(accounts),
        IX_NEXT_TOKEN => next_token(accounts),
        IX_CLAIM_WIN => claim_win(pid, accounts),
        IX_CLOSE_SESSION => close_session(pid, accounts),
        IX_WRITE_SHARD => write_shard(accounts, &ix_data[1..]),
        IX_ADMIN_CLOSE => admin_close_account(accounts),
        _ => Err(ProgramError::InvalidInstructionData),
    }
}

// ═══════════════════════════════════════════════════════════════
// GAME LOGIC
// ═══════════════════════════════════════════════════════════════

/// Write data to a shard/state/worker account. Admin only.
/// Data format: [offset: u32 LE][data bytes...]
/// Accounts: [0] target account (writable), [1] admin (signer)
fn write_shard(accounts: &[AccountInfo], data: &[u8]) -> ProgramResult {
    let accs = &mut accounts.iter();
    let target = next_account_info(accs)?;
    let admin = next_account_info(accs)?;
    
    // Only the dev wallet can write
    if admin.key.to_bytes() != DEV_WALLET {
        msg!("Only admin can write shard data");
        return Err(ProgramError::InvalidAccountData);
    }
    if !admin.is_signer {
        return Err(ProgramError::MissingRequiredSignature);
    }
    if data.len() < 4 {
        return Err(ProgramError::InvalidInstructionData);
    }
    
    let offset = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let payload = &data[4..];
    
    let mut target_data = target.try_borrow_mut_data()?;
    if offset + payload.len() > target_data.len() {
        msg!("Write out of bounds: {} + {} > {}", offset, payload.len(), target_data.len());
        return Err(ProgramError::InvalidInstructionData);
    }
    
    target_data[offset..offset + payload.len()].copy_from_slice(payload);
    Ok(())
}

/// Initialize the pot PDA. Called once by admin.
/// Accounts: [pot_pda (writable), admin (signer, writable), system_program]
fn init_pot(pid: &Pubkey, accounts: &[AccountInfo]) -> ProgramResult {
    let pot = &accounts[0];
    let admin = &accounts[1];
    let sys_prog = &accounts[2];
    if !admin.is_signer { return Err(ProgramError::MissingRequiredSignature); }
    // Verify PDA
    let (expected_pda, bump) = Pubkey::find_program_address(&[b"pot"], pid);
    if *pot.key != expected_pda { return Err(ProgramError::InvalidSeeds); }
    
    // If pot already exists and is owned by program, just re-init
    if *pot.owner == *pid && pot.data_len() >= POT_SIZE {
        let mut pd = pot.try_borrow_mut_data()?;
        for i in 0..POT_SIZE { pd[i] = 0; }
        msg!("Pot re-initialized");
        return Ok(());
    }
    
    // Create the PDA account using invoke_signed
    let rent = 890880 + (POT_SIZE as u64) * 6960; // approximate rent-exempt minimum
    let seeds: &[&[u8]] = &[b"pot", &[bump]];
    invoke_signed(
        &system_instruction::create_account(
            admin.key,
            pot.key,
            rent,
            POT_SIZE as u64,
            pid,
        ),
        &[admin.clone(), pot.clone(), sys_prog.clone()],
        &[seeds],
    )?;
    
    let mut pd = pot.try_borrow_mut_data()?;
    for i in 0..POT_SIZE { pd[i] = 0; }
    msg!("Pot PDA created and initialized");
    Ok(())
}

/// Init a game session. User pays ENTRY_FEE: 80% to pot, 20% to dev.
/// Accounts: [state (writable), user (signer, writable), pot_pda (writable), dev_wallet (writable), system_program]
fn init(pid: &Pubkey, accounts: &[AccountInfo], data: &[u8]) -> ProgramResult {
    let state = &accounts[0];
    let user = &accounts[1];
    let pot = &accounts[2];
    let dev = &accounts[3];
    let _sys = &accounts[4]; // system_program

    if !user.is_signer { return Err(ProgramError::MissingRequiredSignature); }
    if !state.is_writable || state.data_len() < STATE_SIZE {
        return Err(ProgramError::AccountDataTooSmall);
    }
    if *state.owner != *pid { return Err(ProgramError::IllegalOwner); }

    // Verify pot PDA
    let (expected_pda, _bump) = Pubkey::find_program_address(&[b"pot"], pid);
    if *pot.key != expected_pda { return Err(ProgramError::InvalidSeeds); }

    // Verify dev wallet
    if dev.key.to_bytes() != DEV_WALLET {
        return Err(ProgramError::InvalidAccountData);
    }

    // Transfer 80% to pot (0.8 SOL)
    invoke(
        &system_instruction::transfer(user.key, pot.key, POT_FEE),
        &[user.clone(), pot.clone()],
    )?;

    // Transfer 20% to dev (0.2 SOL)
    invoke(
        &system_instruction::transfer(user.key, dev.key, DEV_FEE),
        &[user.clone(), dev.clone()],
    )?;

    // Update pot stats
    {
        let mut pd = pot.try_borrow_mut_data()?;
        let total = read_u64_le(&pd, POT_TOTAL_POT);
        write_u64_le(&mut pd, POT_TOTAL_POT, total + POT_FEE);
        let plays = read_u64_le(&pd, POT_TOTAL_PLAYS);
        write_u64_le(&mut pd, POT_TOTAL_PLAYS, plays + 1);
    }

    // Parse prompt tokens
    let seq_len = read_u16_le(data, 0) as usize;
    if data.len() < 2 + seq_len * 2 || seq_len > MAX_SEQ || seq_len == 0 {
        return Err(ProgramError::InvalidInstructionData);
    }

    // Reject prompts containing any "magic"-related tokens
    for i in 0..seq_len {
        let tok = read_u16_le(data, 2 + i * 2);
        for &banned in &BANNED_TOKENS {
            if tok == banned {
                msg!("Banned token {} in prompt — no magic allowed!", tok);
                return Err(ProgramError::InvalidInstructionData);
            }
        }
    }

    // Initialize state
    let mut sd = state.try_borrow_mut_data()?;
    sd.fill(0);
    sd[0] = 1; // initialized
    write_u16_le(&mut sd, 3, seq_len as u16);
    write_u16_le(&mut sd, 10, 0);
    for i in 0..seq_len {
        write_u16_le(&mut sd, TOKEN_AREA + i * 2, read_u16_le(data, 2 + i * 2));
    }

    msg!("Game: user={}, pot=0.8SOL, dev=0.2SOL, seq={}", user.key, seq_len);
    Ok(())
}

/// Claim win: if pred_token == MAGIC_TOKEN, user wins the pot.
/// Accounts: [state (writable), user (signer, writable), pot_pda (writable)]
fn claim_win(pid: &Pubkey, accounts: &[AccountInfo]) -> ProgramResult {
    let state = &accounts[0];
    let user = &accounts[1];
    let pot = &accounts[2];

    if !user.is_signer { return Err(ProgramError::MissingRequiredSignature); }

    // Verify pot PDA
    let (expected_pda, bump) = Pubkey::find_program_address(&[b"pot"], pid);
    if *pot.key != expected_pda { return Err(ProgramError::InvalidSeeds); }

    let mut sd = state.try_borrow_mut_data()?;
    // Check that inference is done (done flag set)
    if sd[7] != 1 { return Err(ProgramError::InvalidAccountData); }
    // Check that this session hasn't already won
    if sd[12] == 1 { return Err(ProgramError::InvalidAccountData); }

    let pred = read_u16_le(&sd, 8);
    if pred != MAGIC_TOKEN {
        msg!("Not magic: got token {}", pred);
        return Err(ProgramError::InvalidAccountData);
    }

    // WINNER! Transfer pot to user
    let pot_lamports = pot.lamports();
    // Keep rent-exempt minimum in pot
    let rent_min = 890880 + (POT_SIZE as u64) * 6960; // ~1.4K lamports
    let payout = if pot_lamports > rent_min { pot_lamports - rent_min } else { 0 };

    if payout > 0 {
        // PDA transfer: debit pot, credit user
        **pot.try_borrow_mut_lamports()? -= payout;
        **user.try_borrow_mut_lamports()? += payout;
    }

    // Update pot stats
    {
        let mut pd = pot.try_borrow_mut_data()?;
        write_u64_le(&mut pd, POT_TOTAL_POT, 0); // pot emptied
        // Record winner
        let user_bytes = user.key.to_bytes();
        for i in 0..32 { pd[POT_LAST_WINNER + i] = user_bytes[i]; }
        write_u64_le(&mut pd, POT_LAST_WIN_AMT, payout);
        let total_won = read_u64_le(&pd, POT_TOTAL_WON);
        write_u64_le(&mut pd, POT_TOTAL_WON, total_won + payout);
    }

    // Mark session as won
    sd[12] = 1;

    msg!("WINNER! user={} payout={}", user.key, payout);
    Ok(())
}

/// Close session: send all lamports from state + workers to pot PDA.
/// Accounts: [state (writable), user (signer), pot_pda (writable), worker0..workerN (writable)]
fn close_session(pid: &Pubkey, accounts: &[AccountInfo]) -> ProgramResult {
    let state = &accounts[0];
    let user = &accounts[1];
    let pot = &accounts[2];

    if !user.is_signer { return Err(ProgramError::MissingRequiredSignature); }

    // Verify pot PDA
    let (expected_pda, _bump) = Pubkey::find_program_address(&[b"pot"], pid);
    if *pot.key != expected_pda { return Err(ProgramError::InvalidSeeds); }

    // Transfer all lamports from state account to pot
    let state_lamports = state.lamports();
    if state_lamports > 0 {
        **state.try_borrow_mut_lamports()? -= state_lamports;
        **pot.try_borrow_mut_lamports()? += state_lamports;
    }

    // Transfer all lamports from worker accounts to pot
    for i in 3..accounts.len() {
        let worker = &accounts[i];
        let wl = worker.lamports();
        if wl > 0 {
            **worker.try_borrow_mut_lamports()? -= wl;
            **pot.try_borrow_mut_lamports()? += wl;
        }
    }

    // Update pot total
    {
        let mut pd = pot.try_borrow_mut_data()?;
        let total = read_u64_le(&pd, POT_TOTAL_POT);
        write_u64_le(&mut pd, POT_TOTAL_POT, total + state_lamports);
    }

    // Log the generated output tokens for the live feed to read
    {
        let sd = state.try_borrow_data()?;
        let seq_len = read_u16_le(&sd, STATE_SEQ_LEN) as usize;
        let gen_count = read_u16_le(&sd, STATE_GEN_COUNT) as usize;
        let pred_token = read_u16_le(&sd, STATE_PRED_TOKEN);
        let has_won = sd[STATE_HAS_WON] != 0;
        
        // Log individual output tokens (up to 10)
        for g in 0..gen_count.min(10) {
            let off = STATE_TOKEN_AREA + (seq_len + g) * 2;
            if off + 2 <= sd.len() {
                let tok = read_u16_le(&sd, off);
                msg!("OUT_TOKEN:{}", tok);
            }
        }
        
        // Log summary
        msg!("Session closed: lamports={}, seq_len={}, gen_count={}, pred={}, won={}",
            state_lamports, seq_len, gen_count, pred_token, has_won);
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════
// INFERENCE LOGIC (unchanged from working version)
// ═══════════════════════════════════════════════════════════════

fn embed(accounts: &[AccountInfo]) -> ProgramResult {
    let state = &accounts[0];
    let shard0 = &accounts[1];
    let mut sd = state.try_borrow_mut_data()?;
    if sd[0] != 1 { return Err(ProgramError::UninitializedAccount); }
    let seq_len = read_u16_le(&sd, 3) as usize;
    let sh = shard0.try_borrow_data()?;
    for pos in 0..seq_len {
        let tid = read_u16_le(&sd, TOKEN_AREA + pos * 2) as usize;
        for d in 0..DIM {
            let te = read_f16_le(&sh, TOK_EMB_OFFSET + (tid * DIM + d) * 2);
            let pe = read_f16_le(&sh, POS_EMB_OFFSET + (pos * DIM + d) * 2);
            write_f32_le(&mut sd, HIDDEN_STATES + (pos * DIM + d) * 4, te + pe);
        }
    }
    sd[1] = 1; sd[2] = 0;
    Ok(())
}

fn ln1_q(accounts: &[AccountInfo], data: &[u8]) -> ProgramResult {
    let state = &accounts[0]; let shard1 = &accounts[1];
    let pos = data[0] as usize;
    let mut sd = state.try_borrow_mut_data()?;
    let layer = sd[2] as usize;
    let sh = shard1.try_borrow_data()?;
    let lo = layer * LAYER_SIZE;
    let mut h = [0.0f32; DIM];
    for d in 0..DIM { h[d] = read_f32_le(&sd, HIDDEN_STATES + (pos * DIM + d) * 4); }
    let mut mean = 0.0f32;
    for d in 0..DIM { mean += h[d]; }
    mean /= DIM as f32;
    let mut var = 0.0f32;
    for d in 0..DIM { let diff = h[d] - mean; var += diff * diff; }
    let inv = 1.0 / (var / DIM as f32 + 1e-5f32).sqrt();
    for d in 0..DIM {
        h[d] = (h[d] - mean) * inv * read_f32_le(&sh, lo + LN1_W_OFF + d * 4)
            + read_f32_le(&sh, lo + LN1_B_OFF + d * 4);
    }
    for d in 0..DIM { write_f32_le(&mut sd, SCRATCH_AREA + d * 4, h[d]); }
    for r in 0..DIM {
        let mut acc = 0.0f32;
        for c in 0..DIM { acc += read_f32_le(&sh, lo + (r * DIM + c) * 4) * h[c]; }
        write_f32_le(&mut sd, Q_AREA + r * 4, acc);
    }
    Ok(())
}

fn k_proj(accounts: &[AccountInfo], data: &[u8]) -> ProgramResult {
    let state = &accounts[0]; let shard1 = &accounts[1];
    let pos = data[0] as usize;
    let mut sd = state.try_borrow_mut_data()?;
    let layer = sd[2] as usize;
    let sh = shard1.try_borrow_data()?;
    let lo = layer * LAYER_SIZE + ATTN_PROJ_SIZE;
    let mut h = [0.0f32; DIM];
    for d in 0..DIM { h[d] = read_f32_le(&sd, SCRATCH_AREA + d * 4); }
    for r in 0..DIM {
        let mut acc = 0.0f32;
        for c in 0..DIM { acc += read_f32_le(&sh, lo + (r * DIM + c) * 4) * h[c]; }
        write_f32_le(&mut sd, kv_k_off(layer, pos, r), acc);
    }
    Ok(())
}

fn v_proj(accounts: &[AccountInfo], data: &[u8]) -> ProgramResult {
    let state = &accounts[0]; let shard1 = &accounts[1];
    let pos = data[0] as usize;
    let mut sd = state.try_borrow_mut_data()?;
    let layer = sd[2] as usize;
    let sh = shard1.try_borrow_data()?;
    let lo = layer * LAYER_SIZE + 2 * ATTN_PROJ_SIZE;
    let mut h = [0.0f32; DIM];
    for d in 0..DIM { h[d] = read_f32_le(&sd, SCRATCH_AREA + d * 4); }
    for r in 0..DIM {
        let mut acc = 0.0f32;
        for c in 0..DIM { acc += read_f32_le(&sh, lo + (r * DIM + c) * 4) * h[c]; }
        write_f32_le(&mut sd, kv_v_off(layer, pos, r), acc);
    }
    Ok(())
}

fn attn(accounts: &[AccountInfo], data: &[u8]) -> ProgramResult {
    let state = &accounts[0];
    let pos = data[0] as usize;
    let mut sd = state.try_borrow_mut_data()?;
    let layer = sd[2] as usize;
    let mut q = [0.0f32; DIM];
    for d in 0..DIM { q[d] = read_f32_le(&sd, Q_AREA + d * 4); }
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let n_kv = pos + 1;
    let mut attn_out = [0.0f32; DIM];
    for head in 0..N_HEADS {
        let ho = head * HEAD_DIM;
        let mut scores = [0.0f32; MAX_SEQ];
        let mut max_s = -1e30f32;
        for kp in 0..n_kv {
            let mut dot = 0.0f32;
            for d in 0..HEAD_DIM { dot += q[ho + d] * read_f32_le(&sd, kv_k_off(layer, kp, ho + d)); }
            scores[kp] = dot * scale;
            if scores[kp] > max_s { max_s = scores[kp]; }
        }
        let mut sum = 0.0f32;
        for i in 0..n_kv { scores[i] = (scores[i] - max_s).exp(); sum += scores[i]; }
        let inv = 1.0 / sum;
        for kp in 0..n_kv {
            let w = scores[kp] * inv;
            if w < 1e-6 { continue; }
            for d in 0..HEAD_DIM { attn_out[ho + d] += w * read_f32_le(&sd, kv_v_off(layer, kp, ho + d)); }
        }
    }
    for d in 0..DIM { write_f32_le(&mut sd, ATTN_OUT_AREA + d * 4, attn_out[d]); }
    Ok(())
}

fn o_proj(accounts: &[AccountInfo], data: &[u8]) -> ProgramResult {
    let state = &accounts[0]; let shard1 = &accounts[1];
    let pos = data[0] as usize;
    let mut sd = state.try_borrow_mut_data()?;
    let layer = sd[2] as usize;
    let sh = shard1.try_borrow_data()?;
    let lo = layer * LAYER_SIZE;
    let mut ao = [0.0f32; DIM];
    for d in 0..DIM { ao[d] = read_f32_le(&sd, ATTN_OUT_AREA + d * 4); }
    let o_w = lo + 3 * ATTN_PROJ_SIZE;
    for r in 0..DIM {
        let mut acc = 0.0f32;
        for c in 0..DIM { acc += read_f32_le(&sh, o_w + (r * DIM + c) * 4) * ao[c]; }
        let bias = read_f32_le(&sh, lo + O_BIAS_OFF + r * 4);
        let res = read_f32_le(&sd, HIDDEN_STATES + (pos * DIM + r) * 4);
        write_f32_le(&mut sd, HIDDEN_STATES + (pos * DIM + r) * 4, acc + bias + res);
    }
    Ok(())
}

fn ln2(accounts: &[AccountInfo], data: &[u8]) -> ProgramResult {
    let state = &accounts[0]; let shard1 = &accounts[1];
    let pos = data[0] as usize;
    let mut sd = state.try_borrow_mut_data()?;
    let layer = sd[2] as usize;
    let sh = shard1.try_borrow_data()?;
    let lo = layer * LAYER_SIZE;
    let mut h = [0.0f32; DIM];
    for d in 0..DIM { h[d] = read_f32_le(&sd, HIDDEN_STATES + (pos * DIM + d) * 4); }
    let mut mean = 0.0f32;
    for d in 0..DIM { mean += h[d]; }
    mean /= DIM as f32;
    let mut var = 0.0f32;
    for d in 0..DIM { let diff = h[d] - mean; var += diff * diff; }
    let inv = 1.0 / (var / DIM as f32 + 1e-5f32).sqrt();
    for d in 0..DIM {
        h[d] = (h[d] - mean) * inv * read_f32_le(&sh, lo + LN2_W_OFF + d * 4)
            + read_f32_le(&sh, lo + LN2_B_OFF + d * 4);
    }
    for d in 0..DIM { write_f32_le(&mut sd, SCRATCH_AREA + d * 4, h[d]); }
    Ok(())
}

fn ffn_up(accounts: &[AccountInfo], data: &[u8]) -> ProgramResult {
    let state = &accounts[0]; let shard1 = &accounts[1];
    let chunk = data[1] as usize;
    let mut sd = state.try_borrow_mut_data()?;
    let layer = sd[2] as usize;
    let sh = shard1.try_borrow_data()?;
    let lo = layer * LAYER_SIZE;
    let mut h = [0.0f32; DIM];
    for d in 0..DIM { h[d] = read_f32_le(&sd, SCRATCH_AREA + d * 4); }
    let start = chunk * FFN_UP_CHUNK;
    let end = start + FFN_UP_CHUNK;
    let fc_w = lo + FC_W_OFF;
    let fc_b = lo + FC_B_OFF;
    for r in start..end {
        let mut acc = 0.0f32;
        for c in 0..DIM { acc += read_f32_le(&sh, fc_w + (r * DIM + c) * 4) * h[c]; }
        acc += read_f32_le(&sh, fc_b + r * 4);
        write_f32_le(&mut sd, FFN_INTER_AREA + r * 4, gelu_new(acc));
    }
    Ok(())
}

fn ffn_down(accounts: &[AccountInfo], data: &[u8]) -> ProgramResult {
    let state = &accounts[0]; let shard1 = &accounts[1];
    let chunk = data[1] as usize;
    let mut sd = state.try_borrow_mut_data()?;
    let layer = sd[2] as usize;
    let sh = shard1.try_borrow_data()?;
    let lo = layer * LAYER_SIZE;
    let mut inter = [0.0f32; INTERMEDIATE];
    for i in 0..INTERMEDIATE { inter[i] = read_f32_le(&sd, FFN_INTER_AREA + i * 4); }
    let start = chunk * FFN_DOWN_CHUNK;
    let end = start + FFN_DOWN_CHUNK;
    let proj_w = lo + PROJ_W_OFF;
    let proj_b = lo + PROJ_B_OFF;
    for r in start..end {
        let mut acc = 0.0f32;
        for c in 0..INTERMEDIATE { acc += read_f32_le(&sh, proj_w + (r * INTERMEDIATE + c) * 4) * inter[c]; }
        acc += read_f32_le(&sh, proj_b + r * 4);
        write_f32_le(&mut sd, FFN_DOWN_AREA + r * 4, acc);
    }
    Ok(())
}

fn ffn_residual(accounts: &[AccountInfo], data: &[u8]) -> ProgramResult {
    let state = &accounts[0];
    let pos = data[0] as usize;
    let mut sd = state.try_borrow_mut_data()?;
    let seq_len = read_u16_le(&sd, 3) as usize;
    let layer = sd[2] as usize;
    for r in 0..DIM {
        let ffn_out = read_f32_le(&sd, FFN_DOWN_AREA + r * 4);
        let res = read_f32_le(&sd, HIDDEN_STATES + (pos * DIM + r) * 4);
        write_f32_le(&mut sd, HIDDEN_STATES + (pos * DIM + r) * 4, ffn_out + res);
    }
    if pos == seq_len - 1 {
        let next = layer + 1;
        if next < N_LAYERS { sd[2] = next as u8; } else { sd[1] = 4; }
        msg!("L{} done", layer);
    }
    Ok(())
}

fn output_ln(accounts: &[AccountInfo]) -> ProgramResult {
    let state = &accounts[0]; let shard0 = &accounts[1];
    let mut sd = state.try_borrow_mut_data()?;
    let seq_len = read_u16_le(&sd, 3) as usize;
    let last = seq_len - 1;
    let sh = shard0.try_borrow_data()?;
    let mut h = [0.0f32; DIM];
    for d in 0..DIM { h[d] = read_f32_le(&sd, HIDDEN_STATES + (last * DIM + d) * 4); }
    let mut mean = 0.0f32;
    for d in 0..DIM { mean += h[d]; }
    mean /= DIM as f32;
    let mut var = 0.0f32;
    for d in 0..DIM { let diff = h[d] - mean; var += diff * diff; }
    let inv = 1.0 / (var / DIM as f32 + 1e-5f32).sqrt();
    for d in 0..DIM {
        h[d] = (h[d] - mean) * inv * read_f32_le(&sh, FINAL_LN_OFFSET + d * 4)
            + read_f32_le(&sh, FINAL_LN_OFFSET + DIM * 4 + d * 4);
    }
    for d in 0..DIM { write_f32_le(&mut sd, LN_HIDDEN_OFFSET + d * 4, h[d]); }
    let mut max_abs = 0.0f32;
    for d in 0..DIM { let a = h[d].abs(); if a > max_abs { max_abs = a; } }
    let scale = if max_abs > 0.0 { 127.0 / max_abs } else { 1.0 };
    for d in 0..DIM { sd[ARGMAX_Q_AREA + d] = (h[d] * scale).round() as i8 as u8; }
    write_f32_le(&mut sd, ARGMAX_Q_AREA + DIM, 1.0 / scale);
    sd[14] = 1;
    msg!("OutLN done");
    Ok(())
}

fn copy_hidden(accounts: &[AccountInfo]) -> ProgramResult {
    let worker = &accounts[0];
    let mut wd = worker.try_borrow_mut_data()?;
    write_u16_le(&mut wd, 0, 0);
    let min_bytes = i32::MIN.to_le_bytes();
    wd[2] = min_bytes[0]; wd[3] = min_bytes[1]; wd[4] = min_bytes[2]; wd[5] = min_bytes[3];
    Ok(())
}

fn worker_argmax(accounts: &[AccountInfo], data: &[u8]) -> ProgramResult {
    let worker = &accounts[0]; let state = &accounts[1]; let shard0 = &accounts[2];
    let worker_id = data[0] as usize;
    let sub_chunk = data[1] as usize;
    let sd = state.try_borrow_data()?;
    let sh = shard0.try_borrow_data()?;
    let mut wd = worker.try_borrow_mut_data()?;
    let mut hi8 = [0i8; DIM];
    for d in 0..DIM { hi8[d] = sd[ARGMAX_Q_AREA + d] as i8; }
    let ws = worker_id * TOKENS_PER_WORKER;
    let we = core::cmp::min(ws + TOKENS_PER_WORKER, VOCAB_SIZE);
    let ss = ws + sub_chunk * TOKENS_PER_SUB_CHUNK;
    let se = core::cmp::min(ss + TOKENS_PER_SUB_CHUNK, we);
    if ss >= we { return Ok(()); }
    let mut bt = read_u16_le(&wd, 0);
    let mut bd = i32::MIN;
    if wd.len() > 2 { bd = i32::from_le_bytes([wd[2], wd[3], wd[4], wd[5]]); }
    for tok in ss..se {
        let ro = LM_HEAD_OFFSET + tok * DIM;
        let mut dot: i32 = 0;
        for d in 0..DIM { dot += sh[ro + d] as i8 as i32 * hi8[d] as i32; }
        if dot > bd { bd = dot; bt = tok as u16; }
    }
    write_u16_le(&mut wd, 0, bt);
    let bd_bytes = bd.to_le_bytes();
    wd[2] = bd_bytes[0]; wd[3] = bd_bytes[1]; wd[4] = bd_bytes[2]; wd[5] = bd_bytes[3];
    Ok(())
}

fn worker_merge(accounts: &[AccountInfo]) -> ProgramResult {
    let state = &accounts[0];
    let mut sd = state.try_borrow_mut_data()?;
    let mut bt: u16 = 0;
    let mut bd: i32 = i32::MIN;
    for i in 1..=N_WORKERS {
        let wd = accounts[i].try_borrow_data()?;
        let t = read_u16_le(&wd, 0);
        let d = if wd.len() > 2 { i32::from_le_bytes([wd[2], wd[3], wd[4], wd[5]]) } else { i32::MIN };
        if d > bd { bd = d; bt = t; }
    }
    write_u16_le(&mut sd, 8, bt);
    sd[7] = 1;
    let gen = read_u16_le(&sd, 5);
    write_u16_le(&mut sd, 5, gen + 1);
    let seq_len = read_u16_le(&sd, 3);
    write_u16_le(&mut sd, 10, seq_len);
    msg!("Merge: tok={}", bt);
    Ok(())
}

fn next_token(accounts: &[AccountInfo]) -> ProgramResult {
    let state = &accounts[0]; let shard0 = &accounts[1];
    let mut sd = state.try_borrow_mut_data()?;
    if sd[7] != 1 { return Err(ProgramError::InvalidAccountData); }
    let pred = read_u16_le(&sd, 8) as usize;
    let seq_len = read_u16_le(&sd, 3) as usize;
    if seq_len >= MAX_SEQ { return Ok(()); }
    write_u16_le(&mut sd, TOKEN_AREA + seq_len * 2, pred as u16);
    write_u16_le(&mut sd, 3, (seq_len + 1) as u16);
    let sh = shard0.try_borrow_data()?;
    for d in 0..DIM {
        let te = read_f16_le(&sh, TOK_EMB_OFFSET + (pred * DIM + d) * 2);
        let pe = read_f16_le(&sh, POS_EMB_OFFSET + (seq_len * DIM + d) * 2);
        write_f32_le(&mut sd, HIDDEN_STATES + (seq_len * DIM + d) * 4, te + pe);
    }
    sd[1] = 1; sd[2] = 0; sd[7] = 0; sd[14] = 0;
    msg!("Next: tok={} pos={}", pred, seq_len);
    Ok(())
}

// ═══════════════════════════════════════════════════════════════
// ADMIN: CLOSE ANY PROGRAM-OWNED ACCOUNT AND RECLAIM RENT
// ═══════════════════════════════════════════════════════════════

/// Admin-only instruction to close any account owned by this program.
/// Transfers ALL lamports from the target account to the admin wallet.
/// Zeros out the account data so the runtime garbage-collects it.
///
/// Accounts: [0] target account (writable), [1] admin (signer, writable)
///
/// The admin must be the DEV_WALLET. This works for:
/// - State accounts (player sessions)
/// - Worker accounts
/// - Shard accounts (model weights)
/// - Pot PDA
/// - Any other program-owned account
fn admin_close_account(accounts: &[AccountInfo]) -> ProgramResult {
    let accs = &mut accounts.iter();
    let target = next_account_info(accs)?;
    let admin = next_account_info(accs)?;

    // Only the dev wallet can close accounts
    if admin.key.to_bytes() != DEV_WALLET {
        msg!("Only admin can close accounts");
        return Err(ProgramError::InvalidAccountData);
    }
    if !admin.is_signer {
        return Err(ProgramError::MissingRequiredSignature);
    }

    // Transfer all lamports from target to admin
    let target_lamports = target.lamports();
    if target_lamports > 0 {
        **target.try_borrow_mut_lamports()? -= target_lamports;
        **admin.try_borrow_mut_lamports()? += target_lamports;
    }

    // Zero out the data so the account gets garbage collected
    let mut data = target.try_borrow_mut_data()?;
    for byte in data.iter_mut() {
        *byte = 0;
    }

    msg!("Admin closed account: {} | reclaimed {} lamports ({} SOL)",
        target.key, target_lamports, target_lamports as f64 / 1_000_000_000.0);
    Ok(())
}
