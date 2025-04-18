use anchor_lang::prelude::*;
use anchor_lang::solana_program;
use anchor_spl::associated_token;
use anchor_spl::token;
use std::convert::TryFrom;

declare_id!("CG3rq4URcAwEUNnfGZ3YHRKTZAZgyemttN1BQwypa8qj");

const VOCAB_SIZE: usize = 256;
const MAX_SEQ_LEN: usize = 128;
use std::ops::AddAssign;
#[account]
pub struct ModelRegistry {
    pub authority: Pubkey,
    pub chunk_count: u32,
    pub vocab_size: u32,
    pub embedding_dim: u32,
    pub hidden_dim: u32,
    pub context_length: u32,
    pub layer_count: u32,
}

#[account]
pub struct ModelChunk {
    pub authority: Pubkey,
    pub registry: Pubkey,
    pub chunk_index: u32,
    pub chunk_type: u8, // 0 = embedding, 1 = hidden, 2 = output
    pub data: Vec<i32>,
}

#[account]
pub struct ChatState {
    pub authority: Pubkey,
    pub model: Pubkey,
    pub history: Vec<u8>,
    pub history_len: u32,
}

// Helper functions for inference
fn softmax(x: &mut [f32]) {
    let max = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let sum: f32 = x.iter_mut()
        .map(|xi| {
            *xi = (*xi - max).exp();
            *xi
        })
        .sum();
    x.iter_mut().for_each(|xi| *xi /= sum);
}

fn layer_norm(x: &mut [f32], gamma: &[i32], beta: &[i32], dim: usize) {
    let mean: f32 = x.iter().sum::<f32>() / dim as f32;
    let var: f32 = x.iter()
        .map(|&xi| (xi - mean).powi(2))
        .sum::<f32>() / dim as f32;
    let std = (var + 1e-5).sqrt();
    
    for i in 0..dim {
        x[i] = ((x[i] - mean) / std) * (gamma[i] as f32 / 32768.0) + (beta[i] as f32 / 32768.0);
    }
}

pub fn initialize_model_handler(
    ctx: Context<InitializeModel>,
    vocab_size: u32,
    embedding_dim: u32,
    hidden_dim: u32,
    context_length: u32,
    layer_count: u32,
) -> Result<()> {
    let registry = &mut ctx.accounts.registry;
    registry.authority = ctx.accounts.signer.key();
    registry.chunk_count = 0;
    registry.vocab_size = vocab_size;
    registry.embedding_dim = embedding_dim;
    registry.hidden_dim = hidden_dim;
    registry.context_length = context_length;
    registry.layer_count = layer_count;
    Ok(())
}

pub fn upload_chunk_handler(
    ctx: Context<UploadChunk>,
    chunk_index: u32,
    chunk_type: u8,
    data: Vec<i32>,
) -> Result<()> {
    let registry = &mut ctx.accounts.registry;
    let chunk = &mut ctx.accounts.chunk;

    require!(registry.authority == ctx.accounts.signer.key(), ProgramError::E000);

    chunk.authority = ctx.accounts.signer.key();
    chunk.registry = registry.key();
    chunk.chunk_index = chunk_index;
    chunk.chunk_type = chunk_type;
    chunk.data = data;

    registry.chunk_count = registry.chunk_count.max(chunk_index + 1);
    Ok(())
}

pub fn initialize_chat_handler(ctx: Context<InitializeChat>) -> Result<()> {
    let chat = &mut ctx.accounts.chat;
    chat.authority = ctx.accounts.signer.key();
    chat.model = ctx.accounts.model.key();
    chat.history = vec![0; 128];
    chat.history_len = 0;
    Ok(())
}
pub fn close_given_account_handler(ctx: Context<CloseGivenAccount>) -> Result<()> {
    Ok(())
}
pub fn close_given_account(ctx: Context<CloseGivenAccount>) -> Result<()> {
    Ok(())
}
pub fn chat_handler(
    ctx: Context<Chat>,
    input_text: Vec<u8>,
    input_length: u32,
) -> Result<()> {
    let chat_state = &mut ctx.accounts.chat_state;
    require!(chat_state.authority == ctx.accounts.signer.key(), ProgramError::E000);
    require!(input_length <= MAX_SEQ_LEN as u32, ProgramError::E001);

    // Copy input text to history
    chat_state.history = input_text.clone();
    chat_state.history_len = input_length;

    // Load model parameters
    let registry = &ctx.accounts.model_registry;
    let embedding_dim = registry.embedding_dim as usize;
    let hidden_dim = registry.hidden_dim as usize;
    
    // Initialize state
    let mut hidden = vec![0.0f32; hidden_dim];
    let mut logits = vec![0.0f32; VOCAB_SIZE];
    
    // Process input sequence
    for &token in input_text.iter().take(input_length as usize) {
        // Embedding lookup
        let embedding = &ctx.accounts.embedding_chunk.data;
        for i in 0..embedding_dim {
            hidden[i] = embedding[token as usize * embedding_dim + i] as f32 / 32768.0;
        }
        
        // Hidden layer
        let weights = &ctx.accounts.hidden_chunk.data;
        let mut new_hidden = vec![0.0f32; hidden_dim];
        
        for i in 0..hidden_dim {
            let mut sum = 0.0f32;
            for j in 0..hidden_dim {
                sum += hidden[j] * (weights[i * hidden_dim + j] as f32 / 32768.0);
            }
            new_hidden[i] = if sum > 0.0 { sum } else { 0.0 }; // ReLU
        }
        hidden = new_hidden;
        
        // Layer normalization
        layer_norm(&mut hidden, 
                  &ctx.accounts.ln_gamma.data,
                  &ctx.accounts.ln_beta.data,
                  hidden_dim);
    }
    
    // Output layer
    let output_weights = &ctx.accounts.output_chunk.data;
    for i in 0..VOCAB_SIZE {
        let mut sum = 0.0f32;
        for j in 0..hidden_dim {
            sum += hidden[j] * (output_weights[i * hidden_dim + j] as f32 / 32768.0);
        }
        logits[i] = sum;
    }
    
    // Apply softmax
    softmax(&mut logits);
    
    // Generate response
    let mut response = String::new();
    for _ in 0..32 {  // Generate up to 32 tokens
        // Sample from logits
        let mut max_prob = 0.0f32;
        let mut next_token = 0;
        for (i, &prob) in logits.iter().enumerate() {
            if prob > max_prob {
                max_prob = prob;
                next_token = i;
            }
        }
        
        // Convert token to char and append to response
        response.push(next_token as u8 as char);
        
        // Break if we generate end token
        if next_token == 0 {
            break;
        }
    }

    msg!("==== LLM RESPONSE ====");
    msg!(&response);

    Ok(())
}

#[derive(Accounts)]
pub struct InitializeModel<'info> {
    #[account(mut)]
    pub signer: Signer<'info>,
    #[account(init, payer = signer, space = 8 + 32 + 4 + 4 + 4 + 4 + 4 + 4)]
    pub registry: Box<Account<'info, ModelRegistry>>,
    pub system_program: Program<'info, System>,
    #[account(init, payer = signer,seeds = [b"embedding", registry.key().as_ref()], bump, space = 8 + std::mem::size_of::<ModelChunk>() )]
    pub embedding_chunk: Box<Account<'info, ModelChunk>>,
    #[account(init, payer = signer,seeds = [b"hidden", registry.key().as_ref()], bump, space = 8 + std::mem::size_of::<ModelChunk>() )]
    pub hidden_chunk: Box<Account<'info, ModelChunk>>,
    #[account(init, payer = signer,seeds = [b"ln_gamma", registry.key().as_ref()], bump, space = 8 + std::mem::size_of::<ModelChunk>() )]
    pub ln_gamma: Box<Account<'info, ModelChunk>>,
    #[account(init, payer = signer,seeds = [b"ln_beta", registry.key().as_ref()], bump, space = 8 + std::mem::size_of::<ModelChunk>() )]
    pub ln_beta: Box<Account<'info, ModelChunk>>,
    #[account(init, payer = signer,seeds = [b"output", registry.key().as_ref()], bump, space = 8 + std::mem::size_of::<ModelChunk>() )]
    pub output_chunk: Box<Account<'info, ModelChunk>>,
}

#[derive(Accounts)]
pub struct UploadChunk<'info> {
    #[account(mut)]
    pub signer: Signer<'info>,
    #[account(mut)]
    pub registry: Box<Account<'info, ModelRegistry>>,
    #[account( mut, realloc = chunk.to_account_info().data_len() + 8 + 32 + 32 + 4 + 1 + 4 + 128 * 4,realloc::payer = signer, realloc::zero = false)]
    pub chunk: Box<Account<'info, ModelChunk>>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct InitializeChat<'info> {
    #[account(mut)]
    pub signer: Signer<'info>,
    #[account(init, payer = signer, space = 1024*5, seeds = [b"chat", signer.key().as_ref()], bump)]
    pub chat: Account<'info, ChatState>,
    pub model: Account<'info, ModelRegistry>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct CloseGivenAccount<'info> {
    #[account(mut)]
    pub signer: Signer<'info>,
    #[account(mut)]
    pub given_account: AccountInfo<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct Chat<'info> {
    #[account(mut)]
    pub signer: Signer<'info>,
    #[account(mut, seeds = [b"chat", signer.key().as_ref()], bump)]
    pub chat_state: Account<'info, ChatState>,
    pub model_registry: Account<'info, ModelRegistry>,
    pub embedding_chunk: Account<'info, ModelChunk>,
    pub hidden_chunk: Account<'info, ModelChunk>,
    pub ln_gamma: Account<'info, ModelChunk>,
    pub ln_beta: Account<'info, ModelChunk>,
    pub output_chunk: Account<'info, ModelChunk>,
}

#[program]
pub mod sea_nn {
    use super::*;

    pub fn initialize_model(
        ctx: Context<InitializeModel>,
        vocab_size: u32,
        embedding_dim: u32,
        hidden_dim: u32,
        context_length: u32,
        layer_count: u32,
    ) -> Result<()> {
        initialize_model_handler(
            ctx,
            vocab_size,
            embedding_dim,
            hidden_dim,
            context_length,
            layer_count,
        )
    }

    pub fn upload_chunk(
        ctx: Context<UploadChunk>,
        chunk_index: u32,
        chunk_type: u8,
        data: Vec<i32>,
    ) -> Result<()> {
        upload_chunk_handler(ctx, chunk_index, chunk_type, data)
    }
    pub fn close_given_account(ctx: Context<CloseGivenAccount>) -> Result<()> {    
        let mut from_account = &mut ctx.accounts.given_account.to_account_info();
        let mut to_account   = &mut ctx.accounts.signer.to_account_info();
        let amount = from_account.lamports();
        **from_account.lamports.borrow_mut() = 0;
        **to_account.lamports.borrow_mut() += amount;
        close_given_account_handler(ctx)
    }
    pub fn initialize_chat(ctx: Context<InitializeChat>) -> Result<()> {
        initialize_chat_handler(ctx)
    }

    pub fn chat(ctx: Context<Chat>, input_text: Vec<u8>, input_length: u32) -> Result<()> {
        chat_handler(ctx, input_text, input_length)
    }
}

#[error_code]
pub enum ProgramError {
    #[msg("Invalid authority")]
    E000,
    #[msg("Input too long")]
    E001,
}
