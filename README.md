# Break Solana: Edition 2

An LLM that runs entirely on-chain on Solana — every matrix multiplication, every attention head, every token generation is a Solana transaction.

## What is this?

Break Solana is a GPT-Neo 125M model deployed as a native Solana program. The model weights are stored on-chain as account data, and inference is performed through sequential transactions that execute the transformer forward pass.

**Key stats:**
- ~191 transactions per token generated
- 9.6 MB of model weights stored on-chain
- 8 transformer layers executed as Solana instructions
- Constant-product game mechanics with 1 SOL entry fee

## Repository Structure

```
program/          — Solana program (Rust/BPF)
  src/lib.rs      — Full program source with inference, game logic, and admin close instruction
  target/deploy/  — Compiled .so binary

website/          — Landing page (React + Tailwind)
  client/         — Frontend source

reclaim-rent.mjs  — Script to close all program-owned accounts and reclaim rent
reclaim-now.mjs   — Simplified reclaim script
check-accounts.mjs — Utility to list all program-owned accounts
```

## Program Instructions

| IX Code | Instruction | Description |
|---------|-------------|-------------|
| 0 | `start_session` | Begin inference, pay 1 SOL entry fee |
| 1-19 | `run_layer_*` | Execute transformer layers |
| 20 | `finalize` | Complete token generation |
| 21 | `close_session` | Close a player session |
| 22 | `claim_win` | Claim pot if magic word is generated |
| 99 | `admin_close` | Admin-only: close any account, reclaim rent |

## How the Game Works

1. Player pays 1 SOL (0.8 to pot, 0.2 dev fee)
2. Player provides a prompt (up to 32 tokens)
3. The on-chain LLM generates one token
4. If the generated token matches the secret "magic word" → player wins the entire pot
5. If not → SOL stays in pot, pot grows

## Building

```bash
# Install Solana CLI
sh -c "$(curl -sSfL https://release.anza.xyz/v2.2.12/install)"

# Build the program
cd program
cargo-build-sbf
```

## Deploying

```bash
solana program deploy program/target/deploy/solana_llm.so \
  --program-id <PROGRAM_KEYPAIR> \
  -k <AUTHORITY_KEYPAIR>
```

## Reclaiming Rent

To close all program accounts and reclaim SOL:

```bash
ADMIN_KEY="<your_base58_key>" node reclaim-now.mjs
```

## Website

```bash
cd website
pnpm install
pnpm dev
```

## License

MIT
