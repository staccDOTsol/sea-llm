# Break Solana: Rent Reclaim Instructions

## What was done

A new instruction `IX_ADMIN_CLOSE` (code `99`) was added to the Solana program. This instruction:

- Can only be called by the DEV_WALLET (admin)
- Transfers ALL lamports from any program-owned account to the admin wallet
- Zeros out the account data so the runtime garbage-collects it
- Works on: state accounts, worker accounts, shard accounts, pot PDA — any account owned by the program

## Files

| File | Description |
|------|-------------|
| `program/target/deploy/solana_llm.so` | Rebuilt program binary with close instruction |
| `reclaim-rent.mjs` | Node.js script to find and close all accounts |
| `program/src/lib.rs` | Updated Rust source |

## Step 1: Redeploy the program

```bash
# Set your PATH
export PATH="$HOME/.local/share/solana/install/active_release/bin:$PATH"

# Deploy the updated program (uses your authority keypair)
solana program deploy program/target/deploy/solana_llm.so \
  --program-id Dwu4RUjRPhYkvmvVaY6NdRZMyN5yKuwWR3Vi1SQrwEma \
  --keypair <path-to-authority-keypair.json> \
  --url "https://beta.helius-rpc.com/?api-key=YOUR_KEY"
```

## Step 2: Run the reclaim script

```bash
# Install dependencies
npm install @solana/web3.js bs58

# Set environment variables
export RPC_URL="https://beta.helius-rpc.com/?api-key=YOUR_KEY"
export ADMIN_KEY="your-base58-private-key"

# Run
node reclaim-rent.mjs
```

## Step 3 (Optional): Close the program itself

After reclaiming all account rent, you can also close the program data account:

```bash
solana program close Dwu4RUjRPhYkvmvVaY6NdRZMyN5yKuwWR3Vi1SQrwEma \
  --bypass-warning \
  --keypair <path-to-authority-keypair.json> \
  --url "https://beta.helius-rpc.com/?api-key=YOUR_KEY"
```

This is **irreversible** — the program will no longer be executable.

## How the close instruction works

```
Instruction code: 99
Accounts:
  [0] target account (writable) — the account to close
  [1] admin wallet (signer, writable) — must be DEV_WALLET, receives lamports
```

The instruction verifies the signer matches DEV_WALLET, transfers all lamports from the target to the admin, and zeros the data.
