import { Connection, PublicKey } from "@solana/web3.js";

const RPC = process.env.RPC_URL || "https://api.mainnet-beta.solana.com";
const PROGRAM_ID = new PublicKey("Dwu4RUjRPhYkvmvVaY6NdRZMyN5yKuwWR3Vi1SQrwEma");

const conn = new Connection(RPC, "confirmed");

console.log("Searching for accounts owned by", PROGRAM_ID.toBase58(), "...");

try {
  const accounts = await conn.getProgramAccounts(PROGRAM_ID);
  console.log(`Found ${accounts.length} accounts:`);
  let totalLamports = 0n;
  for (const { pubkey, account } of accounts) {
    const lamports = BigInt(account.lamports);
    totalLamports += lamports;
    console.log(`  ${pubkey.toBase58()} — ${Number(lamports) / 1e9} SOL (${account.data.length} bytes)`);
  }
  console.log(`\nTotal rent locked: ${Number(totalLamports) / 1e9} SOL across ${accounts.length} accounts`);
} catch (e) {
  console.error("Error:", e.message);
}

// Also check the wallet balance
const walletKey = new PublicKey(process.env.WALLET || "YOUR_WALLET_ADDRESS");
const balance = await conn.getBalance(walletKey);
console.log(`\nWallet balance: ${balance / 1e9} SOL`);
