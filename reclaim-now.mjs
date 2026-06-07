import { Connection, PublicKey, Keypair, Transaction, TransactionInstruction } from "@solana/web3.js";
import bs58 from "bs58";

const RPC = "https://api.mainnet-beta.solana.com";
const PROGRAM_ID = new PublicKey("Dwu4RUjRPhYkvmwVaY6NdRZMyN5yKuxWR3Vi1SQrwEma");
const ADMIN_KEY = process.env.ADMIN_KEY || "YOUR_BASE58_PRIVATE_KEY_HERE";
const IX_ADMIN_CLOSE = 99;

const conn = new Connection(RPC, "confirmed");
const adminKeypair = Keypair.fromSecretKey(bs58.decode(ADMIN_KEY));

console.log("Admin wallet:", adminKeypair.publicKey.toBase58());
console.log("Program:", PROGRAM_ID.toBase58());
console.log("");

// Find all accounts owned by the program
const accounts = await conn.getProgramAccounts(PROGRAM_ID);
console.log(`Found ${accounts.length} accounts to close.\n`);

let totalReclaimed = 0n;
let closed = 0;

for (const { pubkey, account } of accounts) {
  const lamports = BigInt(account.lamports);
  console.log(`Closing ${pubkey.toBase58()} (${Number(lamports) / 1e9} SOL, ${account.data.length} bytes)...`);
  
  try {
    const ix = new TransactionInstruction({
      programId: PROGRAM_ID,
      keys: [
        { pubkey: pubkey, isSigner: false, isWritable: true },
        { pubkey: adminKeypair.publicKey, isSigner: true, isWritable: true },
      ],
      data: Buffer.from([IX_ADMIN_CLOSE]),
    });

    const tx = new Transaction().add(ix);
    tx.feePayer = adminKeypair.publicKey;
    const { blockhash } = await conn.getLatestBlockhash();
    tx.recentBlockhash = blockhash;

    const sig = await conn.sendTransaction(tx, [adminKeypair], { skipPreflight: false });
    await conn.confirmTransaction(sig, "confirmed");
    
    totalReclaimed += lamports;
    closed++;
    console.log(`  ✓ Closed! Sig: ${sig}`);
  } catch (e) {
    console.error(`  ✗ Failed: ${e.message}`);
  }
}

console.log(`\n========================================`);
console.log(`Closed ${closed}/${accounts.length} accounts`);
console.log(`Total reclaimed: ${Number(totalReclaimed) / 1e9} SOL`);

const balance = await conn.getBalance(adminKeypair.publicKey);
console.log(`Wallet balance now: ${balance / 1e9} SOL`);
