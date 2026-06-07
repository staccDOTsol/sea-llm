#!/usr/bin/env node
/**
 * reclaim-rent.mjs
 * 
 * Finds ALL accounts owned by the Break Solana program and closes them
 * using the admin_close_account instruction (IX 99), reclaiming all rent
 * back to the admin wallet.
 *
 * Prerequisites:
 *   1. Program must be redeployed with the IX_ADMIN_CLOSE instruction
 *   2. npm install @solana/web3.js bs58
 *
 * Usage:
 *   node reclaim-rent.mjs
 */

import { Connection, Keypair, PublicKey, Transaction, TransactionInstruction, sendAndConfirmTransaction } from "@solana/web3.js";
import bs58 from "bs58";

// ═══════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════

const RPC_URL = process.env.RPC_URL || "YOUR_RPC_URL_HERE";
const ADMIN_KEY = process.env.ADMIN_KEY || "YOUR_ADMIN_PRIVATE_KEY_HERE";
const PROGRAM_ID = new PublicKey("Dwu4RUjRPhYkvmvVaY6NdRZMyN5yKuwWR3Vi1SQrwEma");
const IX_ADMIN_CLOSE = 99;

// ═══════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════

async function main() {
  console.log("═══════════════════════════════════════════════");
  console.log("  BREAK SOLANA — RENT RECLAIM TOOL");
  console.log("═══════════════════════════════════════════════\n");

  const connection = new Connection(RPC_URL, "confirmed");
  const admin = Keypair.fromSecretKey(bs58.decode(ADMIN_KEY));

  console.log(`Admin:      ${admin.publicKey.toBase58()}`);
  console.log(`Program:    ${PROGRAM_ID.toBase58()}`);

  const balanceBefore = await connection.getBalance(admin.publicKey);
  console.log(`Balance:    ${(balanceBefore / 1e9).toFixed(6)} SOL\n`);

  // Find all accounts owned by the program
  console.log("Scanning for program-owned accounts...\n");

  const accounts = await connection.getProgramAccounts(PROGRAM_ID, {
    commitment: "confirmed",
  });

  if (accounts.length === 0) {
    console.log("No accounts found. Nothing to reclaim.");
    return;
  }

  console.log(`Found ${accounts.length} account(s):\n`);

  let totalLamports = 0;
  for (const acc of accounts) {
    const lamports = acc.account.lamports;
    totalLamports += lamports;
    console.log(`  ${acc.pubkey.toBase58()}`);
    console.log(`    Data: ${acc.account.data.length} bytes | Lamports: ${lamports} (${(lamports / 1e9).toFixed(6)} SOL)`);
  }

  console.log(`\nTotal to reclaim: ${(totalLamports / 1e9).toFixed(6)} SOL`);
  console.log(`\nClosing accounts...\n`);

  let closed = 0;
  let failed = 0;
  let reclaimedTotal = 0;

  for (const acc of accounts) {
    const pubkey = acc.pubkey;
    const lamports = acc.account.lamports;

    try {
      // Build IX_ADMIN_CLOSE instruction
      const ix = new TransactionInstruction({
        programId: PROGRAM_ID,
        keys: [
          { pubkey: pubkey, isSigner: false, isWritable: true },       // target account
          { pubkey: admin.publicKey, isSigner: true, isWritable: true }, // admin (receives lamports)
        ],
        data: Buffer.from([IX_ADMIN_CLOSE]),
      });

      const tx = new Transaction().add(ix);
      const sig = await sendAndConfirmTransaction(connection, tx, [admin], {
        commitment: "confirmed",
        skipPreflight: false,
      });

      closed++;
      reclaimedTotal += lamports;
      console.log(`  ✓ Closed ${pubkey.toBase58().slice(0, 12)}... | ${(lamports / 1e9).toFixed(6)} SOL | tx: ${sig.slice(0, 20)}...`);
    } catch (err) {
      failed++;
      console.log(`  ✗ FAILED ${pubkey.toBase58().slice(0, 12)}... | Error: ${err.message?.slice(0, 80)}`);
    }

    // Small delay to avoid rate limiting
    await new Promise(r => setTimeout(r, 500));
  }

  const balanceAfter = await connection.getBalance(admin.publicKey);

  console.log(`\n═══════════════════════════════════════════════`);
  console.log(`  RESULTS`);
  console.log(`═══════════════════════════════════════════════`);
  console.log(`  Accounts closed:  ${closed}/${accounts.length}`);
  console.log(`  Failed:           ${failed}`);
  console.log(`  SOL reclaimed:    ${(reclaimedTotal / 1e9).toFixed(6)} SOL`);
  console.log(`  Balance before:   ${(balanceBefore / 1e9).toFixed(6)} SOL`);
  console.log(`  Balance after:    ${(balanceAfter / 1e9).toFixed(6)} SOL`);
  console.log(`  Net gain:         ${((balanceAfter - balanceBefore) / 1e9).toFixed(6)} SOL`);
  console.log(`═══════════════════════════════════════════════\n`);
}

main().catch(err => {
  console.error("Fatal error:", err);
  process.exit(1);
});
