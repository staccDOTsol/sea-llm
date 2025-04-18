import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { Connection, PublicKey } from "@solana/web3.js";
import * as  IDL from "../target/idl/sea_nn.json";

describe("close accounts", () => {
  const provider = new anchor.AnchorProvider(
    new Connection(""),
    new anchor.Wallet(anchor.web3.Keypair.fromSecretKey(new Uint8Array(JSON.parse(require('fs').readFileSync('/', 'utf8'))))),
    anchor.AnchorProvider.defaultOptions()
  );
  const program = anchor.workspace.SeaNn as Program<any>;

  const MODEL_ID = new PublicKey("HU7aembtpriQpx2wruJMoqjrXH3EMMaw1JCshxb74X5R");

  it("Closes all chat accounts", async () => {
    // Get all chat PDAs
    const getAlLProgramAccounts = await new Connection("").getProgramAccounts(new PublicKey("CG3rq4URcAwEUNnfGZ3YHRKTZAZgyemttN1BQwypa8qj"), {
    
    });
    console.log(getAlLProgramAccounts);
    for (const account of getAlLProgramAccounts) {
    try {
      const tx = await program.methods
        .closeGivenAccount()
        .accounts({
          signer: provider.publicKey,
          givenAccount: account.pubkey,
          systemProgram: anchor.web3.SystemProgram.programId,
        })
        .transaction();
        tx.recentBlockhash = (await provider.connection.getLatestBlockhash()).blockhash;
        tx.feePayer = provider.publicKey;
        tx.sign(anchor.web3.Keypair.fromSecretKey(new Uint8Array(JSON.parse(require('fs').readFileSync('/', 'utf8')))));
        const txHash = await provider.connection.sendRawTransaction(tx.serialize());
        console.log("Closed chat account:", txHash);
    } catch (e) {
      console.log("Failed to close chat account:", e);
    }

    }
  });
}); 