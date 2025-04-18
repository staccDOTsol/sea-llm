import * as anchor from "@coral-xyz/anchor";
import * as web3 from "@solana/web3.js";
import fs from "fs";
import path from "path";
import * as IDL from "../target/idl/sea_nn.json";
import { ComputeBudgetProgram, Keypair } from "@solana/web3.js";

describe("sea_nn", () => {
  // Load keypair from file
  const keypairFile = fs.readFileSync("/Users/jarettdunn/ddgb.json");
  const keypair = web3.Keypair.fromSecretKey(
    Buffer.from(JSON.parse(keypairFile.toString()))
  );

  // Configure the client to use the local cluster with our keypair
  const provider = new anchor.AnchorProvider(
    new web3.Connection("https://mainnet.helius-rpc.com/?api-key=9c2d784e-dde3-4420-bd1c-64bcb41f4c51"),
    new anchor.Wallet(keypair),
    { commitment: "confirmed" }
  );
  
  const program = new anchor.Program(
    IDL as any,
    provider
  );
  const connection = provider.connection;
  const wallet = provider.wallet;
  const publicKey = wallet.publicKey;

  anchor.setProvider(provider);

  const wait = (ms: number) =>
    new Promise((resolve) => setTimeout(resolve, ms));

  const confirm = async (signature: string) =>
    connection.confirmTransaction(
      { ...(await connection.getLatestBlockhash()), signature },
      "confirmed"
    );

  const sendIx = async (
    ix: web3.TransactionInstruction,
    signers: web3.Keypair[] = [],
    debug = true
  ) => {
    while (true) {
      try {
        const tx = new web3.Transaction();
        tx.add(ix);
        tx.recentBlockhash = (await connection.getLatestBlockhash()).blockhash;
        tx.feePayer = publicKey;

      // Sign with all signers including the fee payer
      if (!signers.some(s => s.publicKey.equals(publicKey))) {
        signers.push(keypair);
      }
      tx.sign(...signers);
      
      const sig = await provider.connection.sendRawTransaction(tx.serialize());
      if (debug) {
        console.log((await connection.getTransaction(sig)).meta.logMessages);
      }
      break;
    } catch (e: any) {
      if (e.logs) {
        console.error("Transaction logs:", e.logs);
      } else {
        console.error("Error:", e);
      }
      
    }
  }
  };

  it("Initializes and runs LLM", async () => {
    // Create registry account
    const registry = Keypair.generate();
    console.log("registry pk", registry.publicKey.toBase58());

    // Calculate the space needed for the registry account
    const registrySpace = 8 + // discriminator
      32 + // authority
      4 + // chunk_count
      4 + // vocab_size
      4 + // embedding_dim
      4 + // hidden_dim
      4 + // context_length
      4 + // layer_count
      8; // padding for alignment


    // Load model config from generated model
    const modelConfig = JSON.parse(fs.readFileSync(path.join(__dirname, "../../models/test_model_quantized/model_registry.json"), "utf-8"));
    // Initialize model with parameters from our generated model
    await program.methods
      .initializeModel(
        modelConfig.vocab_size,
        modelConfig.embedding_dim,
        modelConfig.hidden_dim,
        modelConfig.context_length,
        modelConfig.layer_count
      )
      .accounts({
        signer: publicKey,
        registry: registry.publicKey
      })
      .preInstructions([
        ComputeBudgetProgram.setComputeUnitLimit({ units: 1_400_000 })])
      .signers([registry])
      .rpc();

      await new Promise(resolve => setTimeout(resolve, 20000));
    // Load model weights from binary chunk files
    const chunkTypes = [
      { name: 'embedding', type: 0 },
      { name: 'hidden', type: 1 },
      { name: 'ln_gamma', type: 2 },
      { name: 'ln_beta', type: 3 },
      { name: 'output', type: 4 }
    ];

    const chunkSize = 128; // Size of data array in ModelChunk
    
    for (const chunkType of chunkTypes) {
      console.log(`\nUploading ${chunkType.name} chunks...`);
      let chunkIndex = 0;
      
      // Read the binary chunk file
      const chunkPath = path.join(__dirname, `../../models/test_model_quantized/chunk_${chunkType.type}.bin`);
      if (!fs.existsSync(chunkPath)) {
        console.log(`No chunk file found for ${chunkType.name}, skipping...`);
        continue;
      }
      
      const chunkBuffer = fs.readFileSync(chunkPath);
      const typeWeights = Array.from(new Uint16Array(chunkBuffer.buffer));
      const totalChunks = Math.ceil(typeWeights.length / chunkSize);
      
      for (let i = 0; i < typeWeights.length; i += chunkSize) {
        const endIndex = Math.min(i + chunkSize, typeWeights.length);
        const data = typeWeights.slice(i, endIndex);
        while (true) {
          try {
            // Create chunk account
            const chunk = new web3.Keypair();
            console.log(`Uploading ${chunkType.name} chunk ${chunkIndex + 1}/${totalChunks} (${Math.round(((chunkIndex + 1)/totalChunks)*100)}% complete)`);

        await sendIx(
          await program.methods
            .uploadChunk(
              chunkIndex,
              chunkType.type,
              data
            )
            .accounts({
              signer: publicKey,
              registry: registry.publicKey,
              chunk: chunk.publicKey,
              systemProgram: web3.SystemProgram.programId,
            })
            .instruction(),
          [chunk],
          false
        );

        chunkIndex++;
        break;
      } catch (e: any) {
        console.error("Error:", e);
          }

        }
      }
      // Add a small delay between chunks to avoid rate limiting
    }

    // Initialize chat
    const chat = new web3.Keypair();
    
    await sendIx(
      await program.methods
        .initializeChat()
        .accounts({
          signer: publicKey,
          chat: chat.publicKey,
          model: registry.publicKey,
          systemProgram: web3.SystemProgram.programId,
        })
        .instruction(),
      [chat],
      false
    );

    // Send a test message
    const testMessage = "Hello, LLM!";
    const inputText = Array.from(testMessage).map(c => c.charCodeAt(0));

    await sendIx(
      await program.methods
        .chat(inputText, testMessage.length)
        .accounts({
          signer: publicKey,
          chatState: chat.publicKey,
        })
        .instruction(),
      [],
      false
    );
  });
});
