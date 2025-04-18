// Sea-LLM Chat Client - Interact with an on-chain LLM
const anchor = require('@project-serum/anchor');
const readline = require('readline');
const { SystemProgram } = anchor.web3;

// Configure the LLM program ID
const PROGRAM_ID = 'D3ymjhwwXbJ4eAuHrQABVVD4aGP51cZXHN7cA8eoNXfb';

// Create interface for reading user input
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

async function main() {
  // Configure the client
  const provider = anchor.AnchorProvider.env();
  anchor.setProvider(provider);
  
  try {
    // Load the IDL
    const idl = JSON.parse(require('fs').readFileSync('./contract/target/idl/sea_llm.json', 'utf8'));
    const programId = new anchor.web3.PublicKey(PROGRAM_ID);
    const program = new anchor.Program(idl, programId);

    console.log("🌊 Sea-LLM Chat Interface 🧠");
    console.log("--------------------------------------");
    console.log("Connected with wallet:", provider.wallet.publicKey.toString());

    // Ask the user for the model and chat account addresses
    const modelAddress = await askQuestion("Enter the model account address: ");
    const chatAddress = await askQuestion("Enter the chat account address (or 'new' to create one): ");

    // Load model account
    const modelPublicKey = new anchor.web3.PublicKey(modelAddress);
    let chatPublicKey;
    
    // Create or load chat account
    if (chatAddress.toLowerCase() === 'new') {
      // Create a new chat account
      const chat = anchor.web3.Keypair.generate();
      chatPublicKey = chat.publicKey;
      
      console.log("Creating new chat account:", chatPublicKey.toString());
      
      // Calculate space for chat account
      const chatSpace = 8 + 32 + 32 + (64 * 5) + 4;
      
      // Create chat account transaction
      try {
        const createChatTx = await program.methods
          .createChat()
          .accounts({
            signer: provider.wallet.publicKey,
            chat: chatPublicKey,
            model: modelPublicKey,
            systemProgram: SystemProgram.programId,
          })
          .signers([chat])
          .rpc();
        
        console.log("Chat account created:", createChatTx);
      } catch (error) {
        console.error("Error creating chat account:", error);
        process.exit(1);
      }
    } else {
      // Use existing chat account
      chatPublicKey = new anchor.web3.PublicKey(chatAddress);
    }

    console.log("\nConnected to chat session:", chatPublicKey.toString());
    console.log("Type your messages below (or 'exit' to quit)");
    console.log("--------------------------------------");
    
    // Chat loop
    await chatLoop(program, chatPublicKey, modelPublicKey);
    
  } catch (error) {
    console.error("Error:", error);
  }
  
  rl.close();
}

// Function to handle user chat input and LLM responses
async function chatLoop(program, chatPublicKey, modelPublicKey) {
  while (true) {
    const userMessage = await askQuestion("\nYou: ");
    
    if (userMessage.toLowerCase() === 'exit') {
      console.log("Goodbye!");
      break;
    }
    
    // Convert message to bytes (using ASCII encoding, max 64 bytes)
    const messageBytes = Buffer.from(userMessage.slice(0, 64), 'utf8');
    
    console.log("Sending message to on-chain LLM...");
    
    try {
      // Call the chat_with_llm instruction
      const tx = await program.methods
        .chatWithLlm(messageBytes, messageBytes.length)
        .accounts({
          signer: program.provider.wallet.publicKey,
          chat: chatPublicKey,
          model: modelPublicKey,
        })
        .rpc();
        
      // In a real implementation, we would listen for program logs to get the response
      // For this demo, we're simply confirming the transaction went through
      console.log("Transaction confirmed:", tx);
      
      // To see the actual response, we'd need to parse the program logs
      // For demo, we'll simulate a response
      console.log("\nSea-LLM: This is a simulated response from the on-chain LLM. In a real implementation, the response would be extracted from the program logs.");
      
    } catch (error) {
      console.error("Error sending message:", error);
    }
  }
}

// Helper function to ask questions via readline
function askQuestion(question) {
  return new Promise((resolve) => {
    rl.question(question, (answer) => {
      resolve(answer);
    });
  });
}

// Run the main function
main().catch(console.error); 