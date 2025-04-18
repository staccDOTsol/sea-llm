import React, { useRef, useState } from "react";
import {
  Connection,
  PublicKey,
  SystemProgram,
} from "@solana/web3.js";
import { AnchorProvider, Program, Idl } from "@coral-xyz/anchor";
import { Buffer } from "buffer";
import idl from "./sea_nn.json";
import clsx from "clsx";
import { WalletContextProvider } from "./components/WalletProvider";
import { useAnchorWallet, useWallet } from "@solana/wallet-adapter-react";
import { WalletMultiButton } from "@solana/wallet-adapter-react-ui";
// Cast the imported JSON as Idl type with required properties
const IDL = {
  "address": "CG3rq4URcAwEUNnfGZ3YHRKTZAZgyemttN1BQwypa8qj",
  "metadata": {
    "name": "sea_nn",
    "version": "0.1.0",
    "spec": "0.1.0",
    "description": "Created with Anchor"
  },
  "instructions": [
    {
      "name": "chat",
      "discriminator": [
        189,
        34,
        82,
        12,
        30,
        178,
        146,
        97
      ],
      "accounts": [
        {
          "name": "signer",
          "writable": true,
          "signer": true
        },
        {
          "name": "chat_state",
          "writable": true,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "value": [
                  99,
                  104,
                  97,
                  116
                ]
              },
              {
                "kind": "account",
                "path": "signer"
              }
            ]
          }
        },
        {
          "name": "model_registry"
        },
        {
          "name": "embedding_chunk"
        },
        {
          "name": "hidden_chunk"
        },
        {
          "name": "ln_gamma"
        },
        {
          "name": "ln_beta"
        },
        {
          "name": "output_chunk"
        }
      ],
      "args": [
        {
          "name": "input_text",
          "type": "bytes"
        },
        {
          "name": "input_length",
          "type": "u32"
        }
      ]
    },
    {
      "name": "initialize_chat",
      "discriminator": [
        4,
        81,
        179,
        155,
        220,
        138,
        192,
        182
      ],
      "accounts": [
        {
          "name": "signer",
          "writable": true,
          "signer": true
        },
        {
          "name": "chat",
          "writable": true,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "value": [
                  99,
                  104,
                  97,
                  116
                ]
              },
              {
                "kind": "account",
                "path": "signer"
              }
            ]
          }
        },
        {
          "name": "model"
        },
        {
          "name": "system_program",
          "address": "11111111111111111111111111111111"
        }
      ],
      "args": []
    },
    {
      "name": "initialize_model",
      "discriminator": [
        150,
        196,
        232,
        118,
        138,
        43,
        140,
        244
      ],
      "accounts": [
        {
          "name": "signer",
          "writable": true,
          "signer": true
        },
        {
          "name": "registry",
          "writable": true,
          "signer": true
        },
        {
          "name": "system_program",
          "address": "11111111111111111111111111111111"
        }
      ],
      "args": [
        {
          "name": "vocab_size",
          "type": "u32"
        },
        {
          "name": "embedding_dim",
          "type": "u32"
        },
        {
          "name": "hidden_dim",
          "type": "u32"
        },
        {
          "name": "context_length",
          "type": "u32"
        },
        {
          "name": "layer_count",
          "type": "u32"
        }
      ]
    },
    {
      "name": "upload_chunk",
      "discriminator": [
        130,
        219,
        165,
        153,
        119,
        149,
        252,
        162
      ],
      "accounts": [
        {
          "name": "signer",
          "writable": true,
          "signer": true
        },
        {
          "name": "registry",
          "writable": true
        },
        {
          "name": "chunk",
          "writable": true,
          "signer": true
        },
        {
          "name": "system_program",
          "address": "11111111111111111111111111111111"
        }
      ],
      "args": [
        {
          "name": "chunk_index",
          "type": "u32"
        },
        {
          "name": "chunk_type",
          "type": "u8"
        },
        {
          "name": "data",
          "type": {
            "vec": "i32"
          }
        }
      ]
    }
  ],
  "accounts": [
    {
      "name": "ChatState",
      "discriminator": [
        67,
        109,
        181,
        186,
        97,
        123,
        128,
        236
      ]
    },
    {
      "name": "ModelChunk",
      "discriminator": [
        254,
        185,
        27,
        224,
        107,
        118,
        26,
        25
      ]
    },
    {
      "name": "ModelRegistry",
      "discriminator": [
        174,
        72,
        180,
        46,
        185,
        165,
        246,
        200
      ]
    }
  ],
  "errors": [
    {
      "code": 6000,
      "name": "E000",
      "msg": "Invalid authority"
    },
    {
      "code": 6001,
      "name": "E001",
      "msg": "Input too long"
    }
  ],
  "types": [
    {
      "name": "ChatState",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "authority",
            "type": "pubkey"
          },
          {
            "name": "model",
            "type": "pubkey"
          },
          {
            "name": "history",
            "type": "bytes"
          },
          {
            "name": "history_len",
            "type": "u32"
          }
        ]
      }
    },
    {
      "name": "ModelChunk",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "authority",
            "type": "pubkey"
          },
          {
            "name": "registry",
            "type": "pubkey"
          },
          {
            "name": "chunk_index",
            "type": "u32"
          },
          {
            "name": "chunk_type",
            "type": "u8"
          },
          {
            "name": "data",
            "type": {
              "vec": "i32"
            }
          }
        ]
      }
    },
    {
      "name": "ModelRegistry",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "authority",
            "type": "pubkey"
          },
          {
            "name": "chunk_count",
            "type": "u32"
          },
          {
            "name": "vocab_size",
            "type": "u32"
          },
          {
            "name": "embedding_dim",
            "type": "u32"
          },
          {
            "name": "hidden_dim",
            "type": "u32"
          },
          {
            "name": "context_length",
            "type": "u32"
          },
          {
            "name": "layer_count",
            "type": "u32"
          }
        ]
      }
    }
  ]
} as Idl

interface Message {
  text: string;
  sender: 'user' | 'assistant';
}

const PROGRAM_ID = new PublicKey(
  "CG3rq4URcAwEUNnfGZ3YHRKTZAZgyemttN1BQwypa8qj"
);
const MODEL_ID = new PublicKey("5oBPsE6GwscJhzATvekHcALLp4Fvb3q9uPFyxYCRFznh"); // devnet

const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

// Polyfill Buffer for the browser
if (typeof window !== 'undefined') {
  window.Buffer = window.Buffer || Buffer;
}

const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    { sender: 'assistant', text: 'Welcome to Sea-NN! I\'m an on-chain language model running on Solana. How can I help you today?' }
  ]);
  const [inputText, setInputText] = useState('');
  const [chatStatePda, setChatStatePda] = useState<PublicKey | null>(null);
  const [isTyping, setIsTyping] = useState(false);
  const chatRef = useRef<HTMLDivElement>(null);
  const wallet = useAnchorWallet();
  const sendMessage = async (message: string) => {
    setMessages([...messages, { text: message, sender: 'user' }]);
    setInputText('');
    setIsTyping(true);

    try {
      if (!wallet) {
        alert('Please connect your wallet');
        return;
      }
      const connection = new Connection('', 'confirmed');
      const provider = new AnchorProvider(
        connection,
        wallet,
        AnchorProvider.defaultOptions()
      );
      const program = new Program(IDL, provider);
      const chatStatePda = await connection.getAccountInfo(
        PublicKey.findProgramAddressSync(
          [Buffer.from('chat'), wallet.publicKey.toBuffer()],
          PROGRAM_ID
        )[0]
      );
      // Initialize chat if needed
      if (!chatStatePda) {
        const [chatState] = PublicKey.findProgramAddressSync(
          [Buffer.from('chat'), wallet.publicKey.toBuffer()],
          PROGRAM_ID
        );
        setChatStatePda(chatState);

        await program.methods
          .initializeChat()
          .accounts({
            signer: wallet.publicKey,
            chat: chatState,
            model: MODEL_ID,
            systemProgram: SystemProgram.programId,
          })
          .rpc();
        
        // Wait for transaction to confirm
        const wait = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
        await wait(2000);
      }

      // Get model chunk accounts
      const embeddingChunk = PublicKey.findProgramAddressSync(
        [Buffer.from('chunk'), MODEL_ID.toBuffer(), Buffer.from([0])],
        PROGRAM_ID
      )[0];
      const hiddenChunk = PublicKey.findProgramAddressSync(
        [Buffer.from('chunk'), MODEL_ID.toBuffer(), Buffer.from([1])],
        PROGRAM_ID
      )[0];
      const lnGammaChunk = PublicKey.findProgramAddressSync(
        [Buffer.from('chunk'), MODEL_ID.toBuffer(), Buffer.from([2])],
        PROGRAM_ID
      )[0];
      const lnBetaChunk = PublicKey.findProgramAddressSync(
        [Buffer.from('chunk'), MODEL_ID.toBuffer(), Buffer.from([3])],
        PROGRAM_ID
      )[0];
      const outputChunk = PublicKey.findProgramAddressSync(
        [Buffer.from('chunk'), MODEL_ID.toBuffer(), Buffer.from([4])],
        PROGRAM_ID
      )[0];

      const tx = await program.methods
        .chat(
          Buffer.from(message),
          message.length
        )
        .accounts({
          signer: wallet.publicKey,
          chatState: PublicKey.findProgramAddressSync(
            [Buffer.from('chat'), wallet.publicKey.toBuffer()],
            PROGRAM_ID
          )[0],
          modelRegistry: MODEL_ID,
          embeddingChunk: embeddingChunk,
          hiddenChunk: hiddenChunk,  
          lnGamma: lnGammaChunk,
          lnBeta: lnBetaChunk,
        outputChunk: outputChunk,
        })
        .rpc();

      // Get the response from the transaction logs
      const txInfo = await connection.getTransaction(tx, { commitment: 'confirmed' });
      const logs = txInfo?.meta?.logMessages || [];
      const responseLog = logs.find(log => log.startsWith('LLM RESPONSE:'));
      
      if (responseLog) {
        const response = decodeURIComponent(responseLog.split('LLM RESPONSE:')[1]);
        setMessages(prev => [...prev, { text: response, sender: 'assistant' }]);
      } else {
        setMessages(prev => [...prev, { text: 'No response received', sender: 'assistant' }]);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prev => [...prev, { text: 'Error: Failed to get response', sender: 'assistant' }]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
      <div className="min-h-screen bg-gray-100">
        <div className="flex flex-col items-center min-h-screen w-full text-black bg-neutral-50">
          <div className={clsx("flex flex-row items-center justify-between w-full px-8 mt-10 mb-10")}>
            <div className="flex flex-row items-center">
              <img src="./horsea.png" className={clsx("h-24 w-24")} alt="Sea-NN Logo" />
              <div className={clsx("flex flex-col ml-6")}>
                <h1 className={clsx("font-bold text-3xl")}>Sea-NN</h1>
                <h2 className={clsx("text-xl")}>
                  On-Chain Language Model
                </h2>
              </div>
            </div>
            <div className="flex items-center">
              <span className="text-2xl text-blue-500">🤖</span>
            </div>
          </div>

          <div className="w-full max-w-4xl px-4">
            <div className="bg-white rounded-lg shadow-lg p-6">
              <div 
                ref={chatRef}
                className="h-[600px] overflow-y-auto mb-4 space-y-4"
              >
                {messages.map((msg, i) => (
                  <div
                    key={i}
                    className={clsx(
                      "p-3 rounded-lg max-w-[80%]",
                      msg.sender === 'user' 
                        ? "ml-auto bg-blue-100" 
                        : "mr-auto bg-gray-100"
                    )}
                  >
                    {msg.text}
                  </div>
                ))}
                {isTyping && (
                  <div className="mr-auto bg-gray-100 p-3 rounded-lg">
                    <span className="animate-pulse">...</span>
                  </div>
                )}
              </div>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && sendMessage(inputText)}
                  placeholder="Type your message..."
                  className="flex-1 p-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400"
                />
                <button
                  onClick={() => sendMessage(inputText)}
                  disabled={isTyping}
                  className={clsx(
                    "px-4 py-2 rounded-lg",
                    "text-black bg-blue-200 border border-blue-400",
                    "hover:bg-blue-300 disabled:opacity-50"
                  )}
                >
                  ➤
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
  );
}

export default App;
