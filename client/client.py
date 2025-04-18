#!/usr/bin/env python3
import os
import json
import numpy as np
from solana.rpc.api import Client
from solana.publickey import PublicKey
from solana.transaction import AccountMeta, Transaction
from solana.system_program import SYS_PROGRAM_ID
import base64
from solana.rpc.types import TxOpts

# Basic Transformer/RNN implementation for inference
class SeaLLM:
    def __init__(self, vocab_size, embedding_dim, hidden_dim, context_length, layer_count):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.context_length = context_length
        self.layer_count = layer_count
        
        # Model parameters that will be loaded from on-chain storage
        self.embeddings = None
        self.attention_weights = []
        self.ffn_weights = []
        self.output_weights = None
    
    def load_weights_from_chunks(self, chunks):
        """Load weights from the chunk data"""
        for chunk in chunks:
            chunk_type = chunk["chunk_type"]
            data = np.array(chunk["data"], dtype=np.int32)
            
            if chunk_type == 0:  # Embedding weights
                if self.embeddings is None:
                    self.embeddings = data.reshape(self.vocab_size, self.embedding_dim)
            elif chunk_type == 1:  # Attention weights
                layer_idx = len(self.attention_weights)
                self.attention_weights.append(data.reshape(-1, self.hidden_dim))
            elif chunk_type == 2:  # FFN weights
                layer_idx = len(self.ffn_weights)
                self.ffn_weights.append(data.reshape(-1, self.hidden_dim))
            elif chunk_type == 3:  # Output weights
                if self.output_weights is None:
                    self.output_weights = data.reshape(self.hidden_dim, self.vocab_size)
    
    def tokenize(self, text):
        """Simple character-level tokenization"""
        return [ord(c) % self.vocab_size for c in text]
    
    def detokenize(self, tokens):
        """Convert tokens back to text"""
        return ''.join([chr(t) for t in tokens])
    
    def attention(self, x, weights):
        """Simple attention mechanism"""
        # In a real implementation, this would be proper multi-head attention
        return np.tanh(np.dot(x, weights))
    
    def feed_forward(self, x, weights):
        """Simple feed-forward network"""
        return np.tanh(np.dot(x, weights))
    
    def forward(self, tokens):
        """Forward pass through the model"""
        # Embed input tokens
        x = np.zeros((len(tokens), self.embedding_dim))
        for i, token in enumerate(tokens):
            x[i] = self.embeddings[token]
        
        # Process through layers
        for layer_idx in range(min(self.layer_count, len(self.attention_weights))):
            # Attention layer
            attn = self.attention(x, self.attention_weights[layer_idx])
            
            # Feed-forward layer
            x = self.feed_forward(attn, self.ffn_weights[layer_idx])
        
        # Generate output logits
        logits = np.dot(x[-1], self.output_weights)
        
        # Return the token with the highest probability
        return np.argmax(logits)
    
    def generate(self, prompt, max_tokens=50):
        """Generate text from a prompt"""
        tokens = self.tokenize(prompt)
        
        for _ in range(max_tokens):
            # Only use the last context_length tokens
            context = tokens[-self.context_length:]
            
            # Get the next token
            next_token = self.forward(context)
            tokens.append(next_token)
            
            # Stop if we generate an end token (could define a special token for this)
            if next_token == 0:
                break
        
        return self.detokenize(tokens)


class SolanaLLMClient:
    def __init__(self, rpc_url="http://localhost:8899"):
        self.client = Client(rpc_url)
        self.program_id = PublicKey("CG3rq4URcAwEUNnfGZ3YHRKTZAZgyemttN1BQwypa8qj")
    
    def get_model_registry(self, registry_address):
        """Get model registry data from on-chain"""
        account_info = self.client.get_account_info(registry_address)
        
        if not account_info.value:
            raise ValueError(f"Model registry not found: {registry_address}")
        
        data = account_info.value.data
        
        # Parse registry data
        registry = {
            "authority": PublicKey(data[8:40]),
            "chunk_count": int.from_bytes(data[40:44], byteorder='little'),
            "vocab_size": int.from_bytes(data[44:48], byteorder='little'),
            "embedding_dim": int.from_bytes(data[48:52], byteorder='little'),
            "hidden_dim": int.from_bytes(data[52:56], byteorder='little'),
            "context_length": int.from_bytes(data[56:60], byteorder='little'),
            "layer_count": int.from_bytes(data[60:64], byteorder='little'),
        }
        
        return registry
    
    def get_model_chunks(self, registry_address, chunk_count):
        """Get all model chunks from on-chain"""
        chunks = []
        
        # In a real implementation, we would need to derive the PDA addresses for each chunk
        # For demonstration, we'll just show the concept
        for i in range(chunk_count):
            seed = f"chunk-{i}".encode()
            chunk_address, _ = PublicKey.find_program_address(
                [registry_address.to_bytes(), seed], 
                self.program_id
            )
            
            account_info = self.client.get_account_info(chunk_address)
            if not account_info.value:
                print(f"Warning: Chunk {i} not found")
                continue
            
            data = account_info.value.data
            
            # Parse chunk data
            chunk = {
                "authority": PublicKey(data[8:40]),
                "registry": PublicKey(data[40:72]),
                "chunk_index": int.from_bytes(data[72:76], byteorder='little'),
                "chunk_type": data[76],
                "data": [int.from_bytes(data[80+j:80+j+4], byteorder='little') for j in range(0, len(data)-80, 4)]
            }
            
            chunks.append(chunk)
        
        return chunks
    
    def load_model(self, registry_address):
        """Load a model from on-chain storage"""
        registry_key = PublicKey(registry_address)
        registry = self.get_model_registry(registry_key)
        
        # Create model instance
        model = SeaLLM(
            registry["vocab_size"],
            registry["embedding_dim"],
            registry["hidden_dim"],
            registry["context_length"],
            registry["layer_count"],
        )
        
        # Load chunks
        chunks = self.get_model_chunks(registry_key, registry["chunk_count"])
        model.load_weights_from_chunks(chunks)
        
        return model
    
    def chat(self, model, prompt):
        """Generate a response using the loaded model"""
        return model.generate(prompt)


def main():
    # Initialize client
    client = SolanaLLMClient()
    
    # For demonstration, you would need the actual registry address from deployment
    registry_address = input("Enter model registry address: ")
    
    try:
        # Load model from on-chain storage
        model = client.load_model(registry_address)
        
        print("Model loaded successfully!")
        print(f"Vocab size: {model.vocab_size}")
        print(f"Embedding dim: {model.embedding_dim}")
        print(f"Hidden dim: {model.hidden_dim}")
        print(f"Context length: {model.context_length}")
        print(f"Layer count: {model.layer_count}")
        
        # Chat loop
        while True:
            prompt = input("\nEnter your message (or 'exit' to quit): ")
            if prompt.lower() == 'exit':
                break
            
            print("\nGenerating response...")
            response = client.chat(model, prompt)
            print(f"LLM: {response}")
    
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 