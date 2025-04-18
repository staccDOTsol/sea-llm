#!/usr/bin/env python3
import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

class LLMTokenizer:
    def __init__(self, vocab_size=256):  # Reduced vocab size for testing
        self.vocab_size = vocab_size
        
    def encode(self, text):
        """Convert text to token IDs (simple byte-level tokenization)"""
        return [ord(c) % self.vocab_size for c in text]
        
    def decode(self, token_ids):
        """Convert token IDs back to text"""
        return ''.join([chr(i) for i in token_ids])
    
    def __len__(self):
        return self.vocab_size

class LLMDataset(Dataset):
    def __init__(self, data, context_length=128):  # Reduced context length
        self.data = data
        self.context_length = context_length
        
    def __len__(self):
        return max(1, len(self.data) - self.context_length)
    
    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.context_length + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

class LLM(nn.Module):
    def __init__(self, vocab_size=256, d_model=128, n_layers=2, n_heads=4, context_length=128):  # Much smaller architecture
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.context_length = context_length
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional embedding
        self.position_embedding = nn.Embedding(context_length, d_model)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=2 * d_model,  # Reduced feedforward dimension
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output layer
        self.output = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, x):
        # Create position IDs
        seq_length = x.size(1)
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand_as(x)
        
        # Get embeddings
        token_embeds = self.token_embedding(x)
        position_embeds = self.position_embedding(position_ids)
        
        # Combine embeddings
        embeds = token_embeds + position_embeds
        
        # Apply transformer layers
        output = self.transformer(embeds)
        
        # Project to vocabulary
        logits = self.output(output)
        
        return logits

    def export_weights(self, output_path):
        """Export weights in the format expected by the Solana program"""
        weights = []
        
        # Flatten and concatenate all weights
        for name, param in self.named_parameters():
            param_np = param.detach().cpu().numpy().flatten()
            weights.extend(param_np.tolist())
        
        # Convert to uint16
        weights = np.array(weights)
        weights = (weights * 32768).astype(np.int16)  # Scale to int16 range
        weights = weights.tolist()
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump({"weights": weights}, f)

def train_llm(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model with much smaller parameters
    model = LLM(
        vocab_size=256,
        d_model=128,
        n_layers=2,
        n_heads=4,
        context_length=128
    )
    
    # Save model configuration
    config = {
        "model_type": "llm",
        "version": "1.0.0",
        "vocab_size": 256,
        "embedding_dim": 128,
        "hidden_dim": 128,
        "context_length": 128,
        "layer_count": 2
    }
    
    with open(os.path.join(args.output_dir, "model_registry.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create dummy data for testing
    tokenizer = LLMTokenizer()
    dummy_text = "Hello, this is a test." * 100  # Reduced text length
    data = tokenizer.encode(dummy_text)
    
    # Create dataset and dataloader
    dataset = LLMDataset(data, context_length=128)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)  # Increased batch size for faster training
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)  # Increased learning rate
    
    # Training loop
    print("Training model...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (x, y) in enumerate(tqdm(dataloader)):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output.view(-1, tokenizer.vocab_size), y.view(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Early stopping for testing
            if batch_idx >= 10 and args.quick_test:  # Only process 10 batches in test mode
                break
            
        avg_loss = total_loss / (batch_idx + 1)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
    
    # Export weights
    print("Exporting weights...")
    model.export_weights(os.path.join(args.output_dir, "dense_60k_u16.json"))
    print("Done!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="models/test_model_quantized")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--quick_test", action="store_true", help="Run a quick test with minimal data")
    args = parser.parse_args()
    
    train_llm(args)

if __name__ == "__main__":
    main() 