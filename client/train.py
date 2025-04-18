#!/usr/bin/env python3
import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import pickle
from pathlib import Path
import random
from tqdm import tqdm
from model import SeaLLM, create_and_save_model

class SimpleTokenizer:
    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
        
    def encode(self, text):
        """Convert text to token IDs (ASCII values)"""
        return [ord(c) % self.vocab_size for c in text]
        
    def decode(self, token_ids):
        """Convert token IDs back to text"""
        return ''.join([chr(i) for i in token_ids])
    
    def __len__(self):
        return self.vocab_size

class TextDataset(Dataset):
    def __init__(self, data, context_length=128, stride=64):
        self.data = data
        self.context_length = context_length
        self.stride = stride
        
    def __len__(self):
        return max(1, (len(self.data) - self.context_length) // self.stride)
    
    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.context_length + 1  # +1 for the target
        
        if end_idx > len(self.data):
            end_idx = len(self.data)
            start_idx = max(0, end_idx - self.context_length - 1)
        
        chunk = self.data[start_idx:end_idx]
        
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y

class SimpleModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, context_length, dropout=0.1):
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
            dim_feedforward=4 * d_model,
            dropout=dropout,
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
    
    def get_weights_dict(self):
        """Extract weights as dictionary for quantization"""
        weights = {}
        
        # Token embedding
        weights['token_embedding.weight'] = self.token_embedding.weight.detach().cpu().numpy()
        
        # Position embedding
        weights['position_embedding.weight'] = self.position_embedding.weight.detach().cpu().numpy()
        
        # Transformer layers
        for i, layer in enumerate(self.transformer.layers):
            # Self-attention
            weights[f'layer_{i}.self_attn.in_proj_weight'] = layer.self_attn.in_proj_weight.detach().cpu().numpy()
            if layer.self_attn.in_proj_bias is not None:
                weights[f'layer_{i}.self_attn.in_proj_bias'] = layer.self_attn.in_proj_bias.detach().cpu().numpy()
            weights[f'layer_{i}.self_attn.out_proj.weight'] = layer.self_attn.out_proj.weight.detach().cpu().numpy()
            weights[f'layer_{i}.self_attn.out_proj.bias'] = layer.self_attn.out_proj.bias.detach().cpu().numpy()
            
            # Feedforward
            weights[f'layer_{i}.linear1.weight'] = layer.linear1.weight.detach().cpu().numpy()
            weights[f'layer_{i}.linear1.bias'] = layer.linear1.bias.detach().cpu().numpy()
            weights[f'layer_{i}.linear2.weight'] = layer.linear2.weight.detach().cpu().numpy()
            weights[f'layer_{i}.linear2.bias'] = layer.linear2.bias.detach().cpu().numpy()
            
            # Layer norms
            weights[f'layer_{i}.norm1.weight'] = layer.norm1.weight.detach().cpu().numpy()
            weights[f'layer_{i}.norm1.bias'] = layer.norm1.bias.detach().cpu().numpy()
            weights[f'layer_{i}.norm2.weight'] = layer.norm2.weight.detach().cpu().numpy()
            weights[f'layer_{i}.norm2.bias'] = layer.norm2.bias.detach().cpu().numpy()
        
        # Output layer
        weights['output.weight'] = self.output.weight.detach().cpu().numpy()
        weights['output.bias'] = self.output.bias.detach().cpu().numpy()
        
        return weights

def load_dataset(file_path):
    """Load and preprocess the dataset"""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Create tokenizer and encode text
    tokenizer = SimpleTokenizer()
    encoded = tokenizer.encode(text)
    
    return encoded, tokenizer

def train_model(args):
    """Train the model with the specified parameters"""
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    data, tokenizer = load_dataset(args.dataset)
    print(f"Dataset loaded: {len(data)} tokens")
    
    # Create dataset and dataloader
    dataset = TextDataset(data, context_length=args.context_length, stride=args.stride)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Create model
    print("Initializing model...")
    model = SimpleModel(
        vocab_size=len(tokenizer),
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        context_length=args.context_length,
        dropout=args.dropout
    )
    
    # Save model configuration
    config = {
        "vocab_size": len(tokenizer),
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "context_length": args.context_length,
        "dropout": args.dropout
    }
    
    with open(os.path.join(args.output_dir, "model_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Set device
    device = get_device(args)
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Set up optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Set up loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("Starting training...")
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits = model(x)
            loss = criterion(logits.view(-1, len(tokenizer)), y.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # Update weights
            optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            # Update progress bar
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")
        
        # End of epoch
        epoch_loss = total_loss / len(dataloader)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} completed in {epoch_time:.2f}s | Loss: {epoch_loss:.4f}")
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print(f"New best model saved (loss: {best_loss:.4f})")
            
            # Save model weights as numpy arrays
            weights_dict = model.get_weights_dict()
            weights_path = os.path.join(args.output_dir, "model.npy")
            np.save(weights_path, weights_dict)
            
            # Also save the PyTorch model for later use
            torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
        
        # Update learning rate
        scheduler.step()
    
    print("Training completed!")
    print(f"Model saved to {args.output_dir}")
    
    return model, tokenizer, config

def get_device(args):
    if args.cpu:
        return torch.device("cpu")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def main():
    # Model parameters
    params = {
        'vocab_size': 50257,  # GPT-2 vocabulary size
        'embedding_dim': 768,  # Size of embeddings
        'hidden_dim': 768,    # Size of hidden layers
        'context_length': 1024, # Maximum sequence length
        'layer_count': 12,    # Number of transformer layers
        'num_heads': 12       # Number of attention heads
    }
    
    # Create output directory if it doesn't exist
    os.makedirs('models/test_model_quantized', exist_ok=True)
    
    # Initialize and save model
    create_and_save_model(
        output_path='models/test_model_quantized/weights.json',
        **params
    )
    print("Model initialized and saved successfully!")

if __name__ == "__main__":
    main() 