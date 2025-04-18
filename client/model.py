#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=12):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        
    def forward(self, x, mask=None):
        attention_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attention_out)
        feed_forward_out = self.feed_forward(x)
        x = self.norm2(x + feed_forward_out)
        return x

class SeaLLM(nn.Module):
    def __init__(self, vocab_size=50257, embedding_dim=768, hidden_dim=768, context_length=1024, num_layers=12):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.context_length = context_length
        self.num_layers = num_layers
        
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_length, embedding_dim)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim) for _ in range(num_layers)
        ])
        
        self.output = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        # Create position indices
        positions = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
        
        # Get embeddings
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        
        # Combine embeddings
        x = token_emb + pos_emb
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
            
        # Get logits
        logits = self.output(x)
        return logits

    def export_weights(self, path):
        """Export weights in the format expected by the Solana contract"""
        weights = []
        
        # Helper to convert tensor to u16 list
        def to_u16_list(tensor):
            # Scale to u16 range and convert to integers
            tensor = tensor.detach().cpu().numpy()
            tensor = (tensor * 32768).astype(np.int16)
            return tensor.flatten().tolist()
        
        # Export embeddings
        weights.extend(to_u16_list(self.token_embedding.weight))
        weights.extend(to_u16_list(self.position_embedding.weight))
        
        # Export transformer blocks
        for block in self.transformer_blocks:
            # Attention weights
            weights.extend(to_u16_list(block.attention.in_proj_weight))
            weights.extend(to_u16_list(block.attention.in_proj_bias))
            weights.extend(to_u16_list(block.attention.out_proj.weight))
            weights.extend(to_u16_list(block.attention.out_proj.bias))
            
            # Layer norm weights
            weights.extend(to_u16_list(block.norm1.weight))
            weights.extend(to_u16_list(block.norm1.bias))
            weights.extend(to_u16_list(block.norm2.weight))
            weights.extend(to_u16_list(block.norm2.bias))
            
            # Feed forward weights
            weights.extend(to_u16_list(block.feed_forward[0].weight))
            weights.extend(to_u16_list(block.feed_forward[0].bias))
            weights.extend(to_u16_list(block.feed_forward[2].weight))
            weights.extend(to_u16_list(block.feed_forward[2].bias))
        
        # Export output layer
        weights.extend(to_u16_list(self.output.weight))
        weights.extend(to_u16_list(self.output.bias))
        
        # Save to file
        with open(path, 'w') as f:
            json.dump({"weights": weights}, f)

def create_and_save_model(path):
    """Create a model and save its weights"""
    model = SeaLLM()
    
    # Initialize with small random weights
    for param in model.parameters():
        nn.init.normal_(param, mean=0.0, std=0.02)
    
    # Export weights
    model.export_weights(path)
    return model

if __name__ == "__main__":
    create_and_save_model("models/test_model_quantized/weights.json") 