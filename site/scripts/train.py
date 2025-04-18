import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import json

class SmallTransformer(nn.Module):
    def __init__(self, vocab_size=60, embedding_dim=64, hidden_dim=256, context_length=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden = nn.Linear(embedding_dim, hidden_dim)
        self.ln_gamma = nn.Parameter(torch.ones(hidden_dim))
        self.ln_beta = nn.Parameter(torch.zeros(hidden_dim))
        self.output = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.hidden(x)
        x = x * self.ln_gamma + self.ln_beta
        x = self.output(x)
        return x

def quantize_weights(weights, bits=16):
    """Quantize weights to specified number of bits"""
    min_val = weights.min()
    max_val = weights.max()
    scale = (2**bits - 1) / (max_val - min_val)
    quantized = ((weights - min_val) * scale).round().astype(np.uint16)
    return quantized, min_val, max_val

def save_chunk(data, chunk_type, output_dir):
    """Save weights as binary chunk file"""
    output_path = Path(output_dir) / f"chunk_{chunk_type}.bin"
    with open(output_path, 'wb') as f:
        f.write(data.tobytes())

def generate_training_data(vocab_size=60, context_length=128, num_samples=1000):
    """Generate simple training data"""
    data = []
    for _ in range(num_samples):
        # Generate random sequences
        seq = torch.randint(0, vocab_size, (context_length,))
        data.append(seq)
    return torch.stack(data)

def main():
    # Model parameters
    vocab_size = 60
    embedding_dim = 64
    hidden_dim = 256
    context_length = 128
    num_epochs = 100
    batch_size = 32

    # Create model
    model = SmallTransformer(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        context_length=context_length
    )

    # Generate training data
    print("Generating training data...")
    train_data = generate_training_data(vocab_size, context_length)
    
    # Training loop
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            
            # Forward pass
            outputs = model(batch)
            loss = criterion(outputs.view(-1, vocab_size), batch.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_data):.4f}")

    # Save model weights
    print("\nSaving model weights...")
    output_dir = Path("models/test_model_quantized")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract and quantize weights
    weights = {}
    
    # 1. Embedding weights
    embedding_weights = model.embedding.weight.detach().numpy()
    embedding_quantized, embedding_min, embedding_max = quantize_weights(embedding_weights)
    save_chunk(embedding_quantized, 0, output_dir)
    weights['embedding'] = embedding_weights.tolist()
    print(f"Saved embedding weights (min: {embedding_min}, max: {embedding_max})")

    # 2. Hidden weights
    hidden_weights = model.hidden.weight.detach().numpy()
    hidden_quantized, hidden_min, hidden_max = quantize_weights(hidden_weights)
    save_chunk(hidden_quantized, 1, output_dir)
    weights['hidden'] = hidden_weights.tolist()
    print(f"Saved hidden weights (min: {hidden_min}, max: {hidden_max})")

    # 3. Layer norm gamma
    ln_gamma = model.ln_gamma.detach().numpy()
    ln_gamma_quantized, ln_gamma_min, ln_gamma_max = quantize_weights(ln_gamma)
    save_chunk(ln_gamma_quantized, 2, output_dir)
    weights['ln_gamma'] = ln_gamma.tolist()
    print(f"Saved layer norm gamma (min: {ln_gamma_min}, max: {ln_gamma_max})")

    # 4. Layer norm beta
    ln_beta = model.ln_beta.detach().numpy()
    ln_beta_quantized, ln_beta_min, ln_beta_max = quantize_weights(ln_beta)
    save_chunk(ln_beta_quantized, 3, output_dir)
    weights['ln_beta'] = ln_beta.tolist()
    print(f"Saved layer norm beta (min: {ln_beta_min}, max: {ln_beta_max})")

    # 5. Output weights
    output_weights = model.output.weight.detach().numpy()
    output_quantized, output_min, output_max = quantize_weights(output_weights)
    save_chunk(output_quantized, 4, output_dir)
    weights['output'] = output_weights.tolist()
    print(f"Saved output weights (min: {output_min}, max: {output_max})")

    # Save model config and weights
    config = {
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "context_length": context_length,
        "layer_count": 1,
        "quantization": {
            "embedding": {"min": float(embedding_min), "max": float(embedding_max)},
            "hidden": {"min": float(hidden_min), "max": float(hidden_max)},
            "ln_gamma": {"min": float(ln_gamma_min), "max": float(ln_gamma_max)},
            "ln_beta": {"min": float(ln_beta_min), "max": float(ln_beta_max)},
            "output": {"min": float(output_min), "max": float(output_max)}
        }
    }

    with open(output_dir / "model_config.json", "w") as f:
        json.dump(config, f, indent=2)

    with open(output_dir / "dense_60k_u16.json", "w") as f:
        json.dump(weights, f, indent=2)

    print("\nModel training and saving completed successfully!")

if __name__ == "__main__":
    main() 