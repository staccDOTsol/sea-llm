#!/usr/bin/env python3
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from pathlib import Path

def quantize_weights(weights, bits=8):
    """Quantize weights to specified number of bits"""
    # Get min and max values
    min_val = weights.min()
    max_val = weights.max()
    
    # Calculate scale and zero point
    scale = (max_val - min_val) / (2**bits - 1)
    zero_point = -min_val / scale
    
    # Quantize
    quantized = np.round((weights - min_val) / scale).astype(np.int32)
    
    return quantized, scale, zero_point

def prepare_model_for_upload(model_name="facebook/opt-125m", output_dir="models/llm_quantized"):
    """Download, quantize, and prepare an LLM for Solana upload"""
    print(f"Downloading model: {model_name}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Get model configuration
    config = {
        "model_type": "llm",
        "version": "1.0",
        "config": {
            "vocab_size": tokenizer.vocab_size,
            "embedding_dim": model.config.hidden_size,
            "hidden_dim": model.config.hidden_size,
            "context_length": model.config.max_position_embeddings,
            "layer_count": model.config.num_hidden_layers,
            "num_attention_heads": model.config.num_attention_heads,
            "intermediate_size": model.config.ffn_dim
        }
    }
    
    # Save model registry
    with open(os.path.join(output_dir, "model_registry.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Process and quantize weights
    chunk_index = 0
    
    # 1. Embedding weights
    embeddings = model.get_input_embeddings().weight.detach().numpy()
    quantized_embeddings, scale, zero_point = quantize_weights(embeddings)
    
    # Save embedding weights
    with open(os.path.join(output_dir, f"chunk_{chunk_index}.bin"), "wb") as f:
        f.write(quantized_embeddings.tobytes())
    chunk_index += 1
    
    # 2. Layer weights
    for layer_idx in range(model.config.num_hidden_layers):
        layer = model.model.decoder.layers[layer_idx]
        
        # Attention weights
        attn_weights = layer.self_attn.q_proj.weight.detach().numpy()
        quantized_attn, scale, zero_point = quantize_weights(attn_weights)
        with open(os.path.join(output_dir, f"chunk_{chunk_index}.bin"), "wb") as f:
            f.write(quantized_attn.tobytes())
        chunk_index += 1
        
        # FFN weights
        ffn_weights = layer.fc1.weight.detach().numpy()
        quantized_ffn, scale, zero_point = quantize_weights(ffn_weights)
        with open(os.path.join(output_dir, f"chunk_{chunk_index}.bin"), "wb") as f:
            f.write(quantized_ffn.tobytes())
        chunk_index += 1
    
    # 3. Output weights
    output_weights = model.get_output_embeddings().weight.detach().numpy()
    quantized_output, scale, zero_point = quantize_weights(output_weights)
    with open(os.path.join(output_dir, f"chunk_{chunk_index}.bin"), "wb") as f:
        f.write(quantized_output.tobytes())
    
    print(f"Model prepared and saved to {output_dir}")
    print(f"Total chunks: {chunk_index + 1}")
    print(f"Vocab size: {config['config']['vocab_size']}")
    print(f"Embedding dim: {config['config']['embedding_dim']}")
    print(f"Hidden dim: {config['config']['hidden_dim']}")
    print(f"Context length: {config['config']['context_length']}")
    print(f"Layer count: {config['config']['layer_count']}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prepare an LLM for Solana upload")
    parser.add_argument("--model_name", type=str, default="facebook/opt-125m", 
                       help="Hugging Face model name to download")
    parser.add_argument("--output_dir", type=str, default="models/llm_quantized",
                       help="Directory to save quantized model")
    
    args = parser.parse_args()
    prepare_model_for_upload(args.model_name, args.output_dir)

if __name__ == "__main__":
    main() 