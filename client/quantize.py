#!/usr/bin/env python3
import os
import argparse
import json
import numpy as np
import pickle
import sys
from pathlib import Path

def load_model(model_path):
    """Load model weights from file"""
    print(f"Loading model from {model_path}")
    
    # Check file extension
    if model_path.endswith('.npy'):
        # Load from NumPy file
        return np.load(model_path, allow_pickle=True).item()
    elif model_path.endswith('.pkl') or model_path.endswith('.pickle'):
        # Load from pickle file
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {model_path}")

def quantize_weights(weights, bits=8):
    """Quantize model weights to lower precision"""
    print(f"Quantizing weights to {bits} bits...")
    
    quantized_weights = {}
    scales = {}
    weights_info = {}
    
    # Calculate total size in bytes
    total_orig_size = 0
    total_quant_size = 0
    
    for name, weight in weights.items():
        print(f"Processing weight: {name} with shape {weight.shape}")
        
        # Original size in bytes (assuming float32)
        orig_size = weight.nbytes
        total_orig_size += orig_size
        
        # Determine scale factor (max absolute value)
        abs_max = np.max(np.abs(weight))
        
        # For 8-bit quantization, scale to int8 range (-128 to 127)
        if bits == 8:
            scale_factor = 127.0 / abs_max if abs_max > 0 else 1.0
            quantized = np.round(weight * scale_factor).astype(np.int8)
            dtype_str = "int8"
        # For 16-bit quantization, scale to int16 range (-32768 to 32767)
        elif bits == 16:
            scale_factor = 32767.0 / abs_max if abs_max > 0 else 1.0
            quantized = np.round(weight * scale_factor).astype(np.int16)
            dtype_str = "int16"
        # For 32-bit quantization, scale to int32 range
        elif bits == 32:
            scale_factor = (2**31 - 1) / abs_max if abs_max > 0 else 1.0
            quantized = np.round(weight * scale_factor).astype(np.int32)
            dtype_str = "int32"
        else:
            raise ValueError(f"Unsupported bit width: {bits}")
        
        # Store quantized weights and scale factor
        quantized_weights[name] = quantized
        scales[name] = float(scale_factor)
        
        # Calculate quantized size in bytes
        quant_size = quantized.nbytes
        total_quant_size += quant_size
        
        # Store weight information
        weights_info[name] = {
            "shape": list(weight.shape),
            "dtype": dtype_str,
            "size_bytes": int(quant_size),
            "original_dtype": str(weight.dtype)
        }
    
    compression_ratio = total_orig_size / total_quant_size if total_quant_size > 0 else 0
    print(f"Original size: {total_orig_size/1024/1024:.2f} MB")
    print(f"Quantized size: {total_quant_size/1024/1024:.2f} MB")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    
    return quantized_weights, scales, weights_info, total_quant_size

def calculate_chunk_sizes(total_size, max_chunk_size=256*512):
    """Calculate how to divide weights into chunks of maximum size"""
    num_chunks = (total_size + max_chunk_size - 1) // max_chunk_size
    chunk_size = (total_size + num_chunks - 1) // num_chunks
    
    # Ensure chunk size is no larger than max_chunk_size
    chunk_size = min(chunk_size, max_chunk_size)
    
    print(f"Total size: {total_size} bytes")
    print(f"Max chunk size: {max_chunk_size} bytes")
    print(f"Number of chunks: {num_chunks}")
    print(f"Chunk size: {chunk_size} bytes")
    
    return num_chunks, chunk_size

def serialize_weights_to_binary(quantized_weights, weights_info, output_dir, max_chunk_size=256*512):
    """Serialize weights to binary files, split into chunks if necessary"""
    print(f"Serializing weights to binary files in {output_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine all weights into a single binary blob
    binary_data = bytearray()
    
    # Sort weight names for consistent ordering
    weight_names = sorted(quantized_weights.keys())
    
    for name in weight_names:
        weight = quantized_weights[name]
        binary_data.extend(weight.tobytes())
    
    # Calculate number of chunks needed
    total_size = len(binary_data)
    num_chunks, chunk_size = calculate_chunk_sizes(total_size, max_chunk_size)
    
    # Write binary data to chunk files
    chunks = {}
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, total_size)
        
        chunk_data = binary_data[start:end]
        chunk_path = os.path.join(output_dir, f"chunk_{i}.bin")
        
        with open(chunk_path, "wb") as f:
            f.write(chunk_data)
        
        chunks[i] = {
            "path": chunk_path,
            "size": len(chunk_data),
            "start": start,
            "end": end
        }
        
        print(f"Wrote chunk {i+1}/{num_chunks}: {len(chunk_data)} bytes")
    
    return chunks, num_chunks

def save_model_registry(output_dir, config, weights_info, scales, chunks, num_chunks):
    """Save model registry information to a JSON file"""
    registry_path = os.path.join(output_dir, "model_registry.json")
    
    registry = {
        "config": config,
        "weights_info": weights_info,
        "scales": scales,
        "chunks": chunks,
        "chunk_count": num_chunks,
    }
    
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    
    print(f"Model registry saved to {registry_path}")
    
    return registry

def main():
    parser = argparse.ArgumentParser(description="Quantize model weights for Solana upload")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model weights file (.npy or .pkl)")
    parser.add_argument("--output_dir", type=str, default="quantized", help="Directory to save quantized model")
    parser.add_argument("--bits", type=int, default=8, choices=[8, 16, 32], help="Quantization bit width (8, 16, or 32)")
    parser.add_argument("--max_chunk_size", type=int, default=256*512, help="Maximum chunk size in bytes")
    parser.add_argument("--config", type=str, default=None, help="Path to model config JSON file")
    
    args = parser.parse_args()
    
    # Load model config if provided
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error loading config file: {str(e)}")
            config = {}
    else:
        config = {}
    
    # Add quantization bits to config
    config["bits"] = args.bits
    
    try:
        # Load model weights
        weights = load_model(args.model_path)
        
        # Quantize weights
        quantized_weights, scales, weights_info, total_size = quantize_weights(weights, args.bits)
        
        # Serialize weights to binary files
        chunks, num_chunks = serialize_weights_to_binary(
            quantized_weights, weights_info, args.output_dir, args.max_chunk_size
        )
        
        # Save model registry
        save_model_registry(args.output_dir, config, weights_info, scales, chunks, num_chunks)
        
        print("Quantization completed successfully!")
        
    except Exception as e:
        print(f"Error during quantization: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 