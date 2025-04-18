#!/usr/bin/env python3
import os
import argparse
import json
import base64
from pathlib import Path
import subprocess
import time

def load_model_registry(quantized_dir):
    """Load the model registry from the quantized model directory"""
    registry_path = os.path.join(quantized_dir, "model_registry.json")
    with open(registry_path, "r") as f:
        registry = json.load(f)
    return registry

def get_binary_chunks(quantized_dir):
    """Get all binary chunks from the quantized model directory"""
    chunks = {}
    for file in os.listdir(quantized_dir):
        if file.endswith(".bin"):
            file_path = os.path.join(quantized_dir, file)
            with open(file_path, "rb") as f:
                data = f.read()
            chunks[file] = data
    return chunks

def upload_model_chunks(chunks, registry, args):
    """Upload model chunks to the Solana contract"""
    # Update registry with chunk information
    registry["chunks"] = {}
    
    # Track successful uploads
    uploaded_chunks = {}
    
    try:
        # Create model registry on-chain
        print("Creating model registry on-chain...")
        model_type = registry["model_type"]
        version = registry["version"]
        config = registry["config"]
        
        # Convert config to command-line arguments
        config_args = " ".join([f"--{key} {value}" for key, value in config.items()])
        
        cmd = f"sea-nn create-model {model_type} {version} {config_args} --program-id {args.program_id} --keypair {args.keypair}"
        
        if args.dry_run:
            print(f"[DRY RUN] Would execute: {cmd}")
            model_pubkey = "DUMMY_MODEL_PUBKEY_FOR_DRY_RUN"
        else:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error creating model registry: {result.stderr}")
                return False
            
            # Extract model public key from output
            output = result.stdout
            model_pubkey = None
            for line in output.split('\n'):
                if "Model pubkey:" in line:
                    model_pubkey = line.split("Model pubkey:")[1].strip()
                    break
            
            if not model_pubkey:
                print("Failed to extract model public key from output")
                return False
        
        print(f"Created model with pubkey: {model_pubkey}")
        
        # Upload each chunk
        total_chunks = len(chunks)
        for i, (chunk_name, chunk_data) in enumerate(chunks.items()):
            print(f"Uploading chunk {i+1}/{total_chunks}: {chunk_name}")
            
            # Create a base64 encoded version of the binary data
            chunk_b64 = base64.b64encode(chunk_data).decode('ascii')
            
            # Write to temporary file to avoid command line length issues
            temp_file = f"temp_chunk_{i}.b64"
            with open(temp_file, "w") as f:
                f.write(chunk_b64)
            
            # Upload chunk command
            cmd = f"sea-nn upload-chunk {model_pubkey} {i} --data-file {temp_file} --program-id {args.program_id} --keypair {args.keypair}"
            
            if args.dry_run:
                print(f"[DRY RUN] Would execute: {cmd}")
                chunk_pubkey = f"DUMMY_CHUNK_PUBKEY_{i}_FOR_DRY_RUN"
            else:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                # Clean up temporary file
                os.remove(temp_file)
                
                if result.returncode != 0:
                    print(f"Error uploading chunk: {result.stderr}")
                    continue
                
                # Extract chunk public key from output
                output = result.stdout
                chunk_pubkey = None
                for line in output.split('\n'):
                    if "Chunk pubkey:" in line:
                        chunk_pubkey = line.split("Chunk pubkey:")[1].strip()
                        break
                
                if not chunk_pubkey:
                    print(f"Failed to extract chunk public key for {chunk_name}")
                    continue
            
            # Store chunk info in the registry
            uploaded_chunks[chunk_name] = {
                "index": i,
                "pubkey": chunk_pubkey,
                "size": len(chunk_data)
            }
            
            # Respect rate limits for Solana RPC
            if not args.dry_run and i < total_chunks - 1:
                time.sleep(1)
        
        # Update registry with uploaded chunk information
        registry["chunks"] = uploaded_chunks
        registry["model_pubkey"] = model_pubkey
        
        # Save updated registry to file
        registry_path = os.path.join(args.quantized_dir, "model_registry_uploaded.json")
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)
        
        print(f"\nModel upload complete! Total chunks uploaded: {len(uploaded_chunks)}/{total_chunks}")
        print(f"Model pubkey: {model_pubkey}")
        print(f"Updated registry saved to: {registry_path}")
        
        return True
    
    except Exception as e:
        print(f"Error during upload: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Upload quantized model weights to Solana contract")
    parser.add_argument("--quantized_dir", type=str, required=True, help="Directory containing quantized model weights")
    parser.add_argument("--program_id", type=str, required=True, help="Solana program ID for the SeaNN contract")
    parser.add_argument("--keypair", type=str, required=True, help="Path to Solana keypair file")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing them")
    
    args = parser.parse_args()
    
    # Load model registry
    registry = load_model_registry(args.quantized_dir)
    
    # Get binary chunks
    chunks = get_binary_chunks(args.quantized_dir)
    
    # Upload chunks to Solana
    success = upload_model_chunks(chunks, registry, args)
    
    if success:
        print("Model upload completed successfully!")
    else:
        print("Model upload failed. Check the errors above.")

if __name__ == "__main__":
    main() 