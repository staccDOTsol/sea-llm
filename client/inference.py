#!/usr/bin/env python3
import os
import argparse
import json
import numpy as np
import torch
import pickle
from pathlib import Path
import sys

# Add the client directory to the path so we can import the model class
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train import SimpleModel, SimpleTokenizer

def load_model(model_dir, device='cpu'):
    """Load a trained model from the specified directory"""
    # Load model configuration
    config_path = os.path.join(model_dir, "model_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model configuration not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create model with the saved configuration
    model = SimpleModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        context_length=config['context_length'],
        dropout=0.0  # No dropout during inference
    )
    
    # Load model weights
    checkpoint_path = os.path.join(model_dir, "best_model.pt")
    if os.path.exists(checkpoint_path):
        # Load from PyTorch checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Try loading from numpy or pickle files
        weights_npy = os.path.join(model_dir, "model_weights.npy")
        weights_pkl = os.path.join(model_dir, "model_weights.pkl")
        
        if os.path.exists(weights_npy):
            weights_dict = np.load(weights_npy, allow_pickle=True).item()
            load_weights_from_dict(model, weights_dict)
        elif os.path.exists(weights_pkl):
            with open(weights_pkl, 'rb') as f:
                weights_dict = pickle.load(f)
            load_weights_from_dict(model, weights_dict)
        else:
            raise FileNotFoundError(f"No model weights found in {model_dir}")
    
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(vocab_size=config['vocab_size'])
    
    return model, tokenizer, config

def load_weights_from_dict(model, weights_dict):
    """Load weights from dictionary into model"""
    # Token embedding
    if 'token_embedding.weight' in weights_dict:
        model.token_embedding.weight.data = torch.tensor(weights_dict['token_embedding.weight'])
    
    # Position embedding
    if 'position_embedding.weight' in weights_dict:
        model.position_embedding.weight.data = torch.tensor(weights_dict['position_embedding.weight'])
    
    # Transformer layers
    for i, layer in enumerate(model.transformer.layers):
        # Self-attention
        if f'layer_{i}.self_attn.in_proj_weight' in weights_dict:
            layer.self_attn.in_proj_weight.data = torch.tensor(weights_dict[f'layer_{i}.self_attn.in_proj_weight'])
        if f'layer_{i}.self_attn.in_proj_bias' in weights_dict:
            layer.self_attn.in_proj_bias.data = torch.tensor(weights_dict[f'layer_{i}.self_attn.in_proj_bias'])
        if f'layer_{i}.self_attn.out_proj.weight' in weights_dict:
            layer.self_attn.out_proj.weight.data = torch.tensor(weights_dict[f'layer_{i}.self_attn.out_proj.weight'])
        if f'layer_{i}.self_attn.out_proj.bias' in weights_dict:
            layer.self_attn.out_proj.bias.data = torch.tensor(weights_dict[f'layer_{i}.self_attn.out_proj.bias'])
        
        # Feedforward
        if f'layer_{i}.linear1.weight' in weights_dict:
            layer.linear1.weight.data = torch.tensor(weights_dict[f'layer_{i}.linear1.weight'])
        if f'layer_{i}.linear1.bias' in weights_dict:
            layer.linear1.bias.data = torch.tensor(weights_dict[f'layer_{i}.linear1.bias'])
        if f'layer_{i}.linear2.weight' in weights_dict:
            layer.linear2.weight.data = torch.tensor(weights_dict[f'layer_{i}.linear2.weight'])
        if f'layer_{i}.linear2.bias' in weights_dict:
            layer.linear2.bias.data = torch.tensor(weights_dict[f'layer_{i}.linear2.bias'])
        
        # Layer norms
        if f'layer_{i}.norm1.weight' in weights_dict:
            layer.norm1.weight.data = torch.tensor(weights_dict[f'layer_{i}.norm1.weight'])
        if f'layer_{i}.norm1.bias' in weights_dict:
            layer.norm1.bias.data = torch.tensor(weights_dict[f'layer_{i}.norm1.bias'])
        if f'layer_{i}.norm2.weight' in weights_dict:
            layer.norm2.weight.data = torch.tensor(weights_dict[f'layer_{i}.norm2.weight'])
        if f'layer_{i}.norm2.bias' in weights_dict:
            layer.norm2.bias.data = torch.tensor(weights_dict[f'layer_{i}.norm2.bias'])
    
    # Output layer
    if 'output.weight' in weights_dict:
        model.output.weight.data = torch.tensor(weights_dict['output.weight'])
    if 'output.bias' in weights_dict:
        model.output.bias.data = torch.tensor(weights_dict['output.bias'])

def generate_text(model, tokenizer, prompt, max_tokens=100, temperature=0.8, top_k=40, top_p=0.9, device='cpu'):
    """Generate text from a prompt using the model"""
    model.eval()  # Ensure model is in evaluation mode
    
    # Tokenize the prompt
    prompt_tokens = tokenizer.encode(prompt)
    context_length = model.context_length
    
    # Truncate if prompt is too long
    if len(prompt_tokens) > context_length:
        prompt_tokens = prompt_tokens[-context_length:]
    
    # Convert to tensor
    input_tokens = torch.tensor(prompt_tokens, dtype=torch.long).to(device)
    
    # Ensure input is the right shape (add batch dimension if needed)
    if input_tokens.dim() == 1:
        input_tokens = input_tokens.unsqueeze(0)
    
    # Store generated tokens
    generated = prompt_tokens.copy()
    
    # Generate tokens auto-regressively
    with torch.no_grad():
        for _ in range(max_tokens):
            # Forward pass
            if input_tokens.size(1) > context_length:
                # Truncate to context length
                input_tokens = input_tokens[:, -context_length:]
            
            # Get model prediction
            logits = model(input_tokens)
            
            # Get next token logits
            next_token_logits = logits[0, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = torch.topk(next_token_logits, k=top_k)[0][-1]
                next_token_logits[next_token_logits < indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            # Add to generated tokens
            generated.append(next_token)
            
            # Update input tokens for next iteration
            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long).to(device)
            input_tokens = torch.cat([input_tokens, next_token_tensor], dim=1)
    
    # Decode generated tokens
    generated_text = tokenizer.decode(generated)
    
    return generated_text

def interactive_mode(model, tokenizer, config, args):
    """Interactive mode for text generation"""
    print(f"Loaded model with {config['n_layers']} layers, {config['d_model']} dimensions")
    print("Enter prompts for text generation. Type 'exit' to quit.")
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    
    while True:
        try:
            prompt = input("\nPrompt> ")
            if prompt.lower() in ["exit", "quit"]:
                break
            
            # Generate text
            generated = generate_text(
                model, 
                tokenizer, 
                prompt, 
                max_tokens=args.max_tokens, 
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=device
            )
            
            print("\nGenerated text:")
            print("=" * 40)
            print(generated)
            print("=" * 40)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Inference with a trained transformer language model")
    
    # Model loading
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing model files")
    
    # Generation parameters
    parser.add_argument("--prompt", type=str, help="Text prompt for generation (if not using interactive mode)")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling parameter")
    
    # Output options
    parser.add_argument("--output_file", type=str, help="Output file for generated text")
    parser.add_argument("--interactive", action="store_true", help="Interactive generation mode")
    
    # Miscellaneous
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    
    # Load model
    try:
        model, tokenizer, config = load_model(args.model_dir, device)
        print(f"Model loaded from {args.model_dir}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Interactive mode
    if args.interactive:
        interactive_mode(model, tokenizer, config, args)
        return
    
    # Generate from prompt
    if args.prompt:
        generated = generate_text(
            model, 
            tokenizer, 
            args.prompt, 
            max_tokens=args.max_tokens, 
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device
        )
        
        # Print or save output
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(generated)
            print(f"Generated text saved to {args.output_file}")
        else:
            print("\nGenerated text:")
            print("=" * 40)
            print(generated)
            print("=" * 40)
    else:
        # Default to interactive mode if no prompt provided
        print("No prompt provided, entering interactive mode...")
        interactive_mode(model, tokenizer, config, args)

if __name__ == "__main__":
    main() 