import pytest
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

# Add the client directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Import the SeaLLM class directly
from client import SeaLLM

@pytest.fixture
def mock_model():
    model = SeaLLM(
        vocab_size=256,
        embedding_dim=64,
        hidden_dim=128,
        context_length=64,
        layer_count=2
    )
    
    # Initialize some dummy weights
    model.embeddings = np.random.randn(256, 64).astype(np.float32)
    model.attention_weights = [np.random.randn(64, 128).astype(np.float32) for _ in range(2)]
    model.ffn_weights = [np.random.randn(128, 128).astype(np.float32) for _ in range(2)]
    model.output_weights = np.random.randn(128, 256).astype(np.float32)
    
    return model

def test_sea_llm_tokenization(mock_model):
    text = "Hello, World!"
    tokens = mock_model.tokenize(text)
    
    assert len(tokens) == len(text)
    assert all(0 <= t < mock_model.vocab_size for t in tokens)
    
    # Test round trip
    decoded = mock_model.detokenize(tokens)
    assert decoded == text

def test_sea_llm_generation(mock_model):
    prompt = "Hello"
    generated = mock_model.generate(prompt, max_tokens=10)
    
    assert isinstance(generated, str)
    assert len(generated) > len(prompt)
    assert generated.startswith(prompt)

def test_solana_client_initialization(mock_solana_client):
    client = SolanaLLMClient("http://localhost:8899")
    assert client.client == mock_solana_client

def test_model_registry_loading(mock_solana_client):
    client = SolanaLLMClient("http://localhost:8899")
    registry = client.get_model_registry("dummy_address")
    
    assert registry["vocab_size"] == 256
    assert registry["embedding_dim"] == 64
    assert registry["hidden_dim"] == 128
    assert registry["context_length"] == 64
    assert registry["layer_count"] == 2 