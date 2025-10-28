import pytest
import torch
from src.embeddings.embedding_layer import EmbeddingLayer

@pytest.fixture
def vocab_size():
    return 10000

@pytest.fixture
def d_model():
    return 512

@pytest.fixture
def max_seq_length():
    return 100

@pytest.fixture
def batch_size():
    return 2

@pytest.fixture
def sequence_length():
    return 10

def test_output_shape(vocab_size, d_model, max_seq_length, batch_size, sequence_length):
    """Test that output shape is correct"""
    # Create EmbeddingLayer instance
    embedding_layer = EmbeddingLayer(vocab_size, d_model, max_seq_length)
    
    # Generate random token IDs
    token_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
    
    # Pass through embedding layer
    embeddings = embedding_layer(token_ids)
    
    # Assert correct output shape
    assert embeddings.shape == (batch_size, sequence_length, d_model)

def test_contains_position_info(vocab_size, d_model):
    """Test that same token at different positions has different embeddings"""
    # Create EmbeddingLayer
    embedding_layer = EmbeddingLayer(vocab_size, d_model)
    
    # Create batch with same token ID at positions 0 and 5
    token_ids = torch.zeros(1, 10, dtype=torch.long)  # All zeros (same token)
    token_ids[0, 0] = 5  # Token 5 at position 0
    token_ids[0, 5] = 5  # Token 5 at position 5
    
    # Get embeddings
    embeddings = embedding_layer(token_ids)
    
    # Assert embeddings at different positions are NOT equal
    assert not torch.allclose(embeddings[0, 0], embeddings[0, 5], rtol=1e-6)

def test_contains_token_info(vocab_size, d_model):
    """Test that different tokens at same position have different embeddings"""
    # Create EmbeddingLayer
    embedding_layer = EmbeddingLayer(vocab_size, d_model)
    
    # Create batch with different token IDs at same position (position 2)
    token_ids = torch.zeros(2, 10, dtype=torch.long)  # Two sequences
    token_ids[0, 2] = 5   # Token 5 at position 2 in first sequence
    token_ids[1, 2] = 10  # Token 10 at position 2 in second sequence
    
    # Get embeddings
    embeddings = embedding_layer(token_ids)
    
    # Assert embeddings for different tokens at same position are NOT equal
    assert not torch.allclose(embeddings[0, 2], embeddings[1, 2], rtol=1e-6)

def test_dropout_applied_during_training(vocab_size, d_model):
    """Test that dropout changes output in training mode"""
    # Create EmbeddingLayer with dropout
    embedding_layer = EmbeddingLayer(vocab_size, d_model, dropout=0.5)
    
    # Set to training mode
    embedding_layer.train()
    
    # Create same token_ids input
    token_ids = torch.randint(0, vocab_size, (1, 5))
    
    # Pass through twice with different random seeds
    torch.manual_seed(42)
    output1 = embedding_layer(token_ids)
    
    torch.manual_seed(123)
    output2 = embedding_layer(token_ids)
    
    # Assert outputs are different (due to dropout randomness)
    assert not torch.allclose(output1, output2, rtol=1e-6)

def test_no_dropout_in_eval_mode(vocab_size, d_model):
    """Test that output is deterministic in eval mode"""
    # Create EmbeddingLayer with dropout
    embedding_layer = EmbeddingLayer(vocab_size, d_model, dropout=0.5)
    
    # Set to eval mode
    embedding_layer.eval()
    
    # Create same token_ids input
    token_ids = torch.randint(0, vocab_size, (1, 5))
    
    # Pass through twice
    output1 = embedding_layer(token_ids)
    output2 = embedding_layer(token_ids)
    
    # Assert outputs are identical (dropout disabled in eval mode)
    assert torch.allclose(output1, output2, rtol=1e-6)

def test_layer_norm_applied(vocab_size, d_model):
    """Test that layer normalization is applied"""
    # Create EmbeddingLayer
    embedding_layer = EmbeddingLayer(vocab_size, d_model)
    
    # Create token_ids input
    token_ids = torch.randint(0, vocab_size, (2, 10))
    
    # Pass through embedding layer
    embeddings = embedding_layer(token_ids)
    
    # Check layer normalization characteristics along d_model dimension
    # Layer norm should make mean ≈ 0 and std ≈ 1 for each position
    mean_per_position = torch.mean(embeddings, dim=-1)  # Shape: (batch_size, seq_length)
    std_per_position = torch.std(embeddings, dim=-1)   # Shape: (batch_size, seq_length)
    
    # Assert approximately zero mean and unit variance
    assert torch.allclose(mean_per_position, torch.zeros_like(mean_per_position), atol=1e-1)
    assert torch.allclose(std_per_position, torch.ones_like(std_per_position), atol=1e-1)

def test_handles_variable_sequence_lengths(vocab_size, d_model):
    """Test that layer handles different sequence lengths"""
    # Create EmbeddingLayer with max_seq_length=100
    embedding_layer = EmbeddingLayer(vocab_size, d_model, max_seq_length=100)
    
    # Test different sequence lengths
    sequence_lengths = [5, 10, 50]
    
    for seq_len in sequence_lengths:
        # Create token_ids with current sequence length
        token_ids = torch.randint(0, vocab_size, (1, seq_len))
        
        # Pass through embedding layer
        embeddings = embedding_layer(token_ids)
        
        # Assert correct output shape
        assert embeddings.shape == (1, seq_len, d_model)
