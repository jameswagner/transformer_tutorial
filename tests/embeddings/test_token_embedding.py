import pytest
import torch
from src.embeddings.token_embedding import TokenEmbedding

@pytest.fixture
def vocab_size():
    return 10000

@pytest.fixture
def d_embed():
    return 256

@pytest.fixture
def d_model():
    return 512

@pytest.fixture
def batch_size():
    return 2

@pytest.fixture
def sequence_length():
    return 10

def test_output_shape(vocab_size, d_embed, d_model, batch_size, sequence_length):
    """Test that output shape is correct"""
    # Create TokenEmbedding instance
    embedding_layer = TokenEmbedding(vocab_size, d_embed, d_model)
    
    # Generate random token IDs
    token_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
    
    # Pass through embedding layer
    embeddings = embedding_layer(token_ids)
    
    # Assert correct output shape
    assert embeddings.shape == (batch_size, sequence_length, d_model)

def test_input_must_be_long_tensor(vocab_size, d_embed, d_model):
    """Test that input must be long tensor"""
    # Create TokenEmbedding instance
    embedding_layer = TokenEmbedding(vocab_size, d_embed, d_model)
    
    # Create float tensor input (should raise AssertionError)
    float_input = torch.randn(2, 10)  # Float tensor instead of long
    
    # Assert that passing float tensor raises AssertionError
    with pytest.raises(AssertionError, match="Expected torch.long"):
        embedding_layer(float_input)

def test_scaling_applied(vocab_size, d_model):
    """Test that embeddings are scaled by sqrt(d_model)"""
    # Create two TokenEmbedding instances with same d_model to test scaling
    embedding1 = TokenEmbedding(vocab_size, d_model, d_model)  # d_embed = d_model = 512
    
    # Use same token ID
    token_id = torch.tensor([[5]])  # Single token
    
    # Get embedding
    emb1 = embedding1(token_id)
    
    # Calculate norm
    norm1 = torch.norm(emb1)
    
    # The scale factor should be sqrt(d_model)
    expected_scale_factor = torch.sqrt(torch.tensor(float(d_model)))
    
    # Get the actual embedding without scaling by extracting from the embedding layer
    # and manually scaling
    raw_embedding = embedding1.embedding(token_id)
    manual_scaled = raw_embedding * embedding1.scale_factor
    
    # The embedding should be approximately the scale factor times the raw embedding
    # Since embedding weights are random, we just check that scaling is applied correctly
    assert torch.allclose(manual_scaled, emb1, rtol=1e-6)
    
    # Check that the scale factor is correct
    assert abs(embedding1.scale_factor - expected_scale_factor.item()) < 1e-6

def test_different_tokens_different_embeddings(vocab_size, d_model):
    """Different token IDs should produce different embeddings"""
    # Create TokenEmbedding instance
    embedding_layer = TokenEmbedding(vocab_size, d_model, d_model)
    
    # Create two different token IDs
    token_id1 = torch.tensor([[5]])
    token_id2 = torch.tensor([[10]])
    
    # Get embeddings for both tokens
    emb1 = embedding_layer(token_id1)
    emb2 = embedding_layer(token_id2)
    
    # Assert that embeddings are not equal
    assert not torch.allclose(emb1, emb2, rtol=1e-6)

def test_same_token_same_embedding(vocab_size, d_model):
    """Same token ID in different positions should have same embedding"""
    # Create TokenEmbedding instance
    embedding_layer = TokenEmbedding(vocab_size, d_model, d_model)
    
    # Create batch with same token ID in multiple positions
    token_ids = torch.tensor([[5, 5, 5]])  # Same token ID repeated
    
    # Get embeddings
    embeddings = embedding_layer(token_ids)
    
    # All embeddings should be identical
    assert torch.allclose(embeddings[0, 0], embeddings[0, 1], rtol=1e-6)
    assert torch.allclose(embeddings[0, 1], embeddings[0, 2], rtol=1e-6)
    assert torch.allclose(embeddings[0, 0], embeddings[0, 2], rtol=1e-6)

def test_projection_when_dimensions_differ():
    """When d_embed != d_model, projection should change dimensions"""
    # Create TokenEmbedding with different dimensions
    embedding_layer = TokenEmbedding(vocab_size=1000, d_embed=256, d_model=512)
    
    # Verify projection layer exists and has correct dimensions
    assert embedding_layer.projection is not None
    assert isinstance(embedding_layer.projection, torch.nn.Linear)
    assert embedding_layer.projection.in_features == 256
    assert embedding_layer.projection.out_features == 512

def test_no_projection_when_dimensions_same():
    """When d_embed == d_model, no projection layer needed"""
    # Create TokenEmbedding with same dimensions
    embedding_layer = TokenEmbedding(vocab_size=1000, d_embed=512, d_model=512)
    
    # Verify projection layer is None
    assert embedding_layer.projection is None
