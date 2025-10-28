import pytest
import torch
from src.attention.qkv_projection import QKVProjection

@pytest.fixture
def d_model():
    return 512

@pytest.fixture
def num_heads():
    return 8

@pytest.fixture
def batch_size():
    return 2

@pytest.fixture
def seq_length():
    return 10

def test_output_shapes(d_model, num_heads, batch_size, seq_length):
    """Test that Q, K, V have correct output shapes"""
    # Create QKVProjection
    qkv_proj = QKVProjection(d_model, num_heads)
    
    # Create random input
    x = torch.randn(batch_size, seq_length, d_model)
    
    # Get Q, K, V
    Q, K, V = qkv_proj(x)
    
    # Verify shapes
    assert Q.shape == (batch_size, seq_length, d_model)
    assert K.shape == (batch_size, seq_length, d_model)
    assert V.shape == (batch_size, seq_length, d_model)

def test_different_projections(d_model, num_heads):
    """Test that Q, K, V are different for same input"""
    # Create QKVProjection
    qkv_proj = QKVProjection(d_model, num_heads)
    
    # Create random input
    x = torch.randn(1, 5, d_model)
    
    # Get Q, K, V
    Q, K, V = qkv_proj(x)
    
    # Verify Q, K, V are different
    assert not torch.allclose(Q, K, rtol=1e-6)
    assert not torch.allclose(Q, V, rtol=1e-6)
    assert not torch.allclose(K, V, rtol=1e-6)

def test_d_model_divisible_by_heads():
    """Test that error is raised when d_model not divisible by num_heads"""
    # Test case where d_model is not divisible by num_heads
    with pytest.raises(AssertionError, match="d_model .* must be divisible by num_heads"):
        QKVProjection(d_model=512, num_heads=7)  # 512 % 7 != 0
