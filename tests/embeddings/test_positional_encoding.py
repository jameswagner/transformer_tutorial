import pytest
import torch

from src.embeddings.positional_encoding import create_positional_encoding

@pytest.fixture
def batch_size():
    return 2

@pytest.fixture
def sequence_length():
    return 10

@pytest.fixture
def d_model():
    return 512

def test_shape(batch_size, sequence_length, d_model):
    """Test output has correct shape"""
    pe = create_positional_encoding(sequence_length, d_model, batch_size, device='cpu')
    assert pe.shape == (batch_size, sequence_length, d_model)

def test_dtype_and_device(sequence_length, d_model):
    """Test tensor properties"""
    pe = create_positional_encoding(sequence_length, d_model, batch_size=1, device='cpu')
    assert pe.dtype == torch.float32
    assert not pe.requires_grad

def test_deterministic(sequence_length, d_model, batch_size):
    """Same inputs should produce identical outputs"""
    pe1 = create_positional_encoding(sequence_length, d_model, batch_size, device='cpu')
    pe2 = create_positional_encoding(sequence_length, d_model, batch_size, device='cpu')
    assert torch.allclose(pe1, pe2)

def test_batches_identical(sequence_length, d_model, batch_size):
    """All batches should have identical positional encodings"""
    pe = create_positional_encoding(sequence_length, d_model, batch_size, device='cpu')
    for i in range(1, batch_size):
        assert torch.allclose(pe[0], pe[i])

def test_unique_positions(sequence_length, d_model):
    """Each position should have a unique encoding"""
    pe = create_positional_encoding(sequence_length, d_model, batch_size=1, device='cpu')
    for i in range(sequence_length - 1):
        assert not torch.allclose(pe[0, i], pe[0, i+1])

def test_value_range(sequence_length, d_model, batch_size):
    """Values should be bounded between -1 and 1"""
    pe = create_positional_encoding(sequence_length, d_model, batch_size, device='cpu')
    assert pe.min() >= -1.0
    assert pe.max() <= 1.0

def test_even_odd_pattern():
    """Even dimensions use sine, odd use cosine"""
    pe = create_positional_encoding(sequence_length=5, d_model=8, batch_size=1, device='cpu')
    # At position 0, sine values ~0, cosine values ~1
    assert torch.allclose(pe[0, 0, 0::2], torch.zeros(4), atol=1e-6)
    assert torch.allclose(pe[0, 0, 1::2], torch.ones(4), atol=1e-6)