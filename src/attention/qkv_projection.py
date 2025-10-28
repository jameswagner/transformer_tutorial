"""Query, Key, Value projections for attention mechanism"""
import torch
import torch.nn as nn

class QKVProjection(nn.Module):
    """
    Pomorodoro 5: Projects input embeddings to Q, K, V for attention.
    """
    
    def __init__(self, d_model: int, num_heads: int, bias: bool = True):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            bias: Whether to include bias in projections
        """
        super().__init__()
        
        # Assert d_model is divisible by num_heads
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        # Store instance variables
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Calculate d_head
        self.d_head = d_model // num_heads
        
        # Create three linear projections: Q, K, V
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_length, d_model]
        Returns:
            Q, K, V each [batch_size, seq_length, d_model]
        """
        # Apply projections to get Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        return Q, K, V

