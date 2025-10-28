"""
Token embedding layer with optional projection and scaling.
From Section 3.4 of "Attention Is All You Need"
"""
import torch
import torch.nn as nn
import math

class TokenEmbedding(nn.Module):
    """
    Pomodoro 3: Token embedding with optional projection layer.
    Token embedding with optional projection layer.
    Scales embeddings by sqrt(d_model) as per the paper.
    """
    
    def __init__(self, vocab_size: int, d_embed: int, d_model: int):
        """
        Args:
            vocab_size: Size of vocabulary
            d_embed: Embedding dimension
            d_model: Model dimension (if different from d_embed, adds projection)
        """
        super().__init__()
        
        # Store instance variables
        self.vocab_size = vocab_size
        self.d_embed = d_embed
        self.d_model = d_model
        
        # Create embedding layer
        self.embedding = nn.Embedding(vocab_size, d_embed)
        
        # Create projection layer if needed
        if d_embed != d_model:
            self.projection = nn.Linear(d_embed, d_model)
        else:
            self.projection = None
        
        # Store scaling factor
        self.scale_factor = math.sqrt(d_model)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [batch_size, seq_length] of token IDs (long tensor)
        Returns:
            embeddings: [batch_size, seq_length, d_model]
        """
        # Assert input tensor dtype is torch.long
        assert token_ids.dtype == torch.long, f"Expected torch.long, got {token_ids.dtype}"
        
        # Pass through embedding layer
        embeddings = self.embedding(token_ids)
        
        # Apply projection if it exists
        # Note: nn.Embedding and nn.Linear automatically track device when the module is moved with .to(device)
        if self.projection is not None:
            embeddings = self.projection(embeddings)
        
        # Scale embeddings by sqrt(d_model)
        embeddings = embeddings * self.scale_factor
        
        return embeddings
