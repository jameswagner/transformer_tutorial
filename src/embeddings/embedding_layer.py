"""
Complete embedding layer combining token embeddings and positional encodings.
From Section 3.4 and 3.5 of "Attention Is All You Need"
"""
import torch
import torch.nn as nn
from .token_embedding import TokenEmbedding
from .positional_encoding import create_positional_encoding

class EmbeddingLayer(nn.Module):
    """
    Pomodoro 4: Combines token embeddings with positional encodings.
    Optionally applies layer normalization and dropout.
    """
    
    def __init__(self, vocab_size: int, d_model: int, max_seq_length: int = 512, dropout: float = 0.1):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            max_seq_length: Maximum sequence length for positional encoding
            dropout: Dropout rate
        """
        super().__init__()
        
        # Store instance variables
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.dropout = dropout
        
        # Create TokenEmbedding instance (d_embed = d_model for simplicity)
        self.token_embedding = TokenEmbedding(vocab_size, d_model, d_model)
        
        # Create LayerNorm layer
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Create Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [batch_size, seq_length] of token IDs (long tensor)
        Returns:
            embeddings: [batch_size, seq_length, d_model] with position info
        """
        # Get batch_size and seq_length from token_ids shape
        batch_size, seq_length = token_ids.shape
        
        # Get token embeddings
        token_embeddings = self.token_embedding(token_ids)
        
        # Create positional encodings
        positional_encodings = create_positional_encoding(seq_length, self.d_model, batch_size)
        
        # Ensure positional encodings are on the same device as token embeddings
        positional_encodings = positional_encodings.to(token_embeddings.device)
        
        # Add token embeddings and positional encodings together
        embeddings = token_embeddings + positional_encodings
        
        # Apply layer normalization
        embeddings = self.layer_norm(embeddings)
        
        # Apply dropout
        embeddings = self.dropout_layer(embeddings)
        
        return embeddings
