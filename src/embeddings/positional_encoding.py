import torch
import math


def create_positional_encoding(sequence_length: int, d_model: int, batch_size: int) -> torch.Tensor:
    """
    Create positional encoding for a sequence of a given length and model dimension.
    
    Args:
        sequence_length (int): The length of the sequence.
        d_model (int): The dimension of the model.
        batch_size (int): The batch size.
    Returns:
        torch.Tensor: A tensor of shape (sequence_length, d_model) containing the positional encoding.
    """
    positional_encoding = torch.zeros(batch_size, sequence_length, d_model)
    position = torch.arange(sequence_length).unsqueeze(1)
    denominator = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    positional_encoding[:, :, 0::2] = torch.sin(position * denominator)
    positional_encoding[:, :, 1::2] = torch.cos(position * denominator)
    return positional_encoding
