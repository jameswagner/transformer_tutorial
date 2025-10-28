"""
Demonstration of the complete embedding pipeline:
Text → Tokens → Token Embeddings → + Positional Encodings → Final Embeddings
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from src.tokenizers.simple_tokenizer import SimpleTokenizer
from src.embeddings.embedding_layer import EmbeddingLayer

def main():
    # Load and prepare tokenizer with simple corpus
    with open('data/simple_corpus.txt', 'r') as f:
        corpus_text = f.read()
    
    # Split by periods to get individual sentences
    sentences = corpus_text.split('. ')
    
    tokenizer = SimpleTokenizer(vocab_size=1000)
    tokenizer.fit(sentences)
    
    # Create embedding layer
    embedding_layer = EmbeddingLayer(
        vocab_size=len(tokenizer.vocab),
        d_model=512,
        max_seq_length=100,
        dropout=0.1
    )
    
    # Test with real sentence
    sentence = "The cat sat on the mat"
    print(f"Original sentence: {sentence}")
    
    # Encode to token IDs
    token_ids = tokenizer.encode(sentence, add_special_tokens=True)
    print(f"Token IDs: {token_ids}")
    
    # Decode to show what each ID means
    decoded = [tokenizer.get_token(tid) for tid in token_ids]
    print(f"Decoded tokens: {decoded}")
    
    # Convert to tensor and add batch dimension
    token_tensor = torch.tensor([token_ids])
    print(f"\nToken tensor shape: {token_tensor.shape}")
    
    # Get positional encodings for inspection
    from src.embeddings.positional_encoding import create_positional_encoding
    batch_size, seq_length = token_tensor.shape
    positional_encodings = create_positional_encoding(seq_length, 512, batch_size)
    print(f"\nPositional encodings shape: {positional_encodings.shape}")
    print(f"Positional encodings sample (position 0, first 16 dims):")
    print(positional_encodings[0, 0, :16].numpy())
    print(f"\nPositional encodings sample (position 1, first 16 dims):")
    print(positional_encodings[0, 1, :16].numpy())
    print(f"\nPositional encodings sample (position 2, first 16 dims):")
    print(positional_encodings[0, 2, :16].numpy())
    
    # Get embeddings
    embeddings = embedding_layer(token_tensor)
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Embeddings dtype: {embeddings.dtype}")
    
    # Show some stats
    print(f"\nEmbedding statistics:")
    print(f"  Mean: {embeddings.mean().item():.4f}")
    print(f"  Std: {embeddings.std().item():.4f}")
    print(f"  Min: {embeddings.min().item():.4f}")
    print(f"  Max: {embeddings.max().item():.4f}")
    
    print("\n✅ Full pipeline working!")

if __name__ == "__main__":
    main()
