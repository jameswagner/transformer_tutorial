"""Visualize combined token + positional embeddings"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import matplotlib.pyplot as plt

from src.embeddings.embedding_layer import EmbeddingLayer
from src.tokenizers.simple_tokenizer import SimpleTokenizer

def plot_token_vs_positional():
    """Compare pure token embeddings vs token+positional"""
    # Force CPU for visualizations (matplotlib needs numpy/CPU)
    device = torch.device('cpu')
    
    # Create EmbeddingLayer
    vocab_size = 1000
    d_model = 512
    embedding_layer = EmbeddingLayer(vocab_size, d_model).to(device)
    
    # Create token_ids for short sequence
    sequence_length = 10
    token_ids = torch.randint(0, vocab_size, (1, sequence_length)).to(device)
    
    # Extract token embeddings only (before positional encoding)
    token_embeddings = embedding_layer.token_embedding(token_ids)
    
    # Extract positional encodings only
    from src.embeddings.positional_encoding import create_positional_encoding
    positional_encodings = create_positional_encoding(sequence_length, d_model, 1, device=device)
    
    # Get combined embeddings (final result)
    combined_embeddings = embedding_layer(token_ids)
    
    # Convert to numpy for plotting (first 64 dimensions)
    token_emb_np = token_embeddings[0, :, :64].detach().cpu().numpy()
    pos_enc_np = positional_encodings[0, :, :64].detach().cpu().numpy()
    combined_np = combined_embeddings[0, :, :64].detach().cpu().numpy()
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Token embeddings only
    im1 = axes[0].imshow(token_emb_np, aspect='auto', cmap='RdBu_r')
    axes[0].set_title('Token Embeddings Only\n(First 64 Dimensions)')
    axes[0].set_xlabel('Embedding Dimension')
    axes[0].set_ylabel('Position')
    plt.colorbar(im1, ax=axes[0], label='Value')
    
    # Plot 2: Positional encodings only
    im2 = axes[1].imshow(pos_enc_np, aspect='auto', cmap='RdBu_r')
    axes[1].set_title('Positional Encodings Only\n(First 64 Dimensions)')
    axes[1].set_xlabel('Embedding Dimension')
    axes[1].set_ylabel('Position')
    plt.colorbar(im2, ax=axes[1], label='Value')
    
    # Plot 3: Combined embeddings
    im3 = axes[2].imshow(combined_np, aspect='auto', cmap='RdBu_r')
    axes[2].set_title('Combined Embeddings\n(Token + Positional)')
    axes[2].set_xlabel('Embedding Dimension')
    axes[2].set_ylabel('Position')
    plt.colorbar(im3, ax=axes[2], label='Value')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('visualizations/embeddings/token_vs_positional.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Token vs positional comparison plot saved!")

def plot_same_word_different_positions():
    """Show how same word gets different embeddings at different positions"""
    # Force CPU for visualizations (matplotlib needs numpy/CPU)
    device = torch.device('cpu')
    
    # Create EmbeddingLayer
    vocab_size = 1000
    d_model = 512
    embedding_layer = EmbeddingLayer(vocab_size, d_model).to(device)
    
    # Create sequence where same token ID appears at positions 0, 3, 6, 9
    token_id = 42  # Same token ID
    sequence_length = 10
    token_ids = torch.zeros(1, sequence_length, dtype=torch.long).to(device)
    token_ids[0, 0] = token_id  # Position 0
    token_ids[0, 3] = token_id  # Position 3
    token_ids[0, 6] = token_id  # Position 6
    token_ids[0, 9] = token_id  # Position 9
    
    # Get combined embeddings
    embeddings = embedding_layer(token_ids)
    
    # Extract embeddings at the four positions
    positions = [0, 3, 6, 9]
    plt.figure(figsize=(12, 8))
    
    for pos in positions:
        embedding_at_pos = embeddings[0, pos, :64].detach().cpu().numpy()  # First 64 dimensions
        plt.plot(embedding_at_pos, label=f'Position {pos}', alpha=0.7, linewidth=2)
    
    plt.title('Same Token at Different Positions\n(First 64 Embedding Dimensions)')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Embedding Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('visualizations/embeddings/same_word_different_positions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Same word different positions plot saved!")

def plot_real_sentence():
    """Visualize embeddings for a real sentence"""
    # Force CPU for visualizations (matplotlib needs numpy/CPU)
    device = torch.device('cpu')
    
    # Load sample corpus
    with open('data/sample_corpus.txt', 'r') as f:
        corpus_text = f.read()
    
    # Create and fit tokenizer
    tokenizer = SimpleTokenizer(vocab_size=1000)
    sentences = corpus_text.split('. ')
    tokenizer.fit(sentences)
    
    # Create EmbeddingLayer
    vocab_size = tokenizer.get_vocab_size()
    d_model = 512
    embedding_layer = EmbeddingLayer(vocab_size, d_model).to(device)
    
    # Encode the sentence
    sentence = "The cat sat on the mat"
    token_ids = tokenizer.encode(sentence, add_special_tokens=True)
    print(f"Token IDs for '{sentence}': {token_ids}")
    
    # Convert to tensor and add batch dimension
    token_tensor = torch.tensor([token_ids]).to(device)
    
    # Get combined embeddings
    embeddings = embedding_layer(token_tensor)
    
    # Create heatmap (first 128 dimensions)
    embeddings_np = embeddings[0, :, :128].detach().cpu().numpy()  # Shape: (seq_length, 128)
    
    # Get actual words for y-axis labels
    words = []
    for token_id in token_ids:
        word = tokenizer.get_token(token_id)
        words.append(word)
    
    # Create the plot
    plt.figure(figsize=(16, 8))
    plt.imshow(embeddings_np, aspect='auto', cmap='RdBu_r')
    plt.colorbar(label='Embedding Value')
    plt.title(f'Real Sentence Embeddings: "{sentence}"\n(First 128 Dimensions)')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Word Position')
    
    # Set y-axis labels to actual words
    plt.yticks(range(len(words)), words)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('visualizations/embeddings/real_sentence_embeddings.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Real sentence embeddings plot saved!")

if __name__ == "__main__":
    plot_token_vs_positional()
    plot_same_word_different_positions()
    plot_real_sentence()
    print("\n🎨 All visualizations complete!")
