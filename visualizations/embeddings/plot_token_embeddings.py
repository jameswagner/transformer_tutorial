"""Visualize token embeddings"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import matplotlib.pyplot as plt
import numpy as np
from src.embeddings.token_embedding import TokenEmbedding
from src.tokenizers.simple_tokenizer import SimpleTokenizer

def plot_embedding_matrix():
    """Visualize the raw embedding lookup table as a heatmap"""
    # Force CPU for visualizations (matplotlib needs numpy/CPU)
    device = torch.device('cpu')
    
    # Create TokenEmbedding with small vocab and d_model
    vocab_size = 100
    d_model = 64
    embedding_layer = TokenEmbedding(vocab_size, d_model, d_model).to(device)
    
    # Extract the embedding weight matrix
    embedding_weights = embedding_layer.embedding.weight.data.numpy()
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.imshow(embedding_weights, aspect='auto', cmap='RdBu_r')
    plt.colorbar(label='Embedding Value')
    plt.title('Token Embedding Matrix\n(Vocab Size: 100, d_model: 64)')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Token ID')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('visualizations/embeddings/embedding_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Embedding matrix heatmap saved!")

def plot_token_similarity():
    """Show cosine similarity between random token pairs"""
    # Force CPU for visualizations (matplotlib needs numpy/CPU)
    device = torch.device('cpu')
    
    # Create TokenEmbedding
    vocab_size = 100
    d_model = 64
    embedding_layer = TokenEmbedding(vocab_size, d_model, d_model).to(device)
    
    # Select 10 random token IDs
    torch.manual_seed(42)  # For reproducible results
    token_ids = torch.randint(0, vocab_size, (10,)).to(device)
    
    # Get embeddings for selected tokens
    embeddings = embedding_layer(token_ids.unsqueeze(1))  # Shape: (10, 1, 64)
    embeddings = embeddings.squeeze(1)  # Shape: (10, 64)
    
    # Compute cosine similarity matrix
    # Normalize embeddings
    embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    # Compute similarity matrix
    similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
    
    # Convert to numpy for plotting
    similarity_np = similarity_matrix.data.numpy()
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_np, cmap='RdYlBu_r', vmin=-1, vmax=1)
    plt.colorbar(label='Cosine Similarity')
    plt.title('Token Embedding Similarity Matrix\n(10 Random Tokens)')
    plt.xlabel('Token Index')
    plt.ylabel('Token Index')
    
    # Add token IDs as tick labels
    plt.xticks(range(10), token_ids.cpu().numpy())
    plt.yticks(range(10), token_ids.cpu().numpy())
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('visualizations/embeddings/token_similarity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Token similarity heatmap saved!")

def plot_real_words():
    """Plot embeddings for actual words from tokenizer"""
    # Force CPU for visualizations (matplotlib needs numpy/CPU)
    device = torch.device('cpu')
    
    # Load sample corpus
    with open('data/sample_corpus.txt', 'r') as f:
        corpus_text = f.read()
    
    # Create and fit tokenizer
    tokenizer = SimpleTokenizer(vocab_size=1000)
    sentences = corpus_text.split('. ')
    tokenizer.fit(sentences)
    
    # Create TokenEmbedding
    vocab_size = tokenizer.get_vocab_size()
    d_model = 64
    embedding_layer = TokenEmbedding(vocab_size, d_model, d_model).to(device)
    
    # Select some interesting words to visualize
    words_to_plot = ['transformer', 'attention', 'learning', 'language', 'model']
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot embeddings for each word
    for word in words_to_plot:
        if word in tokenizer.vocab:
            # Get token ID and embedding
            token_id = tokenizer.get_token_id(word)
            token_tensor = torch.tensor([[token_id]]).to(device)
            embedding = embedding_layer(token_tensor).squeeze(0).data.numpy()
            
            # Plot first 50 dimensions
            plt.plot(embedding[:50], label=word, alpha=0.7, linewidth=2)
        else:
            print(f"Warning: '{word}' not in vocabulary")
    
    plt.title('Real Word Embeddings (First 50 Dimensions)')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Embedding Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('visualizations/embeddings/real_word_embeddings.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Real word embeddings plot saved!")

if __name__ == "__main__":
    plot_embedding_matrix()
    plot_token_similarity()
    plot_real_words()
    print("\n🎨 All visualizations complete!")
