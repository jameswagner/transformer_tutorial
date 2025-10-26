"""
Visualize positional encoding patterns
Shows how sine/cosine waves create unique position signatures
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from src.embeddings.positional_encoding import create_positional_encoding


def plot_encoding_heatmap():
    """Main visualization: heatmap of positional encodings"""
    pe = create_positional_encoding(sequence_length=100, d_model=512, batch_size=1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Full encoding
    im1 = ax1.imshow(pe[0].numpy(), aspect='auto', cmap='RdBu_r')
    ax1.set_xlabel('Dimension')
    ax1.set_ylabel('Position')
    ax1.set_title('Positional Encoding Heatmap')
    plt.colorbar(im1, ax=ax1)
    
    # First 64 dimensions (easier to see patterns)
    im2 = ax2.imshow(pe[0, :, :64].numpy(), aspect='auto', cmap='RdBu_r')
    ax2.set_xlabel('Dimension')
    ax2.set_ylabel('Position')
    ax2.set_title('First 64 Dimensions (Detail)')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('visualizations/positional_encoding_heatmap.png', dpi=150)
    print("✅ Saved: visualizations/positional_encoding_heatmap.png")


def plot_position_comparison():
    """Compare encodings at different positions"""
    pe = create_positional_encoding(sequence_length=100, d_model=512, batch_size=1)
    
    positions = [0, 10, 50, 99]
    plt.figure(figsize=(12, 6))
    
    for pos in positions:
        plt.plot(pe[0, pos, :64].numpy(), label=f'Position {pos}', alpha=0.7)
    
    plt.xlabel('Dimension')
    plt.ylabel('Encoding Value')
    plt.title('Positional Encoding at Different Positions (First 64 dims)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('visualizations/position_comparison.png', dpi=150)
    print("✅ Saved: visualizations/position_comparison.png")


def plot_dimension_waves():
    """Show how different dimensions have different frequencies"""
    pe = create_positional_encoding(sequence_length=100, d_model=512, batch_size=1)
    
    dims = [0, 10, 50, 100, 200]  # Even dimensions only (sine)
    plt.figure(figsize=(12, 6))
    
    for dim in dims:
        plt.plot(pe[0, :, dim].numpy(), label=f'Dim {dim}', alpha=0.7)
    
    plt.xlabel('Position')
    plt.ylabel('Encoding Value')
    plt.title('Different Dimensions Have Different Wavelengths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('visualizations/dimension_waves.png', dpi=150)
    print("✅ Saved: visualizations/dimension_waves.png")


if __name__ == "__main__":
    plot_encoding_heatmap()
    plot_position_comparison()
    plot_dimension_waves()
    print("\n🎨 All visualizations complete!")