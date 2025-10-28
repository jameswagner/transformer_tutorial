# Transformer Tutorial: Building "Attention Is All You Need" from Scratch

This is a recreation of the groundbreaking paper "Attention is all you need" broken down into Pomodoros.

## Completed Pomodoros

### Pomodoro 1: Positional Encoding
- ✅ **Sinusoidal encoding**: Fixed positional patterns using sin/cos
- ✅ **Device support**: Optional device parameter
- ✅ **Batch processing**: Efficient batch-wise generation
- ✅ **Comprehensive tests**: Shape, uniqueness, value range tests

### Pomodoro 2: Simple Tokenizer
- ✅ **SimpleTokenizer**: Word-level tokenizer with special tokens
- ✅ **Vocabulary building**: Frequency-based word ordering
- ✅ **Encoding/Decoding**: Text ↔ Token ID conversion
- ✅ **Padding/Truncation**: Sequence length management
- ✅ **Comprehensive tests**: Full test suite with edge cases

### Pomodoro 3: Token Embeddings
- ✅ **TokenEmbedding**: Converts token IDs to dense vectors
- ✅ **Scaling**: sqrt(d_model) scaling for stability
- ✅ **Projection**: Optional linear projection when d_embed ≠ d_model
- ✅ **Device handling**: CPU/GPU compatibility
- ✅ **Comprehensive tests**: Shape, scaling, projection tests

### Pomodoro 4: Combined Embedding Layer
- ✅ **EmbeddingLayer**: Combines token + positional embeddings
- ✅ **Layer normalization**: Training stability
- ✅ **Dropout**: Regularization during training
- ✅ **Device handling**: Automatic device matching
- ✅ **Comprehensive tests**: Position info, token info, dropout behavior

### Pomodoro 5: QKV Projections
- ✅ **QKVProjection**: Transforms embeddings to Query, Key, Value
- ✅ **Multi-head support**: Prepares for multi-head attention
- ✅ **Separate projections**: Independent Q, K, V transformations
- ✅ **Comprehensive tests**: Shape validation, projection differences

## Project Structure

```
src/
├── tokenizers/
│   └── simple_tokenizer.py
├── embeddings/
│   ├── token_embedding.py
│   ├── positional_encoding.py
│   └── embedding_layer.py
└── attention/
    └── qkv_projection.py

tests/
├── tokenizers/
│   └── test_simple_tokenizer.py
├── embeddings/
│   ├── test_token_embedding.py
│   ├── test_positional_encoding.py
│   └── test_embedding_layer.py
└── attention/
    └── test_qkv_projection.py

visualizations/
└── embeddings/
    ├── plot_token_embeddings.py
    └── plot_combined_embeddings.py

examples/
└── embeddings_demo.py

data/
├── sample_corpus.txt
└── simple_corpus.txt
```

## Next Steps

- **Pomodoro 6**: Multi-head attention mechanism
- **Pomodoro 7**: Encoder layer
- **Pomodoro 8**: Decoder layer
- **Pomodoro 9**: Complete transformer model
- **Pomodoro 10**: Training and inference

## Key Features

- **Educational focus**: Clear explanations for first-year CS students
- **Modular design**: Each component is independently testable
- **GPU ready**: Device-agnostic code for CPU/GPU usage
- **Comprehensive testing**: Full test coverage with edge cases
- **Visualizations**: Plotting tools for understanding embeddings
- **Documentation**: Detailed explanations in CODE_EXPLANATION.md