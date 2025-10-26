"""Simple word-level tokenizer for learning"""
from typing import List, Dict
import re

class SimpleTokenizer:
    """
    Basic word-level tokenizer with special tokens.
    Good enough for learning, not production-ready.
    """
    
    # Special tokens
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    START_TOKEN = "[START]"
    END_TOKEN = "[END]"
    
    def __init__(self, vocab_size: int = 10000):
        """
        Args:
            vocab_size: Maximum vocabulary size (including special tokens)
        """
        self.vocab_size = vocab_size
        self.vocab = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.START_TOKEN: 2,
            self.END_TOKEN: 3,
        }
        self.id_to_token = {
            0: self.PAD_TOKEN,
            1: self.UNK_TOKEN,
            2: self.START_TOKEN,
            3: self.END_TOKEN,
        }
    
    def _preprocess_text(self, text: str) -> str:
        """Basic text preprocessing."""
        return text.strip().lower()
    
    def fit(self, texts: List[str]) -> None:
        """
        Build vocabulary from training texts.
        
        Args:
            texts: List of training sentences
        """
        word_counts = {}
        for text in texts:
            processed_text = self._preprocess_text(text)
            for word in processed_text.split():
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency, then alphabetically for consistency
        sorted_words = sorted(word_counts.keys(), key=lambda w: (-word_counts[w], w))
        
        for word in sorted_words:
            if len(self.vocab) >= self.vocab_size:
                break
            self.vocab[word] = len(self.vocab)
        
        # Update reverse mapping
        self.id_to_token = {v: k for k, v in self.vocab.items()}
    
    def get_vocab_size(self) -> int:
        """Return actual vocabulary size."""
        return len(self.vocab)
    
    def get_token_id(self, token: str) -> int:
        """Get token ID, returns UNK if not found."""
        return self.vocab.get(token, self.vocab[self.UNK_TOKEN])
    
    def get_token(self, token_id: int) -> str:
        """Get token from ID, returns UNK if not found."""
        return self.id_to_token.get(token_id, self.UNK_TOKEN)
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Convert text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add [START] and [END]
        Returns:
            List of token IDs
        """
        processed_text = self._preprocess_text(text)
        token_ids = []
        for word in processed_text.split():
            if word in self.vocab:
                token_ids.append(self.vocab[word])
            else:
                token_ids.append(self.vocab[self.UNK_TOKEN])
        if add_special_tokens:
            token_ids = [self.vocab[self.START_TOKEN]] + token_ids + [self.vocab[self.END_TOKEN]]
        return token_ids
    
    def encode_batch(self, texts: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        """Encode multiple texts."""
        return [self.encode(text, add_special_tokens) for text in texts]
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Convert token IDs back to text.
        
        Args:
            token_ids: List of token IDs
        Returns:
            Decoded text
        """
        words = [self.id_to_token.get(tid, self.UNK_TOKEN) for tid in token_ids]
        return " ".join(words)
    
    def pad_sequence(self, token_ids: List[int], max_length: int) -> List[int]:
        """
        Pad sequence to max_length with [PAD] tokens.
        """
        if len(token_ids) >= max_length:
            return token_ids[:max_length]  # Truncate if too long
        pad_id = self.vocab[self.PAD_TOKEN]
        return token_ids + [pad_id] * (max_length - len(token_ids))