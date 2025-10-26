"""Test suite for SimpleTokenizer"""

import pytest
from src.tokenizers.simple_tokenizer import SimpleTokenizer


class TestSimpleTokenizer:
    """Test cases for SimpleTokenizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tokenizer = SimpleTokenizer(vocab_size=100)
        self.sample_texts = [
            "hello world",
            "hello there",
            "good morning",
            "good evening",
            "how are you",
            "I am fine"
        ]
    
    def test_initialization(self):
        """Test tokenizer initialization."""
        assert self.tokenizer.vocab_size == 100
        assert self.tokenizer.PAD_TOKEN == "[PAD]"
        assert self.tokenizer.UNK_TOKEN == "[UNK]"
        assert self.tokenizer.START_TOKEN == "[START]"
        assert self.tokenizer.END_TOKEN == "[END]"
        
        # Check special tokens are initialized
        assert self.tokenizer.vocab[self.tokenizer.PAD_TOKEN] == 0
        assert self.tokenizer.vocab[self.tokenizer.UNK_TOKEN] == 1
        assert self.tokenizer.vocab[self.tokenizer.START_TOKEN] == 2
        assert self.tokenizer.vocab[self.tokenizer.END_TOKEN] == 3
        
        # Check reverse mapping
        assert self.tokenizer.id_to_token[0] == self.tokenizer.PAD_TOKEN
        assert self.tokenizer.id_to_token[1] == self.tokenizer.UNK_TOKEN
        assert self.tokenizer.id_to_token[2] == self.tokenizer.START_TOKEN
        assert self.tokenizer.id_to_token[3] == self.tokenizer.END_TOKEN
    
    def test_fit_basic(self):
        """Test basic vocabulary building."""
        self.tokenizer.fit(self.sample_texts)
        
        # Should have special tokens + words from sample texts
        assert len(self.tokenizer.vocab) > 4
        
        # Check that common words are in vocabulary
        assert "hello" in self.tokenizer.vocab
        assert "good" in self.tokenizer.vocab
        assert "world" in self.tokenizer.vocab
        
        # Check reverse mapping is updated
        assert len(self.tokenizer.id_to_token) == len(self.tokenizer.vocab)
    
    def test_fit_frequency_ordering(self):
        """Test that vocabulary is ordered by frequency."""
        texts = ["hello hello hello", "world world", "good"]
        self.tokenizer.fit(texts)
        
        # "hello" should have lower ID than "world" (more frequent)
        hello_id = self.tokenizer.vocab["hello"]
        world_id = self.tokenizer.vocab["world"]
        good_id = self.tokenizer.vocab["good"]
        
        assert hello_id < world_id < good_id
    
    def test_fit_vocab_size_limit(self):
        """Test vocabulary size limit."""
        # Create many unique words
        texts = [f"word{i}" for i in range(200)]
        self.tokenizer.fit(texts)
        
        # Should not exceed vocab_size
        assert len(self.tokenizer.vocab) <= self.tokenizer.vocab_size
    
    def test_fit_less_common_words_pushed_out(self):
        """Test that less common words get pushed out when vocab limit is reached."""
        # Create a small vocab size to force pushing out
        small_tokenizer = SimpleTokenizer(vocab_size=6)  # 4 special + 2 regular words
        
        # Create texts with clear frequency differences
        texts = [
            "frequent frequent frequent",  # "frequent" appears 3 times
            "common common",              # "common" appears 2 times  
            "rare",                       # "rare" appears 1 time
            "uncommon"                    # "uncommon" appears 1 time
        ]
        
        small_tokenizer.fit(texts)
        
        # Should have exactly vocab_size words
        assert len(small_tokenizer.vocab) == small_tokenizer.vocab_size
        
        # Most frequent words should be in vocab
        assert "frequent" in small_tokenizer.vocab
        assert "common" in small_tokenizer.vocab
        
        # Less frequent words should be pushed out
        assert "rare" not in small_tokenizer.vocab
        assert "uncommon" not in small_tokenizer.vocab
        
        # Check that frequent words have lower IDs
        frequent_id = small_tokenizer.vocab["frequent"]
        common_id = small_tokenizer.vocab["common"]
        assert frequent_id < common_id  # "frequent" more common than "common"
    
    def test_preprocessing(self):
        """Test text preprocessing."""
        # Test that preprocessing works
        result = self.tokenizer._preprocess_text("  HELLO WORLD  ")
        assert result == "hello world"
        
        result = self.tokenizer._preprocess_text("Mixed Case Text")
        assert result == "mixed case text"
    
    def test_encode_basic(self):
        """Test basic encoding."""
        self.tokenizer.fit(self.sample_texts)
        
        # Test encoding without special tokens
        token_ids = self.tokenizer.encode("hello world", add_special_tokens=False)
        assert isinstance(token_ids, list)
        assert all(isinstance(tid, int) for tid in token_ids)
        
        # Test encoding with special tokens
        token_ids_with_special = self.tokenizer.encode("hello world", add_special_tokens=True)
        assert len(token_ids_with_special) == len(token_ids) + 2  # START + END
        assert token_ids_with_special[0] == self.tokenizer.vocab[self.tokenizer.START_TOKEN]
        assert token_ids_with_special[-1] == self.tokenizer.vocab[self.tokenizer.END_TOKEN]
    
    def test_encode_unknown_words(self):
        """Test encoding with unknown words."""
        self.tokenizer.fit(self.sample_texts)
        
        # Test with unknown word
        token_ids = self.tokenizer.encode("unknown word", add_special_tokens=False)
        unk_id = self.tokenizer.vocab[self.tokenizer.UNK_TOKEN]
        assert unk_id in token_ids
    
    def test_encode_case_insensitive(self):
        """Test that encoding is case insensitive."""
        self.tokenizer.fit(["hello world"])
        
        # Should produce same tokens regardless of case
        ids1 = self.tokenizer.encode("HELLO WORLD", add_special_tokens=False)
        ids2 = self.tokenizer.encode("hello world", add_special_tokens=False)
        assert ids1 == ids2
    
    def test_decode_basic(self):
        """Test basic decoding."""
        self.tokenizer.fit(self.sample_texts)
        
        # Test round-trip encoding/decoding
        original_text = "hello world"
        token_ids = self.tokenizer.encode(original_text, add_special_tokens=False)
        decoded_text = self.tokenizer.decode(token_ids)
        
        # Should decode back to lowercase (due to preprocessing)
        assert decoded_text == "hello world"
    
    def test_decode_with_special_tokens(self):
        """Test decoding with special tokens."""
        self.tokenizer.fit(self.sample_texts)
        
        token_ids = self.tokenizer.encode("hello world", add_special_tokens=True)
        decoded_text = self.tokenizer.decode(token_ids)
        
        # Should include special tokens in decoded text
        assert self.tokenizer.START_TOKEN in decoded_text
        assert self.tokenizer.END_TOKEN in decoded_text
    
    def test_decode_unknown_ids(self):
        """Test decoding with unknown token IDs."""
        self.tokenizer.fit(self.sample_texts)
        
        # Test with unknown token ID
        unknown_ids = [999, 1000]  # Non-existent IDs
        decoded_text = self.tokenizer.decode(unknown_ids)
        
        # Should decode to UNK tokens
        assert decoded_text == f"{self.tokenizer.UNK_TOKEN} {self.tokenizer.UNK_TOKEN}"
    
    def test_pad_sequence_padding(self):
        """Test sequence padding."""
        self.tokenizer.fit(self.sample_texts)
        
        token_ids = [1, 2, 3]
        padded = self.tokenizer.pad_sequence(token_ids, max_length=5)
        
        assert len(padded) == 5
        assert padded[:3] == [1, 2, 3]
        assert padded[3:] == [0, 0]  # PAD token ID
    
    def test_pad_sequence_truncation(self):
        """Test sequence truncation."""
        self.tokenizer.fit(self.sample_texts)
        
        token_ids = [1, 2, 3, 4, 5]
        truncated = self.tokenizer.pad_sequence(token_ids, max_length=3)
        
        assert len(truncated) == 3
        assert truncated == [1, 2, 3]
    
    def test_pad_sequence_exact_length(self):
        """Test padding when sequence is exact length."""
        self.tokenizer.fit(self.sample_texts)
        
        token_ids = [1, 2, 3]
        result = self.tokenizer.pad_sequence(token_ids, max_length=3)
        
        assert len(result) == 3
        assert result == [1, 2, 3]
    
    def test_get_vocab_size(self):
        """Test getting vocabulary size."""
        initial_size = self.tokenizer.get_vocab_size()
        assert initial_size == 4  # Only special tokens initially
        
        self.tokenizer.fit(self.sample_texts)
        new_size = self.tokenizer.get_vocab_size()
        assert new_size > initial_size
    
    def test_get_token_id(self):
        """Test getting token ID."""
        self.tokenizer.fit(self.sample_texts)
        
        # Test known token
        hello_id = self.tokenizer.get_token_id("hello")
        assert isinstance(hello_id, int)
        assert hello_id == self.tokenizer.vocab["hello"]
        
        # Test unknown token
        unk_id = self.tokenizer.get_token_id("unknown")
        assert unk_id == self.tokenizer.vocab[self.tokenizer.UNK_TOKEN]
    
    def test_get_token(self):
        """Test getting token from ID."""
        self.tokenizer.fit(self.sample_texts)
        
        # Test known ID
        hello_id = self.tokenizer.vocab["hello"]
        token = self.tokenizer.get_token(hello_id)
        assert token == "hello"
        
        # Test unknown ID
        unk_token = self.tokenizer.get_token(999)
        assert unk_token == self.tokenizer.UNK_TOKEN
    
    def test_encode_batch(self):
        """Test batch encoding."""
        self.tokenizer.fit(self.sample_texts)
        
        texts = ["hello world", "good morning", "how are you"]
        batch_encoded = self.tokenizer.encode_batch(texts, add_special_tokens=False)
        
        assert len(batch_encoded) == len(texts)
        assert all(isinstance(seq, list) for seq in batch_encoded)
        assert all(isinstance(tid, int) for seq in batch_encoded for tid in seq)
    
    def test_encode_batch_with_special_tokens(self):
        """Test batch encoding with special tokens."""
        self.tokenizer.fit(self.sample_texts)
        
        texts = ["hello", "world"]
        batch_encoded = self.tokenizer.encode_batch(texts, add_special_tokens=True)
        
        # Each sequence should have START and END tokens
        for seq in batch_encoded:
            assert seq[0] == self.tokenizer.vocab[self.tokenizer.START_TOKEN]
            assert seq[-1] == self.tokenizer.vocab[self.tokenizer.END_TOKEN]
    
    def test_round_trip_consistency(self):
        """Test that encode->decode round trip works."""
        self.tokenizer.fit(self.sample_texts)
        
        original_texts = ["hello world", "good morning", "how are you"]
        
        for text in original_texts:
            # Encode without special tokens for cleaner comparison
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            decoded_text = self.tokenizer.decode(token_ids)
            
            # Should match original (lowercased due to preprocessing)
            assert decoded_text == text.lower()
    
    def test_edge_cases(self):
        """Test edge cases."""
        self.tokenizer.fit(self.sample_texts)
        
        # Empty string
        empty_ids = self.tokenizer.encode("", add_special_tokens=False)
        assert empty_ids == []
        
        # Only whitespace
        whitespace_ids = self.tokenizer.encode("   ", add_special_tokens=False)
        assert whitespace_ids == []
        
        # Single word
        single_word_ids = self.tokenizer.encode("hello", add_special_tokens=False)
        assert len(single_word_ids) == 1
        assert single_word_ids[0] == self.tokenizer.vocab["hello"]


if __name__ == "__main__":
    pytest.main([__file__])
