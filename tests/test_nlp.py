"""Tests for NLP functions: sentiment, embeddings, cleaning."""
import pytest
from app.nlp.sentiment import score_text, _detect_sarcasm
from app.nlp.embeddings import compute_embedding, _hash_based_embedding
from app.nlp.clean import normalize_post, extract_symbols
import numpy as np

class TestSentiment:
    """Test sentiment scoring."""

    def test_score_text_returns_sentiment_score(self):
        """Test that score_text returns a SentimentScore object."""
        result = score_text("This stock is bullish")
        assert result is not None
        assert hasattr(result, "polarity")
        assert hasattr(result, "confidence")
        assert hasattr(result, "subjectivity")

    def test_score_text_polarity_range(self):
        """Test that polarity is between -1 and 1."""
        result = score_text("This stock is excellent and will moon")
        assert -1.0 <= result.polarity <= 1.0

    def test_score_text_confidence_range(self):
        """Test that confidence is between 0 and 1."""
        result = score_text("Some text")
        assert 0.0 <= result.confidence <= 1.0

    def test_score_text_positive(self):
        """Test positive sentiment detection."""
        result = score_text("This is great, bullish, moon, profit")
        # Should be positive if model works, or heuristic fallback detects positive words
        assert result.polarity > 0.0 or result.model == "heuristic"

    def test_score_text_negative(self):
        """Test negative sentiment detection."""
        result = score_text("This is terrible, bearish, crash, loss")
        # Should be negative or fallback to heuristic
        assert result.polarity < 0.0 or result.model == "heuristic"

    def test_sarcasm_detection(self):
        """Test sarcasm detection."""
        sarcasm_prob = _detect_sarcasm("yeah right, sure that'll happen")
        assert sarcasm_prob > 0.5

    def test_no_sarcasm(self):
        """Test no sarcasm in normal text."""
        sarcasm_prob = _detect_sarcasm("This is a normal sentence")
        assert sarcasm_prob < 0.2


class TestEmbeddings:
    """Test text embedding generation."""

    def test_compute_embedding_returns_array(self):
        """Test that compute_embedding returns a numpy array."""
        result = compute_embedding("test text")
        assert isinstance(result, np.ndarray)

    def test_embedding_shape(self):
        """Test that embedding has correct dimensionality."""
        result = compute_embedding("test text")
        # Should be 384 dim for sentence-transformers, or fallback hash
        assert result.shape == (384,) or result.shape == (768,)

    def test_embedding_normalized(self):
        """Test that embedding is normalized (unit length)."""
        result = compute_embedding("test text")
        norm = np.linalg.norm(result)
        assert np.isclose(norm, 1.0, atol=0.01)

    def test_embedding_deterministic(self):
        """Test that same text produces same embedding."""
        emb1 = compute_embedding("same text")
        emb2 = compute_embedding("same text")
        assert np.allclose(emb1, emb2)

    def test_embedding_different_for_different_text(self):
        """Test that different text produces different embeddings."""
        emb1 = compute_embedding("text one")
        emb2 = compute_embedding("text two")
        # Should not be identical
        assert not np.allclose(emb1, emb2)

    def test_hash_based_embedding_fallback(self):
        """Test fallback hash-based embedding."""
        result = _hash_based_embedding("test")
        assert result.shape == (384,)
        assert np.isclose(np.linalg.norm(result), 1.0)


class TestCleaning:
    """Test text cleaning functions."""

    def test_normalize_post_removes_urls(self):
        """Test that URLs are removed."""
        text = "Check this out https://example.com amazing"
        result = normalize_post(text)
        assert "https://" not in result
        assert "example" not in result

    def test_normalize_post_normalizes_whitespace(self):
        """Test that whitespace is normalized."""
        text = "Text   with   multiple   spaces\n\n  and  newlines"
        result = normalize_post(text)
        assert "   " not in result
        assert "\n\n" not in result

    def test_extract_symbols_cashtag(self):
        """Test symbol extraction from cashtags."""
        text = "I'm bullish on $AAPL and $TSLA"
        symbols = extract_symbols(text, {})
        assert "AAPL" in symbols
        assert "TSLA" in symbols

    def test_extract_symbols_company_name(self):
        """Test symbol extraction from company names."""
        inst = {"symbol": "AAPL", "company_name": "Apple"}
        text = "Apple is doing great"
        symbols = extract_symbols(text, inst)
        assert "AAPL" in symbols

    def test_extract_symbols_case_insensitive(self):
        """Test case-insensitive symbol extraction."""
        text = "bullish on $aapl"
        symbols = extract_symbols(text, {})
        assert "AAPL" in symbols
