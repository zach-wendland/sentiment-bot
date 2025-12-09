"""
Sentiment Analysis Module

Uses DistilBERT fine-tuned on SST-2 for fast sentiment classification.
Optimized for serverless cold starts (~250MB, ~5s load time).
"""

import logging
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from app.services.types import SentimentScore

logger = logging.getLogger(__name__)

# Global model cache for warm invocations
_model_cache = {}

# Model configuration
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
# DistilBERT SST-2 classes: 0 = NEGATIVE, 1 = POSITIVE
# We derive neutral from low confidence


def _get_model() -> Optional[Tuple]:
    """
    Load DistilBERT model (cached after first load).

    Returns:
        Tuple of (tokenizer, model, device) or None if loading fails
    """
    if "distilbert" not in _model_cache:
        try:
            logger.info(f"Loading sentiment model: {MODEL_NAME}")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

            # Use CPU (more predictable in serverless environments)
            device = "cpu"
            model.to(device)
            model.eval()

            _model_cache["distilbert"] = (tokenizer, model, device)
            logger.info(f"Loaded DistilBERT sentiment model on {device}")
        except Exception as e:
            logger.error(f"Failed to load DistilBERT: {e}. Falling back to heuristics.")
            return None

    return _model_cache.get("distilbert")


def score_text(text: str) -> SentimentScore:
    """
    Score sentiment using DistilBERT model.

    Falls back to heuristics if model loading fails.

    Args:
        text: Text to analyze for sentiment

    Returns:
        SentimentScore with polarity (-1 to +1) and confidence
    """
    if not text or not text.strip():
        return SentimentScore(
            polarity=0.0,
            subjectivity=0.0,
            sarcasm_prob=0.0,
            confidence=0.0,
            model="empty"
        )

    # Try DistilBERT first
    model_tuple = _get_model()

    if model_tuple:
        try:
            tokenizer, model, device = model_tuple

            # Truncate text to max length (512 tokens for BERT-based models)
            inputs = tokenizer(
                text[:1024],  # Pre-truncate before tokenizing
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

            # DistilBERT SST-2 has 2 classes: [NEGATIVE, POSITIVE]
            probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()

            neg_prob = float(probabilities[0])
            pos_prob = float(probabilities[1])

            # Calculate polarity: -1 (negative) to +1 (positive)
            # Use difference: pos_prob - neg_prob gives range [-1, 1]
            polarity = pos_prob - neg_prob

            # Confidence is the highest probability
            confidence = max(neg_prob, pos_prob)

            # Detect "neutral" sentiment via low confidence
            # If confidence < 0.6, dampen polarity towards neutral
            if confidence < 0.6:
                polarity *= confidence  # Reduce polarity for uncertain predictions

            # Subjectivity: estimated from confidence spread
            # High confidence = more subjective (clear opinion)
            # Low confidence = more objective (factual or unclear)
            subjectivity = abs(polarity)

            return SentimentScore(
                polarity=max(-1.0, min(1.0, polarity)),
                subjectivity=subjectivity,
                sarcasm_prob=_detect_sarcasm(text),
                confidence=confidence,
                model="distilbert"
            )
        except Exception as e:
            logger.warning(f"DistilBERT inference failed: {e}. Falling back to heuristics.")

    # Fallback to heuristics
    return _score_text_heuristic(text)


def _score_text_heuristic(text: str) -> SentimentScore:
    """
    Simple heuristic-based sentiment scoring (fallback).

    Uses financial-specific word lists for basic sentiment detection.
    """
    positive_words = [
        'bullish', 'moon', 'buy', 'long', 'growth', 'profit',
        'gain', 'up', 'surge', 'boom', 'excellent', 'great', 'strong',
        'rocket', 'soar', 'rally', 'breakout', 'undervalued'
    ]
    negative_words = [
        'bearish', 'crash', 'sell', 'short', 'loss', 'down',
        'dump', 'fall', 'decline', 'terrible', 'bad', 'weak',
        'tank', 'plunge', 'overvalued', 'bubble', 'scam'
    ]

    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)

    total = pos_count + neg_count
    if total == 0:
        polarity = 0.0
        confidence = 0.3
    else:
        polarity = (pos_count - neg_count) / total
        confidence = min(0.7, total / 5)

    # Subjectivity based on presence of opinion words
    subjectivity = min(1.0, total / 3) if total > 0 else 0.2

    return SentimentScore(
        polarity=max(-1.0, min(1.0, polarity)),
        subjectivity=subjectivity,
        sarcasm_prob=_detect_sarcasm(text),
        confidence=confidence,
        model="heuristic"
    )


def _detect_sarcasm(text: str) -> float:
    """
    Detect potential sarcasm in text.

    Uses simple pattern matching for common sarcasm indicators.
    Returns probability between 0.0 and 1.0.
    """
    sarcasm_indicators = {
        'yeah right': 0.8,
        'sure thing': 0.7,
        'totally': 0.4,
        '/s': 0.95,  # Reddit sarcasm marker
        '🙄': 0.9,
        'lol': 0.3,
        'obviously': 0.6,
        'brilliant': 0.5,
        'genius': 0.6,
        'great job': 0.5,
        'what could go wrong': 0.9,
        'to the moon': 0.4,  # Often used sarcastically
    }

    text_lower = text.lower()
    max_sarcasm = 0.0

    for indicator, score in sarcasm_indicators.items():
        if indicator in text_lower:
            max_sarcasm = max(max_sarcasm, score)

    # Check for excessive punctuation (often indicates sarcasm)
    if '?!' in text or '!!!' in text:
        max_sarcasm = max(max_sarcasm, 0.5)

    # Default low probability if no indicators
    return max_sarcasm if max_sarcasm > 0 else 0.05
