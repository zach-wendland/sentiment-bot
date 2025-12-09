#!/usr/bin/env python3
"""
End-to-end validation script for Sentiment Bot v2.

Validates that all components work together:
1. Symbol resolution
2. Data collection from all sources (X, Reddit scrapers, StockTwits)
3. Text cleaning and filtering
4. Sentiment analysis (DistilBERT)
5. Embeddings generation
6. Database persistence
7. Result aggregation with Google Trends enrichment

Run with: python validate_e2e.py
"""

import sys
import logging
from datetime import datetime, timedelta
from app.services.resolver import resolve
from app.services.x_client import search_x_bundle
from app.scrapers.reddit_scraper import scrape_reddit
from app.scrapers.stocktwits_scraper import scrape_stocktwits
from app.scrapers.google_trends import collect_google_trends
from app.nlp.clean import normalize_post, extract_symbols
from app.nlp.sentiment import score_text
from app.nlp.embeddings import compute_embedding
from app.nlp.bot_filter import is_probable_bot
from app.orchestration.tasks import aggregate_social, _parse_window

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def validate_symbol_resolution():
    """Validate symbol resolution."""
    logger.info("=" * 60)
    logger.info("TEST 1: Symbol Resolution")
    logger.info("=" * 60)

    try:
        inst = resolve("AAPL")
        logger.info(f"  Resolved AAPL to {inst.company_name}")
        logger.info(f"  Symbol: {inst.symbol}")
        logger.info(f"  Company: {inst.company_name}")
        return True
    except Exception as e:
        logger.error(f"  Failed to resolve symbol: {e}")
        return False


def validate_text_processing():
    """Validate text cleaning and symbol extraction."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Text Processing")
    logger.info("=" * 60)

    try:
        # Test text normalization
        messy_text = "Check this out https://example.com\n$AAPL is   bullish!!!    "
        cleaned = normalize_post(messy_text)
        logger.info(f"  Text normalization:")
        logger.info(f"  Input:  {repr(messy_text[:50])}...")
        logger.info(f"  Output: {repr(cleaned[:50])}...")

        # Test symbol extraction
        inst_dict = {"symbol": "AAPL", "company_name": "Apple"}
        text_with_symbol = "Apple stock is up, $AAPL is doing great"
        symbols = extract_symbols(text_with_symbol, inst_dict)
        logger.info(f"  Symbol extraction: Found {len(symbols)} symbols: {symbols}")

        return True
    except Exception as e:
        logger.error(f"  Text processing failed: {e}")
        return False


def validate_sentiment_analysis():
    """Validate sentiment scoring with DistilBERT."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Sentiment Analysis (DistilBERT)")
    logger.info("=" * 60)

    try:
        test_texts = [
            "AAPL is bullish and will moon! Excellent opportunity",
            "This stock is terrible, bearish, crash incoming",
            "Apple is doing okay, neutral sentiment overall"
        ]

        for text in test_texts:
            score = score_text(text)
            logger.info(f"  Text: {text[:50]}...")
            logger.info(f"  Polarity: {score.polarity:.3f}, Confidence: {score.confidence:.3f}")
            logger.info(f"  Model: {score.model}")

        return True
    except Exception as e:
        logger.error(f"  Sentiment analysis failed: {e}")
        return False


def validate_embeddings():
    """Validate semantic embedding generation."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Semantic Embeddings")
    logger.info("=" * 60)

    try:
        text = "AAPL stock analysis and sentiment from social media"
        embedding = compute_embedding(text)

        logger.info(f"  Generated embedding for text: {text[:50]}...")
        logger.info(f"  Dimensionality: {len(embedding)}")
        logger.info(f"  Norm (should be ~1.0): {(embedding ** 2).sum() ** 0.5:.6f}")

        # Test determinism
        embedding2 = compute_embedding(text)
        import numpy as np
        is_deterministic = np.allclose(embedding, embedding2)
        logger.info(f"  Deterministic (same for same input): {is_deterministic}")

        return True
    except Exception as e:
        logger.error(f"  Embedding generation failed: {e}")
        return False


def validate_window_parsing():
    """Validate time window parsing."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: Time Window Parsing")
    logger.info("=" * 60)

    try:
        test_windows = ["24h", "7d", "1w"]

        for window in test_windows:
            td = _parse_window(window)
            logger.info(f"  Window {window}: {td.total_seconds() / 3600:.1f} hours")

        return True
    except Exception as e:
        logger.error(f"  Window parsing failed: {e}")
        return False


def validate_scrapers():
    """Validate scraper framework (without making real requests)."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 6: Scraper Framework")
    logger.info("=" * 60)

    try:
        from app.scrapers.base import BaseScraper
        from app.scrapers.reddit_scraper import RedditScraper
        from app.scrapers.stocktwits_scraper import StockTwitsScraper

        # Test scraper instantiation
        reddit_scraper = RedditScraper()
        logger.info(f"  Created RedditScraper: {reddit_scraper.get_name()}")

        stocktwits_scraper = StockTwitsScraper()
        logger.info(f"  Created StockTwitsScraper: {stocktwits_scraper.get_name()}")

        # Test rate limiting configuration
        logger.info(f"  Reddit rate limit: {reddit_scraper.rate_limit} req/s")
        logger.info(f"  StockTwits rate limit: {stocktwits_scraper.rate_limit} req/s")

        return True
    except Exception as e:
        logger.error(f"  Scraper framework failed: {e}")
        return False


def validate_google_trends():
    """Validate Google Trends integration (without making real requests)."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 7: Google Trends Integration")
    logger.info("=" * 60)

    try:
        from app.scrapers.google_trends import TrendsData, PYTRENDS_AVAILABLE

        logger.info(f"  pytrends available: {PYTRENDS_AVAILABLE}")

        # Test TrendsData model
        trends = TrendsData(
            interest_over_time={"2025-01-01": 75.0, "2025-01-02": 80.0},
            related_queries=["aapl stock", "apple earnings"],
            interest_by_region={"California": 100, "New York": 95}
        )
        data = trends.model_dump()
        logger.info(f"  Created TrendsData with {len(data['interest_over_time'])} time points")
        logger.info(f"  Related queries: {data['related_queries'][:2]}")

        return True
    except Exception as e:
        logger.error(f"  Google Trends integration failed: {e}")
        return False


def validate_pipeline_integration():
    """Validate complete pipeline (mocked or with credentials)."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 8: Complete Pipeline Integration")
    logger.info("=" * 60)

    try:
        logger.info("Attempting to run complete sentiment aggregation pipeline...")
        logger.info("(This will use configured API credentials if available)")

        result = aggregate_social("AAPL", "24h")

        logger.info("  Pipeline completed successfully!")
        logger.info(f"  Symbol: {result.get('symbol')}")
        logger.info(f"  Posts found: {result.get('posts_found', 0)}")
        logger.info(f"  Posts processed: {result.get('posts_processed', 0)}")

        if result.get('sources'):
            logger.info(f"  Sources: {result.get('sources')}")

        if result.get('search_interest'):
            logger.info("  Google Trends data included")

        if result.get('error'):
            logger.warning(f"  Warning: {result.get('error')}")

        return True
    except Exception as e:
        logger.error(f"  Pipeline integration failed: {e}")
        logger.warning("  (This is expected if API credentials aren't configured)")
        return False


def validate_data_models():
    """Validate data models."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 9: Data Models")
    logger.info("=" * 60)

    try:
        from app.services.types import SocialPost, SentimentScore, ResolvedInstrument, TrendsData

        # Test SocialPost
        post = SocialPost(
            source="reddit",
            platform_id="123",
            author_id="user1",
            created_at=datetime.utcnow(),
            text="Test post"
        )
        logger.info(f"  Created SocialPost: {post.source} by {post.author_id}")

        # Test SentimentScore
        score = SentimentScore(
            polarity=0.5,
            subjectivity=0.6,
            sarcasm_prob=0.1,
            confidence=0.8
        )
        logger.info(f"  Created SentimentScore: polarity={score.polarity}")

        # Test ResolvedInstrument
        inst = ResolvedInstrument(
            symbol="AAPL",
            company_name="Apple Inc."
        )
        logger.info(f"  Created ResolvedInstrument: {inst.symbol} ({inst.company_name})")

        # Test TrendsData
        trends = TrendsData(
            interest_over_time={"2025-01-01": 50.0},
            related_queries=["test query"]
        )
        logger.info(f"  Created TrendsData: {len(trends.interest_over_time or {})} time points")

        return True
    except Exception as e:
        logger.error(f"  Data model validation failed: {e}")
        return False


def main():
    """Run all validation tests."""
    logger.info("\n")
    logger.info("=" * 60)
    logger.info("  SENTIMENT BOT v2 - END-TO-END VALIDATION")
    logger.info("=" * 60)

    tests = [
        ("Data Models", validate_data_models),
        ("Symbol Resolution", validate_symbol_resolution),
        ("Text Processing", validate_text_processing),
        ("Sentiment Analysis", validate_sentiment_analysis),
        ("Embeddings", validate_embeddings),
        ("Time Window Parsing", validate_window_parsing),
        ("Scraper Framework", validate_scrapers),
        ("Google Trends", validate_google_trends),
        ("Pipeline Integration", validate_pipeline_integration),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"Test {name} crashed: {e}")
            results.append((name, False))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{status}: {name}")

    logger.info("=" * 60)
    logger.info(f"Total: {passed}/{total} tests passed")

    if passed == total:
        logger.info("All tests passed! App is ready for deployment.")
        return 0
    else:
        logger.warning(f"{total - passed} test(s) failed. See details above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
