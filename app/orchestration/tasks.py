"""
Pipeline Orchestration

Coordinates the complete sentiment analysis pipeline:
1. Symbol resolution
2. Multi-source data collection (X, Reddit, StockTwits)
3. Text cleaning and filtering
4. Sentiment scoring
5. Embedding generation
6. Database persistence
7. Result aggregation with Google Trends enrichment
"""

import datetime as dt
import logging
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.services.resolver import resolve
from app.services.x_client import search_x_bundle
from app.scrapers.reddit_scraper import scrape_reddit
from app.scrapers.stocktwits_scraper import scrape_stocktwits
from app.scrapers.google_trends import collect_google_trends
from app.nlp.clean import normalize_post, extract_symbols
from app.nlp.sentiment import score_text
from app.nlp.embeddings import compute_embedding
from app.nlp.bot_filter import is_probable_bot
from app.storage.db import DB
from app.services.types import SocialPost
from app.config import get_settings

logger = logging.getLogger(__name__)


def healthcheck() -> Dict:
    """Health check with timestamp and component status."""
    try:
        settings = get_settings()
        return {
            "status": "ok",
            "timestamp": dt.datetime.utcnow().isoformat(),
            "version": "2.0.0",
            "components": {
                "x_api": bool(settings.x_bearer_token),
                "google_trends": settings.google_trends_enabled,
                "dry_run": settings.dry_run
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "timestamp": dt.datetime.utcnow().isoformat(),
            "error": str(e)
        }


def aggregate_social(symbol: str, window: str = "24h") -> Dict:
    """
    Complete pipeline: resolve symbol -> collect posts -> clean -> score -> aggregate.

    Collects data from multiple sources in parallel:
    - X/Twitter (official API v2)
    - Reddit (scraper)
    - StockTwits (scraper)

    Enriches results with Google Trends data.

    Args:
        symbol: Stock symbol to analyze
        window: Time window for analysis (e.g., "24h", "7d")

    Returns:
        Dictionary with aggregated sentiment results

    Raises:
        ValueError: If symbol cannot be resolved
    """
    logger.info(f"Starting sentiment aggregation for symbol={symbol}, window={window}")
    settings = get_settings()

    # Resolve symbol
    try:
        inst = resolve(symbol)
        inst_dict = inst.model_dump()
        logger.debug(f"Resolved {symbol} to {inst.company_name}")
    except Exception as e:
        logger.error(f"Failed to resolve symbol {symbol}: {e}")
        raise ValueError(f"Could not resolve symbol: {symbol}")

    # Parse time window
    try:
        since = dt.datetime.utcnow() - _parse_window(window)
        logger.debug(f"Analyzing posts since {since.isoformat()}")
    except Exception as e:
        logger.error(f"Failed to parse window {window}: {e}")
        raise ValueError(f"Invalid time window: {window}")

    # Collect from all sources (parallel execution)
    posts: List[SocialPost] = []
    sources_status = {}

    # Define collection tasks
    def collect_x():
        if not settings.x_bearer_token:
            logger.warning("X/Twitter API not configured (no bearer token)")
            return [], "x", 0
        try:
            x_posts = search_x_bundle(inst_dict, since)
            return x_posts, "x", len(x_posts)
        except Exception as e:
            logger.warning(f"Failed to collect from X/Twitter: {e}")
            return [], "x", 0

    def collect_reddit():
        try:
            reddit_posts = scrape_reddit(inst_dict, since)
            return reddit_posts, "reddit", len(reddit_posts)
        except Exception as e:
            logger.warning(f"Failed to collect from Reddit: {e}")
            return [], "reddit", 0

    def collect_stocktwits():
        try:
            st_posts = scrape_stocktwits(inst_dict, since)
            return st_posts, "stocktwits", len(st_posts)
        except Exception as e:
            logger.warning(f"Failed to collect from StockTwits: {e}")
            return [], "stocktwits", 0

    # Execute collections in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(collect_x),
            executor.submit(collect_reddit),
            executor.submit(collect_stocktwits)
        ]

        for future in as_completed(futures):
            try:
                source_posts, source_name, count = future.result()
                posts.extend(source_posts)
                sources_status[source_name] = count
                logger.info(f"Collected {count} posts from {source_name}")
            except Exception as e:
                logger.error(f"Collection task failed: {e}")

    logger.info(f"Total posts collected: {len(posts)}")

    if not posts:
        logger.warning(f"No posts found for {symbol}")
        return {
            "symbol": symbol,
            "posts_found": 0,
            "posts_processed": 0,
            "sources": sources_status,
            "resolved_instrument": inst_dict,
            "search_interest": None,
            "error": "No posts found for this symbol"
        }

    # Clean and filter
    clean_posts = []
    filter_stats = {
        "total_input": len(posts),
        "no_symbols": 0,
        "probable_bots": 0,
        "processed": 0
    }

    for p in posts:
        try:
            # Normalize text
            p.text = normalize_post(p.text)

            # Extract symbols
            p.symbols = list(set(extract_symbols(p.text, inst_dict)))

            # Filter out posts with no symbols or probable bots
            if not p.symbols:
                filter_stats["no_symbols"] += 1
                continue

            if is_probable_bot(p):
                filter_stats["probable_bots"] += 1
                continue

            clean_posts.append(p)
            filter_stats["processed"] += 1

        except Exception as e:
            logger.warning(f"Failed to clean post from {p.source}: {e}")
            continue

    logger.info(f"Cleaned posts: {filter_stats['processed']}/{filter_stats['total_input']} "
                f"(filtered: {filter_stats['no_symbols']} no symbols, {filter_stats['probable_bots']} bots)")

    if not clean_posts:
        logger.warning(f"No posts passed filtering for {symbol}")
        return {
            "symbol": symbol,
            "posts_found": len(posts),
            "posts_processed": 0,
            "sources": sources_status,
            "resolved_instrument": inst_dict,
            "search_interest": None,
            "filter_stats": filter_stats,
            "error": "No valid posts after filtering"
        }

    # Persist, score, and embed (skip if dry run)
    processed_count = 0

    if settings.dry_run:
        logger.info("DRY RUN: Skipping database operations")
        processed_count = len(clean_posts)
    else:
        db = DB()
        for p in clean_posts:
            try:
                # Upsert post
                pk = db.upsert_post(p)

                # Score sentiment
                sentiment = score_text(p.text)
                db.upsert_sentiment(pk, sentiment)

                # Compute and store embedding
                emb = compute_embedding(p.text)
                db.upsert_embedding(pk, emb)

                processed_count += 1
            except Exception as e:
                logger.warning(f"Failed to process post {p.platform_id} from {p.source}: {e}")
                continue

    logger.info(f"Successfully processed {processed_count} posts for {symbol}")

    # Fetch Google Trends data (enrichment)
    search_interest = None
    if settings.google_trends_enabled:
        try:
            trends_data = collect_google_trends(inst.symbol, inst.company_name)
            if trends_data:
                search_interest = trends_data.model_dump()
                logger.info(f"Collected Google Trends data for {symbol}")
        except Exception as e:
            logger.warning(f"Failed to collect Google Trends for {symbol}: {e}")

    # Aggregate results
    try:
        if settings.dry_run:
            # Return mock aggregation for dry run
            result = {
                "symbol": inst.symbol,
                "avg_polarity": 0.0,
                "avg_subjectivity": 0.0,
                "avg_confidence": 0.0,
                "total_posts": processed_count,
                "dry_run": True
            }
        else:
            db = DB()
            result = db.aggregate(inst.symbol, since)

        result["resolved_instrument"] = inst_dict
        result["posts_found"] = len(posts)
        result["posts_processed"] = processed_count
        result["sources"] = sources_status
        result["search_interest"] = search_interest
        result["filter_stats"] = filter_stats

        logger.info(f"Aggregation complete for {symbol}: {processed_count} posts")
        return result

    except Exception as e:
        logger.error(f"Aggregation failed for {symbol}: {e}")
        raise


def _parse_window(window: str) -> dt.timedelta:
    """
    Parse time window string to timedelta.

    Args:
        window: String like "24h", "7d", or "1w"

    Returns:
        timedelta object

    Raises:
        ValueError: If window format is invalid
    """
    try:
        n = int(window[:-1])
        unit = window[-1].lower()

        if unit == 'h':
            return dt.timedelta(hours=n)
        elif unit == 'd':
            return dt.timedelta(days=n)
        elif unit == 'w':
            return dt.timedelta(weeks=n)
        else:
            logger.warning(f"Unknown time unit {unit}, defaulting to 24h")
            return dt.timedelta(hours=24)
    except (ValueError, IndexError) as e:
        logger.error(f"Failed to parse window {window}: {e}")
        raise ValueError(f"Invalid window format: {window}. Use format like '24h', '7d', or '1w'.")
