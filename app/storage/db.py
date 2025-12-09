import os
import logging
import time
import psycopg
from psycopg.rows import dict_row
import numpy as np
from datetime import datetime
from typing import Optional, Dict
from contextlib import contextmanager
from app.services.types import SocialPost, SentimentScore
from app.config import get_settings

logger = logging.getLogger(__name__)

# Global connection cache for warm serverless invocations
_connection_cache: Optional[psycopg.Connection] = None


def _get_connection() -> psycopg.Connection:
    """
    Get database connection with retry logic and caching for serverless.

    Supabase uses Supavisor for connection pooling. We cache the connection
    at module level to reuse across warm invocations.
    """
    global _connection_cache

    settings = get_settings()

    # Check if cached connection is still valid
    if _connection_cache is not None:
        try:
            # Test connection with simple query
            with _connection_cache.cursor() as c:
                c.execute("SELECT 1")
            return _connection_cache
        except Exception:
            logger.warning("Cached connection invalid, reconnecting...")
            _connection_cache = None

    # Retry logic with exponential backoff
    max_retries = 3
    base_delay = 0.5

    for attempt in range(max_retries):
        try:
            conn = psycopg.connect(
                settings.database_url,
                autocommit=True,
                connect_timeout=10,
            )
            _connection_cache = conn
            logger.info("Database connection established")
            return conn
        except Exception as e:
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Connection attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                logger.error(f"Failed to connect after {max_retries} attempts")
                raise


@contextmanager
def get_cursor():
    """Context manager for database cursor with automatic connection handling."""
    conn = _get_connection()
    with conn.cursor() as cursor:
        yield cursor


class DB:
    """Database operations for sentiment bot."""

    def __init__(self, skip_schema_init: bool = False):
        """
        Initialize database connection.

        Args:
            skip_schema_init: If True, skip schema initialization (for serverless after first run)
        """
        self.conn = _get_connection()
        if not skip_schema_init:
            self._init_schema()

    def _init_schema(self):
        """Initialize database schema if not exists."""
        schema_path = os.path.join(os.path.dirname(__file__), "schemas.sql")
        try:
            with open(schema_path) as f:
                schema_sql = f.read()
            with self.conn.cursor() as c:
                c.execute(schema_sql)
            logger.debug("Schema initialized")
        except Exception as e:
            # Schema might already exist or pgvector not enabled
            logger.warning(f"Schema initialization warning: {e}")

    def upsert_post(self, p: SocialPost) -> int:
        """Insert or update a social post. Returns the post ID."""
        with self.conn.cursor() as c:
            c.execute("""
                INSERT INTO social_posts
                (source, platform_id, author_id, created_at, text, symbols, urls, lang,
                 reply_to_id, repost_of_id, like_count, reply_count, repost_count,
                 follower_count, permalink)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (source, platform_id) DO UPDATE SET ingested_at = NOW()
                RETURNING id
            """, (
                p.source, p.platform_id, p.author_id, p.created_at, p.text,
                p.symbols, p.urls, p.lang, p.reply_to_id, p.repost_of_id,
                p.like_count, p.reply_count, p.repost_count, p.follower_count, p.permalink
            ))
            return c.fetchone()[0]

    def upsert_sentiment(self, pk: int, s: SentimentScore) -> None:
        """Insert or update sentiment score for a post."""
        with self.conn.cursor() as c:
            c.execute("""
                INSERT INTO sentiment (post_pk, polarity, subjectivity, sarcasm_prob, confidence, model)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (post_pk) DO UPDATE SET
                    polarity = EXCLUDED.polarity,
                    subjectivity = EXCLUDED.subjectivity,
                    sarcasm_prob = EXCLUDED.sarcasm_prob,
                    confidence = EXCLUDED.confidence,
                    model = EXCLUDED.model
            """, (pk, s.polarity, s.subjectivity, s.sarcasm_prob, s.confidence, s.model))

    def upsert_embedding(self, pk: int, emb: np.ndarray) -> None:
        """Insert or update embedding vector for a post."""
        # Convert numpy array to list for pgvector
        emb_list = emb.tolist()
        with self.conn.cursor() as c:
            c.execute("""
                INSERT INTO post_embeddings (post_pk, emb)
                VALUES (%s, %s::vector)
                ON CONFLICT (post_pk) DO UPDATE SET emb = EXCLUDED.emb
            """, (pk, emb_list))

    def aggregate(self, symbol: str, since: datetime) -> Dict:
        """Aggregate sentiment data for a symbol within a time window."""
        with self.conn.cursor() as c:
            c.execute("""
                SELECT
                    COUNT(*) as count,
                    AVG(s.polarity) as avg_polarity,
                    STDDEV(s.polarity) as stddev_polarity,
                    p.source,
                    COUNT(*) as source_count
                FROM social_posts p
                JOIN sentiment s ON s.post_pk = p.id
                WHERE %s = ANY(p.symbols) AND p.created_at >= %s
                GROUP BY p.source
            """, (symbol, since))

            results = c.fetchall()

            if not results:
                return {
                    "symbol": symbol,
                    "window_since": since.isoformat(),
                    "count": 0,
                    "weighted_sentiment": 0.0,
                    "sources": {}
                }

            total_count = sum(r[4] for r in results)
            total_polarity = sum(r[1] * r[4] if r[1] else 0 for r in results)
            source_breakdown = {r[3]: r[4] for r in results}

            return {
                "symbol": symbol,
                "window_since": since.isoformat(),
                "count": total_count,
                "weighted_sentiment": total_polarity / total_count if total_count > 0 else 0.0,
                "sources": source_breakdown
            }

    def cache_resolution(self, query: str, symbol: str, cik: Optional[str],
                        isin: Optional[str], figi: Optional[str], company_name: str):
        """Cache a symbol resolution result."""
        with self.conn.cursor() as c:
            c.execute("""
                INSERT INTO resolver_cache (query, symbol, cik, isin, figi, company_name)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (query) DO UPDATE SET
                    symbol = EXCLUDED.symbol,
                    cik = EXCLUDED.cik,
                    isin = EXCLUDED.isin,
                    figi = EXCLUDED.figi,
                    company_name = EXCLUDED.company_name,
                    cached_at = NOW()
            """, (query, symbol, cik, isin, figi, company_name))

    def get_cached_resolution(self, query: str) -> Optional[Dict]:
        """Get cached symbol resolution if not expired (7 days TTL)."""
        with self.conn.cursor() as c:
            c.execute("""
                SELECT symbol, cik, isin, figi, company_name
                FROM resolver_cache
                WHERE query = %s AND cached_at > NOW() - INTERVAL '7 days'
            """, (query,))
            row = c.fetchone()
            if row:
                return {
                    "symbol": row[0],
                    "cik": row[1],
                    "isin": row[2],
                    "figi": row[3],
                    "company_name": row[4]
                }
            return None

    def close(self):
        """Close database connection (for cleanup)."""
        global _connection_cache
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
        _connection_cache = None
