"""
StockTwits Scraper

Scrapes StockTwits for stock symbol discussions using their public API.
The public API doesn't require authentication but has rate limits.
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from app.scrapers.base import BaseScraper
from app.services.types import SocialPost

logger = logging.getLogger(__name__)

# StockTwits public API endpoint
STOCKTWITS_API_BASE = "https://api.stocktwits.com/api/2"
SYMBOL_STREAM_ENDPOINT = f"{STOCKTWITS_API_BASE}/streams/symbol/{{symbol}}.json"


class StockTwitsScraper(BaseScraper):
    """
    Scraper for StockTwits messages.

    Uses StockTwits public API for symbol streams.
    No authentication required for basic access.
    """

    def __init__(self, rate_limit: float = 0.33):
        """
        Initialize StockTwits scraper.

        Args:
            rate_limit: Requests per second (default: 0.33 = 1 req/3sec)
        """
        super().__init__(rate_limit=rate_limit, max_retries=3, timeout=30.0)

    def get_name(self) -> str:
        return "stocktwits"

    def scrape(self, query: str, since: datetime) -> List[SocialPost]:
        """
        Scrape StockTwits for messages about a symbol.

        Args:
            query: Stock symbol (e.g., "AAPL")
            since: Only return messages after this datetime

        Returns:
            List of SocialPost objects
        """
        # StockTwits works with symbols, not free text
        # Clean the query to get just the symbol
        symbol = query.strip().upper().lstrip("$")

        if not symbol or len(symbol) > 5:
            logger.warning(f"Invalid symbol for StockTwits: {query}")
            return []

        posts = []
        since_timestamp = since.timestamp()
        max_id = None  # For pagination

        # Fetch up to 3 pages
        for page in range(3):
            try:
                result = self._fetch_symbol_stream(symbol, max_id)
                if not result:
                    break

                messages = result.get("messages", [])
                if not messages:
                    break

                page_posts = []
                for msg in messages:
                    try:
                        post = self._parse_message(msg, since_timestamp)
                        if post:
                            page_posts.append(post)
                    except Exception as e:
                        logger.warning(f"Error parsing StockTwits message: {e}")
                        continue

                posts.extend(page_posts)

                # If no posts passed the time filter, stop pagination
                if not page_posts:
                    break

                # Get cursor for next page
                cursor = result.get("cursor", {})
                max_id = cursor.get("max")
                if not max_id:
                    break

            except Exception as e:
                logger.error(f"Error fetching StockTwits page {page} for {symbol}: {e}")
                break

        logger.info(f"StockTwits scraper found {len(posts)} messages for {symbol}")
        return posts

    def _fetch_symbol_stream(
        self,
        symbol: str,
        max_id: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch symbol stream from StockTwits API.

        Args:
            symbol: Stock symbol
            max_id: Maximum message ID for pagination

        Returns:
            API response dict or None on error
        """
        url = SYMBOL_STREAM_ENDPOINT.format(symbol=symbol)
        params = {"limit": 30}  # Max 30 per request

        if max_id:
            params["max"] = max_id

        return self.fetch_json(url, params=params)

    def _parse_message(
        self,
        msg: Dict[str, Any],
        since_timestamp: float
    ) -> Optional[SocialPost]:
        """
        Parse a StockTwits message into a SocialPost.

        Args:
            msg: Message data from StockTwits API
            since_timestamp: Minimum timestamp for filtering

        Returns:
            SocialPost or None if filtered out
        """
        # Parse timestamp
        created_at_str = msg.get("created_at", "")
        try:
            # StockTwits format: "2025-01-15T10:00:00Z"
            created_at = datetime.strptime(created_at_str, "%Y-%m-%dT%H:%M:%SZ")
        except (ValueError, TypeError):
            try:
                # Alternative format
                created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                created_at = created_at.replace(tzinfo=None)
            except (ValueError, TypeError):
                created_at = datetime.utcnow()

        # Filter by time
        if created_at.timestamp() < since_timestamp:
            return None

        msg_id = msg.get("id")
        body = msg.get("body", "")
        user = msg.get("user", {})
        sentiment = msg.get("entities", {}).get("sentiment", {})
        likes = msg.get("likes", {})

        if not msg_id or not body:
            return None

        # Get user info
        user_id = str(user.get("id", ""))
        username = user.get("username", "")
        followers = user.get("followers", 0)

        # Get sentiment tag if present
        sentiment_basic = sentiment.get("basic")
        if sentiment_basic:
            # Add sentiment tag to text
            sentiment_label = f"[{sentiment_basic.upper()}]"
            text = f"{sentiment_label} {body}"
        else:
            text = body

        # Get like count
        like_count = likes.get("total", 0) if isinstance(likes, dict) else 0

        return SocialPost(
            source="stocktwits",
            platform_id=str(msg_id),
            author_id=user_id,
            author_handle=username,
            created_at=created_at,
            text=text,
            like_count=like_count,
            follower_count=followers,
            permalink=f"https://stocktwits.com/{username}/message/{msg_id}" if username else None,
            lang="en"
        )


def scrape_stocktwits(inst: dict, since: datetime) -> List[SocialPost]:
    """
    Convenience function to scrape StockTwits for a stock symbol.

    Args:
        inst: Instrument dict with 'symbol' key
        since: Only return messages after this datetime

    Returns:
        List of SocialPost objects
    """
    symbol = inst.get("symbol", "")

    if not symbol:
        return []

    scraper = StockTwitsScraper()
    posts = scraper.scrape(symbol, since)

    logger.info(f"StockTwits scraper returned {len(posts)} posts for {symbol}")
    return posts
