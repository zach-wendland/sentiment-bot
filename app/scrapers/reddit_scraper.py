"""
Reddit Scraper

Scrapes Reddit posts and comments using Reddit's public JSON endpoints.
No authentication required - uses .json suffix on URLs.
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from app.scrapers.base import BaseScraper
from app.services.types import SocialPost

logger = logging.getLogger(__name__)

# Target subreddits for stock discussion
SUBREDDITS = ["wallstreetbets", "stocks", "investing", "stockmarket"]

# Reddit JSON endpoint patterns
REDDIT_BASE = "https://old.reddit.com"
SEARCH_ENDPOINT = "{base}/r/{subreddit}/search.json"
COMMENTS_ENDPOINT = "{base}/comments/{post_id}.json"


class RedditScraper(BaseScraper):
    """
    Scraper for Reddit posts and comments.

    Uses Reddit's public JSON API (appending .json to URLs).
    No authentication required but respects rate limits.
    """

    def __init__(self, rate_limit: float = 0.5):
        """
        Initialize Reddit scraper.

        Args:
            rate_limit: Requests per second (default: 0.5 = 1 req/2sec)
        """
        super().__init__(rate_limit=rate_limit, max_retries=3, timeout=30.0)
        self.subreddits = SUBREDDITS

    def get_name(self) -> str:
        return "reddit"

    def scrape(self, query: str, since: datetime) -> List[SocialPost]:
        """
        Scrape Reddit for posts matching a query.

        Args:
            query: Search query (e.g., "AAPL" or "Apple")
            since: Only return posts after this datetime

        Returns:
            List of SocialPost objects
        """
        posts = []
        since_timestamp = since.timestamp()

        for subreddit in self.subreddits:
            try:
                subreddit_posts = self._search_subreddit(subreddit, query, since_timestamp)
                posts.extend(subreddit_posts)
                logger.debug(f"Found {len(subreddit_posts)} posts in r/{subreddit}")
            except Exception as e:
                logger.warning(f"Error scraping r/{subreddit}: {e}")
                continue

        logger.info(f"Reddit scraper found {len(posts)} total posts for '{query}'")
        return posts

    def _search_subreddit(
        self,
        subreddit: str,
        query: str,
        since_timestamp: float
    ) -> List[SocialPost]:
        """
        Search a specific subreddit for posts.

        Args:
            subreddit: Subreddit name (without r/)
            query: Search query
            since_timestamp: Unix timestamp for filtering

        Returns:
            List of SocialPost objects
        """
        posts = []
        after = None  # Pagination cursor

        # Fetch up to 3 pages
        for page in range(3):
            url = SEARCH_ENDPOINT.format(base=REDDIT_BASE, subreddit=subreddit)
            params = {
                "q": query,
                "sort": "new",
                "t": "week",  # Last week
                "limit": 100,
                "restrict_sr": "on",
            }

            if after:
                params["after"] = after

            data = self.fetch_json(url, params=params)
            if not data:
                break

            listing = data.get("data", {})
            children = listing.get("children", [])

            if not children:
                break

            for child in children:
                try:
                    post_data = child.get("data", {})
                    post = self._parse_post(post_data, since_timestamp)
                    if post:
                        posts.append(post)

                        # Also fetch top comments for this post
                        comments = self._fetch_comments(post_data.get("id"), since_timestamp)
                        posts.extend(comments)

                except Exception as e:
                    logger.warning(f"Error parsing Reddit post: {e}")
                    continue

            # Get next page cursor
            after = listing.get("after")
            if not after:
                break

        return posts

    def _parse_post(
        self,
        post_data: Dict[str, Any],
        since_timestamp: float
    ) -> Optional[SocialPost]:
        """
        Parse a Reddit post into a SocialPost.

        Args:
            post_data: Post data from Reddit API
            since_timestamp: Minimum timestamp for filtering

        Returns:
            SocialPost or None if filtered out
        """
        created_utc = post_data.get("created_utc", 0)

        # Filter by time
        if created_utc < since_timestamp:
            return None

        post_id = post_data.get("id", "")
        title = post_data.get("title", "")
        selftext = post_data.get("selftext", "")
        author = post_data.get("author", "[deleted]")
        permalink = post_data.get("permalink", "")
        ups = post_data.get("ups", 0)
        num_comments = post_data.get("num_comments", 0)

        # Skip deleted/removed posts
        if author == "[deleted]" or selftext == "[removed]":
            return None

        # Combine title and body
        text = f"{title}\n{selftext}".strip()

        if not text or not post_id:
            return None

        return SocialPost(
            source="reddit",
            platform_id=f"post_{post_id}",
            author_id=author,
            author_handle=author,
            created_at=datetime.utcfromtimestamp(created_utc),
            text=text,
            like_count=ups,
            reply_count=num_comments,
            permalink=f"https://reddit.com{permalink}" if permalink else None,
            lang="en"
        )

    def _fetch_comments(
        self,
        post_id: str,
        since_timestamp: float,
        limit: int = 10
    ) -> List[SocialPost]:
        """
        Fetch top comments for a post.

        Args:
            post_id: Reddit post ID
            since_timestamp: Minimum timestamp for filtering
            limit: Maximum comments to fetch

        Returns:
            List of SocialPost objects for comments
        """
        if not post_id:
            return []

        url = COMMENTS_ENDPOINT.format(base=REDDIT_BASE, post_id=post_id)
        params = {
            "limit": limit,
            "sort": "top",
            "depth": 1,  # Only top-level comments
        }

        data = self.fetch_json(url, params=params)
        if not data or len(data) < 2:
            return []

        comments = []
        comment_listing = data[1].get("data", {}).get("children", [])

        for child in comment_listing[:limit]:
            try:
                if child.get("kind") != "t1":  # t1 = comment
                    continue

                comment_data = child.get("data", {})
                comment = self._parse_comment(comment_data, post_id, since_timestamp)
                if comment:
                    comments.append(comment)

            except Exception as e:
                logger.warning(f"Error parsing Reddit comment: {e}")
                continue

        return comments

    def _parse_comment(
        self,
        comment_data: Dict[str, Any],
        parent_post_id: str,
        since_timestamp: float
    ) -> Optional[SocialPost]:
        """
        Parse a Reddit comment into a SocialPost.

        Args:
            comment_data: Comment data from Reddit API
            parent_post_id: ID of the parent post
            since_timestamp: Minimum timestamp for filtering

        Returns:
            SocialPost or None if filtered out
        """
        created_utc = comment_data.get("created_utc", 0)

        # Filter by time
        if created_utc < since_timestamp:
            return None

        comment_id = comment_data.get("id", "")
        body = comment_data.get("body", "")
        author = comment_data.get("author", "[deleted]")
        permalink = comment_data.get("permalink", "")
        ups = comment_data.get("ups", 0)

        # Skip deleted/removed comments
        if author == "[deleted]" or body in ["[removed]", "[deleted]"]:
            return None

        if not body or not comment_id:
            return None

        return SocialPost(
            source="reddit",
            platform_id=f"comment_{comment_id}",
            author_id=author,
            author_handle=author,
            created_at=datetime.utcfromtimestamp(created_utc),
            text=body,
            like_count=ups,
            reply_to_id=parent_post_id,
            permalink=f"https://reddit.com{permalink}" if permalink else None,
            lang="en"
        )


def scrape_reddit(inst: dict, since: datetime) -> List[SocialPost]:
    """
    Convenience function to scrape Reddit for a stock symbol.

    Args:
        inst: Instrument dict with 'symbol' and 'company_name' keys
        since: Only return posts after this datetime

    Returns:
        List of SocialPost objects
    """
    symbol = inst.get("symbol", "")
    company_name = inst.get("company_name", "")

    if not symbol and not company_name:
        return []

    scraper = RedditScraper()
    posts = []

    # Search for symbol (with cashtag)
    if symbol:
        symbol_posts = scraper.scrape(f"${symbol}", since)
        posts.extend(symbol_posts)

        # Also search without cashtag
        plain_posts = scraper.scrape(symbol, since)
        posts.extend(plain_posts)

    # Search for company name if different from symbol
    if company_name and company_name.upper() != symbol:
        name_posts = scraper.scrape(company_name, since)
        posts.extend(name_posts)

    # Deduplicate by platform_id
    seen_ids = set()
    unique_posts = []
    for post in posts:
        if post.platform_id not in seen_ids:
            seen_ids.add(post.platform_id)
            unique_posts.append(post)

    logger.info(f"Reddit scraper returned {len(unique_posts)} unique posts for {symbol}")
    return unique_posts
