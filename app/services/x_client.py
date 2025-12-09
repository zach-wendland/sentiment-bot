"""
X/Twitter API v2 Client

Implements tweet search using the official Twitter API v2.
Uses Bearer token authentication and handles rate limiting.
"""

import logging
import time
from datetime import datetime
from typing import List, Optional, Dict, Any
import httpx
from app.services.types import SocialPost
from app.config import get_settings

logger = logging.getLogger(__name__)

# Twitter API v2 endpoints
TWITTER_API_BASE = "https://api.twitter.com/2"
SEARCH_RECENT_ENDPOINT = f"{TWITTER_API_BASE}/tweets/search/recent"

# Rate limit: 450 requests per 15 minutes = 1 request per 2 seconds
RATE_LIMIT_DELAY = 2.0


def search_x_bundle(inst: dict, since: datetime) -> List[SocialPost]:
    """
    Search X/Twitter for tweets about a stock symbol.

    Args:
        inst: Instrument dict with 'symbol' and 'company_name' keys
        since: Only return tweets created after this datetime

    Returns:
        List of SocialPost objects from X/Twitter
    """
    settings = get_settings()

    if not settings.x_bearer_token:
        logger.warning("X_BEARER_TOKEN not configured, skipping X/Twitter")
        return []

    symbol = inst.get("symbol", "")
    company_name = inst.get("company_name", "")

    if not symbol and not company_name:
        logger.warning("No symbol or company name provided")
        return []

    # Build search query
    # Search for cashtag OR company name, excluding retweets
    query_parts = []
    if symbol:
        query_parts.append(f"${symbol}")
    if company_name and company_name != symbol:
        query_parts.append(f'"{company_name}"')

    query = f"({' OR '.join(query_parts)}) -is:retweet lang:en"

    logger.info(f"Searching X/Twitter with query: {query}")

    posts = []
    next_token = None
    max_pages = 3  # Limit pagination to avoid rate limits

    for page in range(max_pages):
        try:
            result = _fetch_tweets(
                query=query,
                since=since,
                bearer_token=settings.x_bearer_token,
                next_token=next_token
            )

            if not result:
                break

            tweets = result.get("data", [])
            users = {u["id"]: u for u in result.get("includes", {}).get("users", [])}
            meta = result.get("meta", {})

            for tweet in tweets:
                try:
                    post = _parse_tweet(tweet, users)
                    if post:
                        posts.append(post)
                except Exception as e:
                    logger.warning(f"Failed to parse tweet {tweet.get('id')}: {e}")
                    continue

            # Check for more pages
            next_token = meta.get("next_token")
            if not next_token:
                break

            # Rate limit delay between pages
            time.sleep(RATE_LIMIT_DELAY)

        except Exception as e:
            logger.error(f"Error fetching tweets (page {page}): {e}")
            break

    logger.info(f"Retrieved {len(posts)} tweets from X/Twitter for {symbol}")
    return posts


def _fetch_tweets(
    query: str,
    since: datetime,
    bearer_token: str,
    next_token: Optional[str] = None,
    max_results: int = 100
) -> Optional[Dict[str, Any]]:
    """
    Fetch tweets from Twitter API v2.

    Args:
        query: Search query string
        since: Start time for tweet search
        bearer_token: Twitter API bearer token
        next_token: Pagination token for next page
        max_results: Maximum tweets per request (10-100)

    Returns:
        API response dict or None on error
    """
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "User-Agent": "SentimentBot/1.0"
    }

    params = {
        "query": query,
        "max_results": min(max_results, 100),
        "tweet.fields": "id,text,created_at,author_id,public_metrics,lang",
        "user.fields": "id,username,name,public_metrics",
        "expansions": "author_id",
        "start_time": since.strftime("%Y-%m-%dT%H:%M:%SZ")
    }

    if next_token:
        params["next_token"] = next_token

    max_retries = 3
    base_delay = 1.0

    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(
                    SEARCH_RECENT_ENDPOINT,
                    headers=headers,
                    params=params
                )

                if response.status_code == 200:
                    return response.json()

                elif response.status_code == 429:
                    # Rate limited - get reset time from headers
                    reset_time = response.headers.get("x-rate-limit-reset")
                    if reset_time:
                        wait_time = max(0, int(reset_time) - int(time.time())) + 1
                        logger.warning(f"Rate limited, waiting {wait_time}s")
                        time.sleep(min(wait_time, 60))  # Cap at 60s
                    else:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Rate limited, waiting {delay}s")
                        time.sleep(delay)
                    continue

                elif response.status_code == 401:
                    logger.error("Twitter API authentication failed - check X_BEARER_TOKEN")
                    return None

                elif response.status_code == 403:
                    logger.error("Twitter API access forbidden - check API access level")
                    return None

                else:
                    logger.error(f"Twitter API error {response.status_code}: {response.text}")
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        time.sleep(delay)
                    continue

        except httpx.TimeoutException:
            logger.warning(f"Twitter API timeout (attempt {attempt + 1})")
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
            continue

        except Exception as e:
            logger.error(f"Twitter API request failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
            continue

    return None


def _parse_tweet(tweet: Dict[str, Any], users: Dict[str, Dict]) -> Optional[SocialPost]:
    """
    Parse a tweet into a SocialPost object.

    Args:
        tweet: Tweet data from API
        users: Dict of user_id -> user data

    Returns:
        SocialPost or None if parsing fails
    """
    tweet_id = tweet.get("id")
    text = tweet.get("text", "")
    author_id = tweet.get("author_id", "")
    created_at_str = tweet.get("created_at")
    public_metrics = tweet.get("public_metrics", {})

    if not tweet_id or not text:
        return None

    # Parse created_at
    try:
        created_at = datetime.strptime(created_at_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    except (ValueError, TypeError):
        try:
            created_at = datetime.strptime(created_at_str, "%Y-%m-%dT%H:%M:%SZ")
        except (ValueError, TypeError):
            created_at = datetime.utcnow()

    # Get user info
    user = users.get(author_id, {})
    username = user.get("username", "")
    user_metrics = user.get("public_metrics", {})
    follower_count = user_metrics.get("followers_count")

    return SocialPost(
        source="x",
        platform_id=tweet_id,
        author_id=author_id,
        author_handle=username,
        created_at=created_at,
        text=text,
        like_count=public_metrics.get("like_count"),
        reply_count=public_metrics.get("reply_count"),
        repost_count=public_metrics.get("retweet_count"),
        follower_count=follower_count,
        permalink=f"https://twitter.com/{username}/status/{tweet_id}" if username else None,
        lang=tweet.get("lang", "en")
    )
