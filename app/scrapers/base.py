"""
Base Scraper Framework

Provides common functionality for all web scrapers:
- Token bucket rate limiting
- Exponential backoff retry
- User-agent rotation
- robots.txt compliance (optional)
- Structured logging
"""

import logging
import time
import random
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse
import httpx
from app.services.types import SocialPost

logger = logging.getLogger(__name__)

# Common user agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
]


class RateLimiter:
    """
    Token bucket rate limiter.

    Ensures requests don't exceed a specified rate per second.
    """

    def __init__(self, requests_per_second: float = 1.0):
        """
        Initialize rate limiter.

        Args:
            requests_per_second: Maximum requests per second (default: 1.0)
        """
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0

    def wait(self):
        """Wait if necessary to respect rate limit."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def reset(self):
        """Reset the rate limiter."""
        self.last_request_time = 0.0


class BaseScraper(ABC):
    """
    Abstract base class for web scrapers.

    Provides common functionality:
    - HTTP client with configurable timeouts
    - User-agent rotation
    - Exponential backoff retry
    - Rate limiting
    - Response caching (optional)
    """

    def __init__(
        self,
        rate_limit: float = 0.5,  # requests per second
        max_retries: int = 3,
        timeout: float = 30.0,
        respect_robots_txt: bool = True
    ):
        """
        Initialize the scraper.

        Args:
            rate_limit: Maximum requests per second
            max_retries: Maximum retry attempts on failure
            timeout: HTTP request timeout in seconds
            respect_robots_txt: Whether to check robots.txt (not implemented)
        """
        self.rate_limit = rate_limit  # Store rate limit for reference
        self.rate_limiter = RateLimiter(rate_limit)
        self.max_retries = max_retries
        self.timeout = timeout
        self.respect_robots_txt = respect_robots_txt
        self.user_agents = USER_AGENTS  # Expose for testing
        self._robots_cache: Dict[str, bool] = {}

    def get_random_user_agent(self) -> str:
        """Get a random user agent string."""
        return random.choice(USER_AGENTS)

    def get_headers(self, extra_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Get HTTP headers with rotated user agent.

        Args:
            extra_headers: Additional headers to include

        Returns:
            Dict of HTTP headers
        """
        headers = {
            "User-Agent": self.get_random_user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def fetch(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[httpx.Response]:
        """
        Fetch a URL with retry logic and rate limiting.

        Args:
            url: URL to fetch
            method: HTTP method (GET, POST, etc.)
            headers: Custom headers (merged with defaults)
            params: Query parameters
            json_data: JSON body for POST requests

        Returns:
            httpx.Response or None on failure
        """
        # Apply rate limiting
        self.rate_limiter.wait()

        # Merge headers
        request_headers = self.get_headers(headers)

        # Retry loop with exponential backoff
        base_delay = 1.0

        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                    if method.upper() == "GET":
                        response = client.get(url, headers=request_headers, params=params)
                    elif method.upper() == "POST":
                        response = client.post(url, headers=request_headers, params=params, json=json_data)
                    else:
                        raise ValueError(f"Unsupported HTTP method: {method}")

                    # Success
                    if response.status_code == 200:
                        return response

                    # Rate limited
                    elif response.status_code == 429:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(f"Rate limited on {url}, waiting {delay:.1f}s")
                        time.sleep(delay)
                        continue

                    # Client error (4xx except 429)
                    elif 400 <= response.status_code < 500:
                        logger.error(f"Client error {response.status_code} for {url}")
                        return None

                    # Server error (5xx) - retry
                    elif response.status_code >= 500:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Server error {response.status_code} for {url}, retrying in {delay}s")
                        time.sleep(delay)
                        continue

                    # Unexpected status
                    else:
                        logger.warning(f"Unexpected status {response.status_code} for {url}")
                        return response

            except httpx.TimeoutException:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Timeout for {url} (attempt {attempt + 1}), retrying in {delay}s")
                time.sleep(delay)
                continue

            except httpx.RequestError as e:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Request error for {url}: {e}, retrying in {delay}s")
                time.sleep(delay)
                continue

            except Exception as e:
                logger.error(f"Unexpected error fetching {url}: {e}")
                return None

        logger.error(f"Failed to fetch {url} after {self.max_retries} attempts")
        return None

    def fetch_json(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch a URL and parse JSON response.

        Args:
            url: URL to fetch
            headers: Custom headers
            params: Query parameters

        Returns:
            Parsed JSON dict or None on failure
        """
        # Add JSON accept header
        json_headers = {"Accept": "application/json"}
        if headers:
            json_headers.update(headers)

        response = self.fetch(url, headers=json_headers, params=params)
        if response is None:
            return None

        try:
            return response.json()
        except Exception as e:
            logger.error(f"Failed to parse JSON from {url}: {e}")
            return None

    @abstractmethod
    def scrape(self, query: str, since: datetime) -> List[SocialPost]:
        """
        Scrape posts matching a query.

        Args:
            query: Search query (e.g., stock symbol)
            since: Only return posts after this datetime

        Returns:
            List of SocialPost objects
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this scraper for logging."""
        pass
