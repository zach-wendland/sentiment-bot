"""Tests for the web scraper framework and implementations."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import responses

from app.scrapers.base import BaseScraper
from app.scrapers.reddit_scraper import RedditScraper, scrape_reddit
from app.scrapers.stocktwits_scraper import StockTwitsScraper, scrape_stocktwits
from app.scrapers.google_trends import collect_google_trends, TrendsData, PYTRENDS_AVAILABLE


class TestBaseScraper:
    """Tests for the base scraper framework."""

    def test_base_scraper_is_abstract(self):
        """Test that BaseScraper cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseScraper()

    def test_user_agent_rotation(self):
        """Test that scrapers have multiple user agents."""
        scraper = RedditScraper()
        assert len(scraper.user_agents) >= 5

    def test_rate_limit_configuration(self):
        """Test rate limit can be configured."""
        scraper = RedditScraper(rate_limit=1.0)
        assert scraper.rate_limit == 1.0

        scraper2 = RedditScraper(rate_limit=0.5)
        assert scraper2.rate_limit == 0.5


class TestRedditScraper:
    """Tests for the Reddit scraper."""

    def test_scraper_name(self):
        """Test scraper returns correct name."""
        scraper = RedditScraper()
        assert scraper.get_name() == "reddit"

    def test_default_rate_limit(self):
        """Test default rate limit is set."""
        scraper = RedditScraper()
        assert scraper.rate_limit == 0.5  # 1 req per 2 sec

    def test_subreddits_configured(self):
        """Test subreddits are configured."""
        scraper = RedditScraper()
        assert len(scraper.subreddits) > 0
        assert "wallstreetbets" in scraper.subreddits

    @responses.activate
    def test_scrape_reddit_basic(self):
        """Test basic Reddit scraping with mocked response."""
        # Mock Reddit JSON API response
        responses.add(
            responses.GET,
            "https://old.reddit.com/r/wallstreetbets/search.json",
            json={
                "data": {
                    "children": [
                        {
                            "data": {
                                "id": "post123",
                                "title": "AAPL is looking bullish",
                                "selftext": "I think Apple will moon",
                                "author": "investor1",
                                "created_utc": datetime.utcnow().timestamp(),
                                "score": 100,
                                "num_comments": 50,
                                "permalink": "/r/wallstreetbets/comments/post123"
                            }
                        }
                    ]
                }
            },
            status=200
        )

        # Mock other subreddits to return empty
        for sub in ["stocks", "investing", "stockmarket"]:
            responses.add(
                responses.GET,
                f"https://old.reddit.com/r/{sub}/search.json",
                json={"data": {"children": []}},
                status=200
            )

        since = datetime.utcnow() - timedelta(days=1)
        result = scrape_reddit({"symbol": "AAPL", "company_name": "Apple"}, since)

        assert len(result) >= 1
        assert result[0].source == "reddit"
        assert "AAPL" in result[0].text or "Apple" in result[0].text

    @responses.activate
    def test_scrape_reddit_empty(self):
        """Test Reddit scraping when no posts found."""
        # Mock all subreddits to return empty
        for sub in ["wallstreetbets", "stocks", "investing", "stockmarket"]:
            responses.add(
                responses.GET,
                f"https://old.reddit.com/r/{sub}/search.json",
                json={"data": {"children": []}},
                status=200
            )

        since = datetime.utcnow() - timedelta(days=1)
        result = scrape_reddit({"symbol": "UNKNOWN"}, since)

        assert len(result) == 0

    def test_scrape_reddit_no_symbol(self):
        """Test Reddit scraping with no symbol returns empty."""
        since = datetime.utcnow() - timedelta(days=1)
        result = scrape_reddit({}, since)
        assert len(result) == 0


class TestStockTwitsScraper:
    """Tests for the StockTwits scraper."""

    def test_scraper_name(self):
        """Test scraper returns correct name."""
        scraper = StockTwitsScraper()
        assert scraper.get_name() == "stocktwits"

    def test_default_rate_limit(self):
        """Test default rate limit is set."""
        scraper = StockTwitsScraper()
        assert scraper.rate_limit == 0.33  # 1 req per 3 sec

    def test_invalid_symbol_rejected(self):
        """Test invalid symbols are rejected."""
        since = datetime.utcnow() - timedelta(days=1)

        # Empty symbol
        result = scrape_stocktwits({"symbol": ""}, since)
        assert len(result) == 0

        # Symbol too long
        result = scrape_stocktwits({"symbol": "TOOLONG"}, since)
        assert len(result) == 0

    @responses.activate
    def test_scrape_stocktwits_basic(self):
        """Test basic StockTwits scraping with mocked response."""
        responses.add(
            responses.GET,
            "https://api.stocktwits.com/api/2/streams/symbol/AAPL.json",
            json={
                "response": {"status": 200},
                "symbol": {"symbol": "AAPL"},
                "messages": [
                    {
                        "id": 123456,
                        "body": "AAPL looking strong today",
                        "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "entities": {"sentiment": {"basic": "bullish"}},
                        "likes": {"total": 10},
                        "user": {
                            "id": 1,
                            "username": "trader1",
                            "followers": 500
                        }
                    }
                ],
                "cursor": {}
            },
            status=200
        )

        since = datetime.utcnow() - timedelta(days=1)
        result = scrape_stocktwits({"symbol": "AAPL"}, since)

        assert len(result) == 1
        assert result[0].source == "stocktwits"
        assert "[BULLISH]" in result[0].text

    @responses.activate
    def test_scrape_stocktwits_no_sentiment(self):
        """Test StockTwits scraping with no sentiment tag."""
        responses.add(
            responses.GET,
            "https://api.stocktwits.com/api/2/streams/symbol/AAPL.json",
            json={
                "response": {"status": 200},
                "messages": [
                    {
                        "id": 123456,
                        "body": "Just bought some AAPL",
                        "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "entities": {"sentiment": {}},
                        "likes": {"total": 5},
                        "user": {"id": 1, "username": "user1", "followers": 100}
                    }
                ],
                "cursor": {}
            },
            status=200
        )

        since = datetime.utcnow() - timedelta(days=1)
        result = scrape_stocktwits({"symbol": "AAPL"}, since)

        assert len(result) == 1
        assert "[BULLISH]" not in result[0].text
        assert "[BEARISH]" not in result[0].text


class TestGoogleTrends:
    """Tests for Google Trends integration."""

    def test_trends_data_model(self):
        """Test TrendsData model creation."""
        trends = TrendsData(
            interest_over_time={"2025-01-01": 75.0, "2025-01-02": 80.0},
            related_queries=["aapl stock", "apple earnings"],
            interest_by_region={"California": 100, "New York": 95}
        )

        assert trends.interest_over_time["2025-01-01"] == 75.0
        assert len(trends.related_queries) == 2
        assert trends.interest_by_region["California"] == 100

    def test_trends_data_model_dump(self):
        """Test TrendsData serialization."""
        trends = TrendsData(
            interest_over_time={"2025-01-01": 75.0},
            related_queries=["test"],
            interest_by_region={"CA": 100}
        )

        data = trends.model_dump()
        assert "interest_over_time" in data
        assert "related_queries" in data
        assert "interest_by_region" in data
        assert "fetched_at" in data

    def test_trends_data_defaults(self):
        """Test TrendsData with default values."""
        trends = TrendsData()

        assert trends.interest_over_time == {}
        assert trends.related_queries == []
        assert trends.interest_by_region == {}
        assert trends.fetched_at is not None

    def test_pytrends_availability(self):
        """Test pytrends availability flag."""
        # This just verifies the import worked
        assert isinstance(PYTRENDS_AVAILABLE, bool)

    def test_collect_google_trends_no_symbol(self):
        """Test Google Trends with no symbol returns None."""
        result = collect_google_trends("")
        assert result is None

    @patch("app.scrapers.google_trends.PYTRENDS_AVAILABLE", False)
    def test_collect_google_trends_unavailable(self):
        """Test Google Trends when pytrends not available."""
        result = collect_google_trends("AAPL")
        assert result is None


class TestScraperErrorHandling:
    """Tests for scraper error handling."""

    @responses.activate
    def test_reddit_handles_rate_limit(self):
        """Test Reddit scraper handles 429 rate limit."""
        for sub in ["wallstreetbets", "stocks", "investing", "stockmarket"]:
            responses.add(
                responses.GET,
                f"https://old.reddit.com/r/{sub}/search.json",
                status=429
            )

        since = datetime.utcnow() - timedelta(days=1)
        # Should not raise, returns empty
        result = scrape_reddit({"symbol": "AAPL"}, since)
        assert isinstance(result, list)

    @responses.activate
    def test_stocktwits_handles_error(self):
        """Test StockTwits scraper handles API errors."""
        responses.add(
            responses.GET,
            "https://api.stocktwits.com/api/2/streams/symbol/AAPL.json",
            status=500
        )

        since = datetime.utcnow() - timedelta(days=1)
        # Should not raise, returns empty
        result = scrape_stocktwits({"symbol": "AAPL"}, since)
        assert isinstance(result, list)
        assert len(result) == 0

    @responses.activate
    def test_reddit_handles_malformed_json(self):
        """Test Reddit scraper handles malformed responses."""
        for sub in ["wallstreetbets", "stocks", "investing", "stockmarket"]:
            responses.add(
                responses.GET,
                f"https://old.reddit.com/r/{sub}/search.json",
                body="not valid json",
                status=200
            )

        since = datetime.utcnow() - timedelta(days=1)
        result = scrape_reddit({"symbol": "AAPL"}, since)
        assert isinstance(result, list)
