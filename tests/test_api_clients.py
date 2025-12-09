"""Tests for external API client integrations (with mocked responses)."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import responses

from app.services.x_client import search_x_bundle, _parse_tweet
from app.scrapers.stocktwits_scraper import scrape_stocktwits, StockTwitsScraper


class TestXClient:
    """Tests for X/Twitter API client."""

    def test_x_api_no_token(self):
        """Test X API client returns empty when no token configured."""
        with patch("app.services.x_client.get_settings") as mock_settings:
            mock_settings.return_value.x_bearer_token = ""
            since = datetime.utcnow() - timedelta(days=1)
            result = search_x_bundle({"symbol": "AAPL", "company_name": "Apple"}, since)
            assert result == []

    def test_x_api_no_symbol(self):
        """Test X API client returns empty when no symbol provided."""
        with patch("app.services.x_client.get_settings") as mock_settings:
            mock_settings.return_value.x_bearer_token = "test_token"
            since = datetime.utcnow() - timedelta(days=1)
            result = search_x_bundle({}, since)
            assert result == []

    def test_parse_tweet_basic(self):
        """Test tweet parsing with valid data."""
        tweet = {
            "id": "123456",
            "text": "$AAPL is looking bullish",
            "created_at": "2025-01-15T10:00:00.000Z",
            "author_id": "user123",
            "public_metrics": {
                "like_count": 10,
                "reply_count": 2,
                "retweet_count": 5
            },
            "lang": "en"
        }
        users = {
            "user123": {
                "id": "user123",
                "username": "trader1",
                "public_metrics": {
                    "followers_count": 1000
                }
            }
        }

        post = _parse_tweet(tweet, users)

        assert post is not None
        assert post.source == "x"
        assert post.platform_id == "123456"
        assert post.text == "$AAPL is looking bullish"
        assert post.author_handle == "trader1"
        assert post.like_count == 10
        assert post.follower_count == 1000

    def test_parse_tweet_missing_user(self):
        """Test tweet parsing when user info is missing."""
        tweet = {
            "id": "123456",
            "text": "Test tweet",
            "created_at": "2025-01-15T10:00:00.000Z",
            "author_id": "unknown_user",
            "public_metrics": {}
        }
        users = {}

        post = _parse_tweet(tweet, users)

        assert post is not None
        assert post.author_handle == ""
        assert post.follower_count is None

    def test_parse_tweet_empty(self):
        """Test tweet parsing with missing required fields."""
        tweet = {}
        users = {}

        post = _parse_tweet(tweet, users)
        assert post is None

    @patch("app.services.x_client._fetch_tweets")
    def test_x_api_with_results(self, mock_fetch):
        """Test X API client with mocked results."""
        mock_fetch.return_value = {
            "data": [
                {
                    "id": "123456",
                    "text": "$AAPL is bullish",
                    "created_at": "2025-01-15T10:00:00.000Z",
                    "author_id": "user123",
                    "public_metrics": {"like_count": 10, "reply_count": 2, "retweet_count": 5},
                    "lang": "en"
                }
            ],
            "includes": {
                "users": [
                    {"id": "user123", "username": "trader1", "public_metrics": {"followers_count": 1000}}
                ]
            },
            "meta": {"result_count": 1}
        }

        with patch("app.services.x_client.get_settings") as mock_settings:
            mock_settings.return_value.x_bearer_token = "test_token"
            since = datetime.utcnow() - timedelta(days=1)
            result = search_x_bundle({"symbol": "AAPL", "company_name": "Apple"}, since)

        assert len(result) == 1
        assert result[0].source == "x"
        assert result[0].text == "$AAPL is bullish"

    @patch("app.services.x_client._fetch_tweets")
    def test_x_api_empty_results(self, mock_fetch):
        """Test X API client with no results."""
        mock_fetch.return_value = {"meta": {"result_count": 0}}

        with patch("app.services.x_client.get_settings") as mock_settings:
            mock_settings.return_value.x_bearer_token = "test_token"
            since = datetime.utcnow() - timedelta(days=1)
            result = search_x_bundle({"symbol": "UNKNOWN"}, since)

        assert len(result) == 0

    @patch("app.services.x_client._fetch_tweets")
    def test_x_api_handles_error(self, mock_fetch):
        """Test X API client handles fetch errors."""
        mock_fetch.return_value = None

        with patch("app.services.x_client.get_settings") as mock_settings:
            mock_settings.return_value.x_bearer_token = "test_token"
            since = datetime.utcnow() - timedelta(days=1)
            result = search_x_bundle({"symbol": "AAPL"}, since)

        assert len(result) == 0


class TestStockTwitsScraper:
    """Tests for StockTwits scraper."""

    def test_stocktwits_scraper_instantiation(self):
        """Test StockTwits scraper can be instantiated."""
        scraper = StockTwitsScraper()
        assert scraper.get_name() == "stocktwits"
        assert scraper.rate_limit == 0.33

    def test_stocktwits_invalid_symbol(self):
        """Test StockTwits scraper with invalid symbol."""
        since = datetime.utcnow() - timedelta(days=1)

        # Empty symbol
        result = scrape_stocktwits({"symbol": ""}, since)
        assert len(result) == 0

        # Too long symbol
        result = scrape_stocktwits({"symbol": "TOOLONG"}, since)
        assert len(result) == 0

    @responses.activate
    def test_stocktwits_scraper_basic(self):
        """Test StockTwits scraper with mocked response."""
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
    def test_stocktwits_scraper_no_messages(self):
        """Test StockTwits scraper when no messages found."""
        responses.add(
            responses.GET,
            "https://api.stocktwits.com/api/2/streams/symbol/UNKNOWN.json",
            json={
                "response": {"status": 200},
                "messages": [],
                "cursor": {}
            },
            status=200
        )

        since = datetime.utcnow() - timedelta(days=1)
        result = scrape_stocktwits({"symbol": "UNKNOWN"}, since)

        assert len(result) == 0

    @responses.activate
    def test_stocktwits_handles_api_error(self):
        """Test StockTwits scraper handles API errors."""
        responses.add(
            responses.GET,
            "https://api.stocktwits.com/api/2/streams/symbol/AAPL.json",
            status=500
        )

        since = datetime.utcnow() - timedelta(days=1)
        result = scrape_stocktwits({"symbol": "AAPL"}, since)

        assert isinstance(result, list)
        assert len(result) == 0
