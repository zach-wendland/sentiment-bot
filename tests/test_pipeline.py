"""Tests for the orchestration pipeline."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from app.orchestration.tasks import aggregate_social, _parse_window, healthcheck


def test_parse_window_hours():
    """Test time window parsing for hours."""
    result = _parse_window("24h")
    assert result.total_seconds() == 24 * 3600


def test_parse_window_days():
    """Test time window parsing for days."""
    result = _parse_window("7d")
    assert result.total_seconds() == 7 * 24 * 3600


def test_parse_window_weeks():
    """Test time window parsing for weeks."""
    result = _parse_window("1w")
    assert result.total_seconds() == 7 * 24 * 3600


def test_parse_window_invalid():
    """Test invalid window format raises ValueError."""
    with pytest.raises(ValueError):
        _parse_window("invalid")


def test_parse_window_default():
    """Test default fallback for unknown unit."""
    result = _parse_window("5x")  # Unknown unit 'x'
    assert result.total_seconds() == 24 * 3600  # Should default to 24h


def test_healthcheck():
    """Test health check returns ok status."""
    result = healthcheck()
    assert result["status"] == "ok"
    assert "timestamp" in result
    assert "version" in result
    assert "components" in result


def test_healthcheck_components():
    """Test health check includes component status."""
    result = healthcheck()
    assert "x_api" in result["components"]
    assert "google_trends" in result["components"]
    assert "dry_run" in result["components"]


@patch("app.orchestration.tasks.resolve")
@patch("app.orchestration.tasks.search_x_bundle")
@patch("app.orchestration.tasks.scrape_reddit")
@patch("app.orchestration.tasks.scrape_stocktwits")
@patch("app.orchestration.tasks.collect_google_trends")
@patch("app.orchestration.tasks.DB")
@patch("app.orchestration.tasks.get_settings")
def test_aggregate_social_complete_pipeline(
    mock_settings, mock_db, mock_trends, mock_st, mock_reddit, mock_x, mock_resolve
):
    """Test complete aggregation pipeline with mocked dependencies."""
    # Mock settings
    mock_settings.return_value.x_bearer_token = "test_token"
    mock_settings.return_value.google_trends_enabled = True
    mock_settings.return_value.dry_run = False

    # Mock symbol resolution
    mock_inst = MagicMock()
    mock_inst.symbol = "AAPL"
    mock_inst.company_name = "Apple Inc."
    mock_inst.model_dump.return_value = {
        "symbol": "AAPL",
        "company_name": "Apple Inc.",
        "cik": None,
        "isin": None,
        "figi": None
    }
    mock_resolve.return_value = mock_inst

    # Mock data collection
    from app.services.types import SocialPost
    test_post = SocialPost(
        source="x",
        platform_id="123",
        author_id="user1",
        author_handle="trader1",
        created_at=datetime.utcnow(),
        text="$AAPL looking bullish",
        like_count=10
    )
    mock_x.return_value = [test_post]
    mock_reddit.return_value = []
    mock_st.return_value = []

    # Mock Google Trends
    mock_trends.return_value = None

    # Mock database
    mock_db_inst = MagicMock()
    mock_db.return_value = mock_db_inst
    mock_db_inst.upsert_post.return_value = 1
    mock_db_inst.aggregate.return_value = {
        "symbol": "AAPL",
        "posts_count": 1,
        "avg_sentiment": 0.5
    }

    # Call pipeline
    result = aggregate_social("AAPL", "24h")

    # Verify results
    assert result is not None
    assert "symbol" in result
    assert "sources" in result
    assert result["sources"]["x"] == 1


@patch("app.orchestration.tasks.resolve")
def test_aggregate_social_symbol_not_found(mock_resolve):
    """Test pipeline when symbol resolution fails."""
    mock_resolve.side_effect = ValueError("Symbol not found")

    with pytest.raises(ValueError):
        aggregate_social("INVALID")


@patch("app.orchestration.tasks.resolve")
@patch("app.orchestration.tasks.search_x_bundle")
@patch("app.orchestration.tasks.scrape_reddit")
@patch("app.orchestration.tasks.scrape_stocktwits")
@patch("app.orchestration.tasks.get_settings")
def test_aggregate_social_no_posts(
    mock_settings, mock_st, mock_reddit, mock_x, mock_resolve
):
    """Test pipeline when no posts collected from any source."""
    # Mock settings
    mock_settings.return_value.x_bearer_token = "test_token"
    mock_settings.return_value.google_trends_enabled = False
    mock_settings.return_value.dry_run = False

    mock_inst = MagicMock()
    mock_inst.symbol = "UNKNOWN"
    mock_inst.company_name = "Unknown Corp"
    mock_inst.model_dump.return_value = {
        "symbol": "UNKNOWN",
        "company_name": "Unknown Corp"
    }
    mock_resolve.return_value = mock_inst

    # All sources return empty
    mock_x.return_value = []
    mock_reddit.return_value = []
    mock_st.return_value = []

    result = aggregate_social("UNKNOWN", "24h")

    # Should return graceful error
    assert result["posts_found"] == 0
    assert result["posts_processed"] == 0
    assert "error" in result


@patch("app.orchestration.tasks.resolve")
@patch("app.orchestration.tasks.search_x_bundle")
@patch("app.orchestration.tasks.scrape_reddit")
@patch("app.orchestration.tasks.scrape_stocktwits")
@patch("app.orchestration.tasks.get_settings")
def test_aggregate_social_partial_failure(
    mock_settings, mock_st, mock_reddit, mock_x, mock_resolve
):
    """Test pipeline continues when one source fails."""
    # Mock settings
    mock_settings.return_value.x_bearer_token = "test_token"
    mock_settings.return_value.google_trends_enabled = False
    mock_settings.return_value.dry_run = True  # Use dry run to avoid DB

    mock_inst = MagicMock()
    mock_inst.symbol = "AAPL"
    mock_inst.company_name = "Apple Inc."
    mock_inst.model_dump.return_value = {
        "symbol": "AAPL",
        "company_name": "Apple Inc."
    }
    mock_resolve.return_value = mock_inst

    # X works, Reddit fails
    from app.services.types import SocialPost
    test_post = SocialPost(
        source="x",
        platform_id="123",
        author_id="user1",
        author_handle="trader1",
        created_at=datetime.utcnow(),
        text="$AAPL bullish",
        like_count=10
    )
    mock_x.return_value = [test_post]
    mock_reddit.side_effect = Exception("Reddit API error")
    mock_st.return_value = []

    # Should still collect from X despite Reddit failure
    result = aggregate_social("AAPL", "24h")

    # Should have X posts but Reddit failed
    assert result is not None
    assert result["sources"]["x"] == 1
    assert result["sources"]["reddit"] == 0


@patch("app.orchestration.tasks.resolve")
@patch("app.orchestration.tasks.search_x_bundle")
@patch("app.orchestration.tasks.scrape_reddit")
@patch("app.orchestration.tasks.scrape_stocktwits")
@patch("app.orchestration.tasks.collect_google_trends")
@patch("app.orchestration.tasks.get_settings")
def test_aggregate_social_with_google_trends(
    mock_settings, mock_trends, mock_st, mock_reddit, mock_x, mock_resolve
):
    """Test pipeline includes Google Trends data."""
    # Mock settings
    mock_settings.return_value.x_bearer_token = ""  # No X token
    mock_settings.return_value.google_trends_enabled = True
    mock_settings.return_value.dry_run = True

    mock_inst = MagicMock()
    mock_inst.symbol = "AAPL"
    mock_inst.company_name = "Apple Inc."
    mock_inst.model_dump.return_value = {
        "symbol": "AAPL",
        "company_name": "Apple Inc."
    }
    mock_resolve.return_value = mock_inst

    # No posts from any source
    mock_x.return_value = []
    mock_reddit.return_value = []
    mock_st.return_value = []

    # Mock Google Trends data
    mock_trends_data = MagicMock()
    mock_trends_data.model_dump.return_value = {
        "interest_over_time": {"2025-01-01": 75.0},
        "related_queries": ["aapl stock"],
        "interest_by_region": {"California": 100}
    }
    mock_trends.return_value = mock_trends_data

    result = aggregate_social("AAPL", "24h")

    # Should have Google Trends even with no posts
    assert result["search_interest"] is not None
    assert "interest_over_time" in result["search_interest"]


@patch("app.orchestration.tasks.get_settings")
def test_aggregate_social_dry_run(mock_settings):
    """Test dry run mode skips database operations."""
    # Mock settings for dry run
    mock_settings.return_value.x_bearer_token = ""
    mock_settings.return_value.google_trends_enabled = False
    mock_settings.return_value.dry_run = True

    with patch("app.orchestration.tasks.resolve") as mock_resolve:
        mock_inst = MagicMock()
        mock_inst.symbol = "AAPL"
        mock_inst.company_name = "Apple Inc."
        mock_inst.model_dump.return_value = {"symbol": "AAPL", "company_name": "Apple Inc."}
        mock_resolve.return_value = mock_inst

        with patch("app.orchestration.tasks.search_x_bundle", return_value=[]):
            with patch("app.orchestration.tasks.scrape_reddit", return_value=[]):
                with patch("app.orchestration.tasks.scrape_stocktwits", return_value=[]):
                    result = aggregate_social("AAPL", "24h")

    # Dry run should complete without DB
    assert result is not None
    assert "error" in result or "posts_found" in result
