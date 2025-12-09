from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # X/Twitter API
    x_bearer_token: str = ""

    # Scraper rate limits (requests per second)
    reddit_rate_limit: float = 0.5  # 1 req/2sec
    stocktwits_rate_limit: float = 0.33  # 1 req/3sec

    # Google Trends
    google_trends_enabled: bool = True

    # Feature flags
    dry_run: bool = False

    # Database (Supabase compatible)
    database_url: str = "postgresql://user:pass@localhost:5432/sentiment"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
