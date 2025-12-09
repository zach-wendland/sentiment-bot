from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict
from datetime import datetime


class SocialPost(BaseModel):
    """Represents a social media post from any platform."""

    source: Literal["reddit", "x", "stocktwits"]
    platform_id: str
    author_id: str
    author_handle: Optional[str] = None
    created_at: datetime
    text: str
    symbols: List[str] = []
    urls: List[str] = []
    lang: Optional[str] = None
    reply_to_id: Optional[str] = None
    repost_of_id: Optional[str] = None
    like_count: Optional[int] = None
    reply_count: Optional[int] = None
    repost_count: Optional[int] = None
    follower_count: Optional[int] = None
    permalink: Optional[str] = None


class SentimentScore(BaseModel):
    """Sentiment analysis result for a piece of text."""

    polarity: float  # -1.0 (negative) to +1.0 (positive)
    subjectivity: float  # 0.0 (objective) to 1.0 (subjective)
    sarcasm_prob: float  # 0.0 to 1.0 probability of sarcasm
    confidence: float  # 0.0 to 1.0 model confidence
    model: str = "distilbert"  # Model used for scoring


class ResolvedInstrument(BaseModel):
    """Resolved financial instrument information."""

    symbol: str
    cik: Optional[str] = None
    isin: Optional[str] = None
    figi: Optional[str] = None
    company_name: str


class TrendsData(BaseModel):
    """Google Trends data for a symbol."""

    interest_over_time: Optional[Dict[str, float]] = None  # date -> interest (0-100)
    related_queries: Optional[List[str]] = None  # Related search terms
    interest_by_region: Optional[Dict[str, float]] = None  # region -> interest
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
