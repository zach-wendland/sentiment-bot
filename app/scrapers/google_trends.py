"""
Google Trends Integration

Fetches search interest data from Google Trends using the pytrends library.
Provides additional context on public interest in a stock symbol.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

# Try to import pytrends - graceful fallback if not available
try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False
    logger.warning("pytrends not installed. Google Trends integration disabled.")


class TrendsData:
    """Data class for Google Trends results."""

    def __init__(
        self,
        interest_over_time: Optional[Dict[str, float]] = None,
        related_queries: Optional[List[str]] = None,
        interest_by_region: Optional[Dict[str, float]] = None,
        fetched_at: Optional[datetime] = None
    ):
        self.interest_over_time = interest_over_time or {}
        self.related_queries = related_queries or []
        self.interest_by_region = interest_by_region or {}
        self.fetched_at = fetched_at or datetime.utcnow()

    def model_dump(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "interest_over_time": self.interest_over_time,
            "related_queries": self.related_queries,
            "interest_by_region": self.interest_by_region,
            "fetched_at": self.fetched_at.isoformat() if self.fetched_at else None
        }


def collect_google_trends(symbol: str, company_name: Optional[str] = None) -> Optional[TrendsData]:
    """
    Collect Google Trends data for a stock symbol.

    Args:
        symbol: Stock symbol (e.g., "AAPL")
        company_name: Company name (e.g., "Apple")

    Returns:
        TrendsData object or None if unavailable
    """
    if not PYTRENDS_AVAILABLE:
        logger.warning("pytrends not available, skipping Google Trends")
        return None

    if not symbol:
        return None

    try:
        # Build search terms
        search_terms = [symbol]
        if company_name and company_name.upper() != symbol:
            # Add company name as second term
            search_terms.append(company_name)

        # Limit to 2 terms for comparison
        search_terms = search_terms[:2]

        logger.info(f"Fetching Google Trends for: {search_terms}")

        # Initialize pytrends
        pytrends = TrendReq(
            hl='en-US',
            tz=360,
            timeout=(10, 25),  # Connect timeout, read timeout
            retries=2,
            backoff_factor=0.5
        )

        # Build payload
        pytrends.build_payload(
            kw_list=search_terms,
            timeframe='now 7-d',  # Last 7 days
            geo='US',
            gprop=''  # Web search
        )

        # Get interest over time
        interest_over_time = _get_interest_over_time(pytrends, symbol)

        # Get related queries
        related_queries = _get_related_queries(pytrends, symbol)

        # Get interest by region
        interest_by_region = _get_interest_by_region(pytrends, symbol)

        return TrendsData(
            interest_over_time=interest_over_time,
            related_queries=related_queries,
            interest_by_region=interest_by_region,
            fetched_at=datetime.utcnow()
        )

    except Exception as e:
        logger.error(f"Error fetching Google Trends for {symbol}: {e}")
        return None


def _get_interest_over_time(pytrends: 'TrendReq', symbol: str) -> Dict[str, float]:
    """
    Get interest over time data.

    Args:
        pytrends: Initialized TrendReq object
        symbol: Stock symbol to get data for

    Returns:
        Dict mapping date strings to interest values (0-100)
    """
    try:
        df = pytrends.interest_over_time()
        if df is None or df.empty:
            return {}

        # Get data for the primary symbol
        if symbol in df.columns:
            result = {}
            for idx, row in df.iterrows():
                date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, 'strftime') else str(idx)
                result[date_str] = float(row[symbol])
            return result
        else:
            # Return first column if symbol not found
            first_col = df.columns[0] if len(df.columns) > 0 else None
            if first_col and first_col != 'isPartial':
                result = {}
                for idx, row in df.iterrows():
                    date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, 'strftime') else str(idx)
                    result[date_str] = float(row[first_col])
                return result

        return {}

    except Exception as e:
        logger.warning(f"Error getting interest over time: {e}")
        return {}


def _get_related_queries(pytrends: 'TrendReq', symbol: str) -> List[str]:
    """
    Get related search queries.

    Args:
        pytrends: Initialized TrendReq object
        symbol: Stock symbol to get data for

    Returns:
        List of related query strings
    """
    try:
        related = pytrends.related_queries()
        if not related:
            return []

        queries = []

        # Get queries for the symbol
        symbol_data = related.get(symbol, {})

        # Top queries
        top_df = symbol_data.get('top')
        if top_df is not None and not top_df.empty:
            top_queries = top_df['query'].tolist()[:5]
            queries.extend(top_queries)

        # Rising queries
        rising_df = symbol_data.get('rising')
        if rising_df is not None and not rising_df.empty:
            rising_queries = rising_df['query'].tolist()[:5]
            queries.extend(rising_queries)

        # Deduplicate while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)

        return unique_queries[:10]

    except Exception as e:
        logger.warning(f"Error getting related queries: {e}")
        return []


def _get_interest_by_region(pytrends: 'TrendReq', symbol: str) -> Dict[str, float]:
    """
    Get interest by region (US states).

    Args:
        pytrends: Initialized TrendReq object
        symbol: Stock symbol to get data for

    Returns:
        Dict mapping region names to interest values (0-100)
    """
    try:
        df = pytrends.interest_by_region(resolution='REGION', inc_low_vol=False)
        if df is None or df.empty:
            return {}

        result = {}
        col = symbol if symbol in df.columns else (df.columns[0] if len(df.columns) > 0 else None)

        if col:
            # Get top 10 regions by interest
            sorted_df = df.sort_values(by=col, ascending=False).head(10)
            for region, row in sorted_df.iterrows():
                result[str(region)] = float(row[col])

        return result

    except Exception as e:
        logger.warning(f"Error getting interest by region: {e}")
        return {}
