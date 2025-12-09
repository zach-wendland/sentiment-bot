import httpx
from typing import Optional
from app.services.types import ResolvedInstrument
from app.storage.db import DB

# Simple symbol map for common tickers
SYMBOL_MAP = {
    "APPLE": "AAPL",
    "TESLA": "TSLA",
    "MICROSOFT": "MSFT",
    "AMAZON": "AMZN",
    "GOOGLE": "GOOGL",
    "META": "META",
    "NVIDIA": "NVDA",
}

def resolve(query: str) -> ResolvedInstrument:
    # Check cache first
    db = DB()
    cached = db.get_cached_resolution(query.upper())
    if cached:
        return ResolvedInstrument(**cached)

    # Normalize query
    query_upper = query.upper().strip('$')

    # Check if it's already a ticker
    if len(query_upper) <= 5 and query_upper.isalpha():
        result = ResolvedInstrument(
            symbol=query_upper,
            company_name=query_upper
        )
    # Check if it's a company name
    elif query_upper in SYMBOL_MAP:
        symbol = SYMBOL_MAP[query_upper]
        result = ResolvedInstrument(
            symbol=symbol,
            company_name=query
        )
    else:
        # Default fallback
        result = ResolvedInstrument(
            symbol=query_upper,
            company_name=query
        )

    # Cache the result
    db.cache_resolution(
        query=query.upper(),
        symbol=result.symbol,
        cik=result.cik,
        isin=result.isin,
        figi=result.figi,
        company_name=result.company_name
    )

    return result
