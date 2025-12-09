import re
from typing import List

def normalize_post(t: str) -> str:
    # Remove URLs
    t = re.sub(r'https?://\S+', '', t)
    # Remove excessive whitespace
    t = re.sub(r'\s+', ' ', t)
    t = t.strip()
    return t

def extract_symbols(t: str, inst: dict) -> List[str]:
    # Extract cashtags
    tickers = set(re.findall(r'\$([A-Z]{1,5})(?![A-Z])', t))

    # Add instrument symbol if mentioned (safely handle missing keys)
    symbol = inst.get("symbol", "")
    company_name = inst.get("company_name", "")

    if symbol and (symbol in t.upper() or symbol in tickers):
        tickers.add(symbol)

    # Add if company name mentioned
    if company_name and company_name.upper() in t.upper():
        tickers.add(symbol)

    return list(tickers)
