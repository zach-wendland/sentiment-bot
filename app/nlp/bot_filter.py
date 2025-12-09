from app.services.types import SocialPost

def is_probable_bot(post: SocialPost) -> bool:
    # Simple heuristics - expand as needed

    # Check for very short posts with just tickers
    if len(post.text) < 20 and len(post.symbols) > 0:
        return True

    # Check for repetitive patterns
    if post.text.count('$') > 5:
        return True

    # Very high post frequency accounts (would need historical data)
    # For now, just basic checks

    return False
