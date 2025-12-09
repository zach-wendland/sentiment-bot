"""
Web Scrapers Package

Contains scraper implementations for various social media platforms.
All scrapers inherit from BaseScraper and implement rate limiting,
retry logic, and user-agent rotation.
"""

from app.scrapers.base import BaseScraper, RateLimiter

__all__ = ["BaseScraper", "RateLimiter"]
