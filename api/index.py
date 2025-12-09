"""
Vercel Serverless Entry Point

This module wraps the FastAPI application for Vercel serverless deployment.
Vercel expects either 'app' or 'handler' to be exported from this module.
"""

import sys
import os

# Ensure the project root is in the path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the FastAPI application
from app.main import app

# Vercel expects 'app' to be the ASGI application
# This is all that's needed - Vercel handles the rest

# Optional: Add any serverless-specific initialization here
# For example, warming up ML models on cold start

def _warm_up_models():
    """
    Pre-load ML models on cold start to reduce first request latency.
    Models are cached at module level, so this only runs once per container.
    """
    try:
        # Import triggers model loading
        from app.nlp.sentiment import score_text
        from app.nlp.embeddings import compute_embedding

        # Warm up with a simple inference
        score_text("warm up")
        compute_embedding("warm up")
    except Exception:
        # Don't fail startup if model warming fails
        pass


# Uncomment to enable model warming on cold start
# This adds ~5-10s to cold start but makes first request faster
# _warm_up_models()
