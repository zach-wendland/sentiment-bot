import logging
import logging.config
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.orchestration.tasks import aggregate_social, healthcheck
from app.config import get_settings

# Configure logging
def _configure_logging():
    """Configure structured logging for the application."""
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": "INFO"
            }
        },
        "root": {
            "level": "INFO",
            "handlers": ["console"]
        },
        "loggers": {
            "app": {
                "level": "DEBUG",
                "handlers": ["console"],
                "propagate": False
            }
        }
    }
    logging.config.dictConfig(logging_config)

_configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sentiment Bot API",
    description="Social media sentiment analysis for financial instruments",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    """Log startup."""
    logger.info("Sentiment Bot API starting up")

@app.on_event("shutdown")
async def shutdown():
    """Log shutdown."""
    logger.info("Sentiment Bot API shutting down")

@app.get("/healthz")
def health_check():
    """Health check endpoint."""
    try:
        result = healthcheck()
        return result
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.get("/query")
def query_sentiment(
    symbol: str = Query(..., min_length=1, max_length=10, description="Stock symbol or company name"),
    window: str = Query("24h", regex="^[0-9]+[hd]$", description="Time window (e.g., 24h, 7d)")
):
    """
    Query sentiment for a stock symbol across multiple social media sources.

    Returns aggregated sentiment scores from X, Reddit, StockTwits, and Discord.
    """
    logger.info(f"Received query for symbol={symbol}, window={window}")

    try:
        result = aggregate_social(symbol.upper(), window)
        logger.info(f"Successfully processed query for {symbol}")
        return result
    except ValueError as e:
        logger.warning(f"Invalid query for {symbol}: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")
    except Exception as e:
        logger.error(f"Error processing query for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process query")

@app.get("/")
def root():
    """Root endpoint - service information."""
    return {
        "service": "Sentiment Bot API",
        "version": "1.0.0",
        "description": "Social media sentiment analysis for financial instruments",
        "endpoints": {
            "health": "/healthz",
            "query": "/query?symbol=AAPL&window=24h",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }
