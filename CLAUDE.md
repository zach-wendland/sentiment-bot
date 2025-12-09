# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
# Local development - Start PostgreSQL with pgvector
cd infra && docker-compose up -d

# Install dependencies
pip install -r requirements.txt

# Run the API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_scrapers.py -v

# Run tests with coverage
pytest tests/ --cov=app

# Run E2E validation
python validate_e2e.py
```

## Architecture Overview

This is a FastAPI-based sentiment analysis service for financial instruments, deployed on Railway with Supabase (PostgreSQL + pgvector).

**Data Flow:**
```
Query → Symbol Resolution → Multi-Source Collection (parallel) → NLP Pipeline → PostgreSQL Storage → Google Trends Enrichment → Aggregation → Response
```

### Key Components

- **`app/main.py`**: FastAPI app with `/query`, `/healthz`, and `/` endpoints
- **`app/orchestration/tasks.py`**: Main pipeline orchestrator (`aggregate_social`) - coordinates symbol resolution, parallel post collection, cleaning, sentiment scoring, embedding generation, Google Trends enrichment, and result aggregation
- **`app/services/resolver.py`**: Maps company names/tickers to standardized symbols (cached in PostgreSQL)
- **`app/services/x_client.py`**: X/Twitter API v2 client (bearer token auth)
- **`app/scrapers/base.py`**: Base scraper framework with rate limiting, retry, user-agent rotation
- **`app/scrapers/reddit_scraper.py`**: Reddit JSON endpoint scraper (no auth)
- **`app/scrapers/stocktwits_scraper.py`**: StockTwits public API scraper
- **`app/scrapers/google_trends.py`**: Google Trends integration via pytrends
- **`app/nlp/sentiment.py`**: DistilBERT sentiment scoring with heuristic fallback
- **`app/nlp/embeddings.py`**: Sentence-transformer embeddings (384-dim, normalized)
- **`app/nlp/clean.py`**: Text normalization and symbol extraction from cashtags
- **`app/nlp/bot_filter.py`**: Heuristic bot detection
- **`app/storage/db.py`**: PostgreSQL + pgvector operations (Supabase-compatible)
- **`app/config.py`**: Pydantic settings from environment variables

### Data Models (app/services/types.py)

- `SocialPost`: Normalized post from any source (reddit, x, stocktwits)
- `SentimentScore`: polarity (-1 to +1), subjectivity, sarcasm_prob, confidence, model
- `ResolvedInstrument`: symbol, cik, isin, figi, company_name
- `TrendsData`: Google Trends interest_over_time, related_queries, interest_by_region

### ML Models

- **DistilBERT** (`distilbert-base-uncased-finetuned-sst-2-english`): Sentiment classification (negative/positive, ~5s cold start)
- **Sentence-Transformers** (`all-MiniLM-L6-v2`): 384-dim text embeddings for vector search

### Infrastructure

- **Railway**: Container-based API hosting (Procfile)
- **Supabase**: PostgreSQL 16 with pgvector extension
- **Local**: Docker Compose for PostgreSQL (in `infra/docker-compose.yaml`)

## Configuration

All settings via environment variables (see `.env.example`):
- `X_BEARER_TOKEN`: X/Twitter API v2 bearer token
- `DATABASE_URL`: PostgreSQL connection string (Supabase pooler URL for production)
- `PORT`: Port to listen on (Railway sets this automatically)
- `REDDIT_RATE_LIMIT`: Requests per second (default: 0.5)
- `STOCKTWITS_RATE_LIMIT`: Requests per second (default: 0.33)
- `GOOGLE_TRENDS_ENABLED`: Enable trends enrichment (default: true)
- `DRY_RUN`: Skip database writes (default: false)

## Code Patterns

- Parallel data collection using ThreadPoolExecutor
- Pydantic models for all data structures
- Global model caching in NLP modules (`_model_cache` dict pattern)
- Token bucket rate limiting in scrapers
- Exponential backoff retry with configurable max retries
- User-agent rotation for web scraping
- Fallback strategies: DistilBERT → heuristics, sentence-transformers → hash-based embeddings
- Graceful degradation: individual source failures don't block other sources
- Connection pooling for Supabase environments

## Testing

Test files:
- `tests/test_scrapers.py`: Scraper framework and implementations
- `tests/test_api_clients.py`: X API and mocked external APIs
- `tests/test_pipeline.py`: Pipeline orchestration
- `tests/test_nlp.py`: NLP modules (sentiment, embeddings, cleaning)
- `tests/test_api_endpoints.py`: FastAPI endpoint tests
- `validate_e2e.py`: End-to-end validation script
- supabase project name: stonk-data, database password: 12R3deng1ne$60
