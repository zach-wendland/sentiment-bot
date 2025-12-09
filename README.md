# Sentiment Bot v2 - Financial Sentiment Analysis API

Aggregates and analyzes social media sentiment about financial instruments across multiple platforms: X (Twitter), Reddit, and StockTwits. Enriched with Google Trends data for market interest context.

**Deployment**: Railway (API) + Supabase (PostgreSQL + pgvector)

## Features

- Real-time sentiment analysis using DistilBERT
- Multi-source data collection: X API v2, Reddit scraper, StockTwits scraper
- Google Trends integration for market interest signals
- Semantic embeddings with sentence-transformers for similarity search
- PostgreSQL with pgvector for vector search
- Container-based deployment (supports large ML models)

---

## Quick Start

### Option 1: Deploy to Railway + Supabase (Recommended)

**1. Set up Supabase:**
```bash
# Create a Supabase project at https://supabase.com
# Enable pgvector extension: SQL Editor -> CREATE EXTENSION IF NOT EXISTS vector;
# Run the migration: infra/supabase/migrations/001_initial_schema.sql
```

**2. Deploy to Railway:**
```bash
# Connect your GitHub repo to Railway at https://railway.app
# Set environment variables:
#   X_BEARER_TOKEN=your_twitter_bearer_token
#   DATABASE_URL=your_supabase_connection_string
#   PORT=8000
```

### Option 2: Local Development

**1. Start PostgreSQL with pgvector:**
```bash
cd infra
docker-compose up -d
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Configure environment:**
```bash
cp .env.example .env
# Edit .env with your API keys
```

**4. Run the API:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Access the API:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/healthz
```

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2025-01-15T10:30:00.123456",
  "version": "2.0.0",
  "components": {
    "x_api": true,
    "google_trends": true,
    "dry_run": false
  }
}
```

### Query Sentiment
```bash
curl "http://localhost:8000/query?symbol=AAPL&window=24h"
```

**Parameters:**
- `symbol` (required): Stock symbol or company name (e.g., `AAPL`, `TSLA`, `Apple`)
- `window` (optional, default: `24h`): Time window (`24h`, `7d`, `1w`)

**Response:**
```json
{
  "symbol": "AAPL",
  "posts_found": 42,
  "posts_processed": 38,
  "sources": {
    "x": 15,
    "reddit": 12,
    "stocktwits": 11
  },
  "avg_polarity": 0.65,
  "avg_subjectivity": 0.52,
  "avg_confidence": 0.78,
  "search_interest": {
    "interest_over_time": {"2025-01-14": 75, "2025-01-15": 82},
    "related_queries": ["aapl stock", "apple earnings"],
    "interest_by_region": {"California": 100, "New York": 95}
  },
  "resolved_instrument": {
    "symbol": "AAPL",
    "company_name": "Apple Inc."
  }
}
```

---

## Architecture

```
sentiment-bot/
├── app/
│   ├── main.py                   # FastAPI application
│   ├── config.py                 # Settings management
│   ├── services/
│   │   ├── types.py              # Data models
│   │   ├── resolver.py           # Symbol resolution
│   │   └── x_client.py           # X/Twitter API v2 client
│   ├── scrapers/
│   │   ├── base.py               # Base scraper framework
│   │   ├── reddit_scraper.py     # Reddit JSON scraper
│   │   ├── stocktwits_scraper.py # StockTwits public API
│   │   └── google_trends.py      # Google Trends integration
│   ├── nlp/
│   │   ├── sentiment.py          # DistilBERT sentiment scoring
│   │   ├── embeddings.py         # Sentence-transformers
│   │   ├── clean.py              # Text normalization
│   │   └── bot_filter.py         # Bot detection
│   ├── orchestration/
│   │   └── tasks.py              # Main pipeline
│   └── storage/
│       ├── db.py                 # PostgreSQL operations
│       └── schemas.sql           # DB schema
├── infra/
│   ├── docker-compose.yaml       # Local PostgreSQL
│   └── supabase/
│       └── migrations/           # Supabase schema migrations
├── tests/                        # Unit & integration tests
├── Procfile                      # Railway deployment config
└── requirements.txt
```

## Data Flow

```
Query (symbol, window)
  ↓
Resolve Symbol (cached in PostgreSQL)
  ↓
Collect Posts (parallel execution):
  ├→ X/Twitter API v2 (official, bearer token)
  ├→ Reddit Scraper (JSON endpoints, no auth)
  └→ StockTwits Scraper (public API)
  ↓
Clean & Filter:
  ├→ Normalize text (URLs, whitespace)
  ├→ Extract symbols (cashtags, company names)
  ├→ Filter bots (heuristics)
  └→ Skip posts without relevant symbols
  ↓
Process & Persist:
  ├→ Score sentiment (DistilBERT)
  ├→ Generate embeddings (sentence-transformers, 384-dim)
  └→ Store in PostgreSQL + pgvector
  ↓
Enrich with Google Trends:
  ├→ Interest over time (7-day)
  ├→ Related queries
  └→ Interest by region
  ↓
Aggregate Results → JSON Response
```

## Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Deployment** | Railway | Container-based API hosting |
| **Database** | Supabase (PostgreSQL + pgvector) | Data storage + vector search |
| **Web Framework** | FastAPI 0.109.0 | REST API, auto-docs |
| **Sentiment** | DistilBERT (SST-2) | Fast sentiment classification |
| **Embeddings** | sentence-transformers | Semantic similarity (384-dim) |
| **HTTP Client** | httpx | Async HTTP for APIs |
| **Scraping** | BeautifulSoup + httpx | Web scraping framework |
| **Trends** | pytrends | Google Trends data |
| **Testing** | pytest 7.4.3 | Unit & integration tests |

## ML Models

- **DistilBERT** (distilbert-base-uncased-finetuned-sst-2-english):
  - Optimized for speed (~5s model load)
  - Classes: negative, positive (neutral via confidence threshold)
  - Output: Polarity (-1 to +1), Confidence

- **Sentence-Transformers** (all-MiniLM-L6-v2):
  - Fast semantic embeddings
  - Dimensionality: 384
  - Use case: Vector similarity search

## Configuration

Environment variables (`.env`):

| Variable | Required | Description |
|----------|----------|-------------|
| `X_BEARER_TOKEN` | For X data | Twitter API v2 bearer token |
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `PORT` | Yes (Railway) | Port to listen on (Railway sets this) |
| `REDDIT_RATE_LIMIT` | No | Requests per second (default: 0.5) |
| `STOCKTWITS_RATE_LIMIT` | No | Requests per second (default: 0.33) |
| `GOOGLE_TRENDS_ENABLED` | No | Enable trends enrichment (default: true) |
| `DRY_RUN` | No | Skip DB writes (default: false) |

## Testing

Run all tests:
```bash
pytest tests/
```

Run specific test file:
```bash
pytest tests/test_scrapers.py -v
```

Run with coverage:
```bash
pytest tests/ --cov=app
```

Run E2E validation:
```bash
python validate_e2e.py
```

## Error Handling

The API gracefully handles failures:

- **Single source fails**: Other sources continue, partial results returned
- **Model loading fails**: Falls back to heuristic sentiment scoring
- **Rate limit (429)**: Exponential backoff with retry
- **Invalid symbol**: Returns 400 with clear error message
- **Network error**: Logs warning, returns empty from that source

## Data Sources

| Source | Method | Rate Limit | Auth |
|--------|--------|------------|------|
| X/Twitter | Official API v2 | 450 req/15min | Bearer token |
| Reddit | JSON endpoints | 1 req/2sec | None |
| StockTwits | Public API | 1 req/3sec | None |
| Google Trends | pytrends | ~10 req/min | None |

## Scraper Maintenance

Scrapers may need updates when sites change their structure:

- **Reddit**: Uses `old.reddit.com/.json` endpoints (stable)
- **StockTwits**: Uses public `api.stocktwits.com` (stable)
- **Google Trends**: Uses pytrends library (maintained)

If scraping fails, check logs and update selectors/endpoints as needed.

## Deployment

### Railway

1. Connect GitHub repo to Railway at https://railway.app
2. Set environment variables in Railway dashboard:
   - `X_BEARER_TOKEN`
   - `DATABASE_URL` (Supabase connection string)
   - `PORT` (Railway sets automatically)
3. Deploy - Railway auto-detects Python and uses Procfile

### Supabase

1. Create project at supabase.com
2. Enable pgvector: `CREATE EXTENSION IF NOT EXISTS vector;`
3. Run migration from `infra/supabase/migrations/`
4. Copy connection string (use pooler URL for production)

## Troubleshooting

**"No posts found" error:**
- Verify X_BEARER_TOKEN is set (for X data)
- Try longer window (`7d` instead of `24h`)
- Check if symbol is valid

**"Failed to load DistilBERT" warning:**
- Model falls back to heuristics, results still valid
- Ensure ~300MB available for model download

**Database connection error:**
- Verify DATABASE_URL format
- For Supabase: use pooler URL (port 6543)
- For local: check Docker is running

**Rate limiting:**
- API automatically retries with backoff
- Check logs for 429 responses
- Adjust rate limits in .env if needed

## License

MIT License - See LICENSE file for details.
