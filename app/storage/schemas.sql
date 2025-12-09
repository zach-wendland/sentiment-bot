CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS source_accounts (
    source TEXT NOT NULL,
    author_id TEXT NOT NULL,
    handle TEXT,
    follower_count INT,
    last_seen TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (source, author_id)
);

CREATE TABLE IF NOT EXISTS social_posts (
    id BIGSERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    platform_id TEXT NOT NULL,
    author_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    text TEXT NOT NULL,
    symbols TEXT[] NOT NULL DEFAULT '{}',
    urls TEXT[] DEFAULT '{}',
    lang TEXT,
    reply_to_id TEXT,
    repost_of_id TEXT,
    like_count INT,
    reply_count INT,
    repost_count INT,
    follower_count INT,
    permalink TEXT,
    UNIQUE (source, platform_id)
);

CREATE TABLE IF NOT EXISTS post_embeddings (
    post_pk BIGINT PRIMARY KEY REFERENCES social_posts(id) ON DELETE CASCADE,
    emb VECTOR(384)
);

CREATE TABLE IF NOT EXISTS sentiment (
    post_pk BIGINT PRIMARY KEY REFERENCES social_posts(id) ON DELETE CASCADE,
    polarity REAL,
    subjectivity REAL,
    sarcasm_prob REAL,
    confidence REAL,
    model TEXT
);

CREATE TABLE IF NOT EXISTS resolver_cache (
    query TEXT PRIMARY KEY,
    symbol TEXT,
    cik TEXT,
    isin TEXT,
    figi TEXT,
    company_name TEXT,
    cached_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_social_posts_symbols ON social_posts USING GIN (symbols);
CREATE INDEX IF NOT EXISTS idx_social_posts_created ON social_posts (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_social_posts_source_created ON social_posts (source, created_at DESC);
