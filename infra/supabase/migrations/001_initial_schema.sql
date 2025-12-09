-- Sentiment Bot Initial Schema for Supabase
-- Run this in the Supabase SQL Editor after enabling pgvector extension

-- Enable pgvector extension (must be done first)
CREATE EXTENSION IF NOT EXISTS vector;

-- Source accounts table - tracks social media accounts
CREATE TABLE IF NOT EXISTS source_accounts (
    source TEXT NOT NULL,
    author_id TEXT NOT NULL,
    handle TEXT,
    follower_count INT,
    last_seen TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (source, author_id)
);

-- Social posts table - stores all collected posts
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

-- Post embeddings table - stores vector embeddings for similarity search
-- Uses 384 dimensions for sentence-transformers/all-MiniLM-L6-v2 model
CREATE TABLE IF NOT EXISTS post_embeddings (
    post_pk BIGINT PRIMARY KEY REFERENCES social_posts(id) ON DELETE CASCADE,
    emb VECTOR(384)
);

-- Sentiment scores table - stores sentiment analysis results
CREATE TABLE IF NOT EXISTS sentiment (
    post_pk BIGINT PRIMARY KEY REFERENCES social_posts(id) ON DELETE CASCADE,
    polarity REAL,
    subjectivity REAL,
    sarcasm_prob REAL,
    confidence REAL,
    model TEXT
);

-- Symbol resolution cache - caches symbol lookups
CREATE TABLE IF NOT EXISTS resolver_cache (
    query TEXT PRIMARY KEY,
    symbol TEXT,
    cik TEXT,
    isin TEXT,
    figi TEXT,
    company_name TEXT,
    cached_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_social_posts_symbols ON social_posts USING GIN (symbols);
CREATE INDEX IF NOT EXISTS idx_social_posts_created ON social_posts (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_social_posts_source_created ON social_posts (source, created_at DESC);

-- Index for vector similarity search (using IVFFlat for better performance)
-- Note: Run this after inserting some data, as IVFFlat requires training data
-- CREATE INDEX IF NOT EXISTS idx_post_embeddings_emb ON post_embeddings USING ivfflat (emb vector_cosine_ops) WITH (lists = 100);

-- Grant permissions (Supabase handles this, but included for reference)
-- GRANT ALL ON ALL TABLES IN SCHEMA public TO postgres;
-- GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO postgres;
