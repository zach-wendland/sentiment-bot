import logging
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Global model cache
_embedding_model = None

def _get_embedding_model():
    """Load sentence-transformers model (cached after first load)."""
    global _embedding_model

    if _embedding_model is None:
        try:
            # all-MiniLM-L6-v2: 384-dim, fast, good for semantic similarity
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            _embedding_model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            _embedding_model = "error"

    return _embedding_model if _embedding_model != "error" else None

def compute_embedding(text: str) -> np.ndarray:
    """
    Compute semantic embedding for text using sentence-transformers.

    Falls back to simple hash-based embedding if model loading fails.

    Args:
        text: Text to embed

    Returns:
        384-dim normalized embedding vector
    """
    model = _get_embedding_model()

    if model is not None:
        try:
            # Truncate very long text
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]

            # Encode and normalize
            embedding = model.encode(text, normalize_embeddings=True)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.warning(f"Embedding inference failed: {e}. Using fallback hash-based embedding.")

    # Fallback: deterministic hash-based embedding (same output for same input)
    return _hash_based_embedding(text, dim=384)

def _hash_based_embedding(text: str, dim: int = 384) -> np.ndarray:
    """
    Generate deterministic embedding from text hash.

    Used as fallback when neural embedding model unavailable.

    Args:
        text: Text to embed
        dim: Embedding dimensionality

    Returns:
        Normalized random vector seeded by text hash
    """
    np.random.seed(hash(text) % (2**32))
    emb = np.random.randn(dim).astype(np.float32)
    # Normalize to unit length
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb
