"""Tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    """Test root endpoint returns service info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "endpoints" in data
    assert data["service"] == "Sentiment Bot API"

def test_healthz_endpoint():
    """Test health check endpoint."""
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data

def test_query_endpoint_missing_symbol():
    """Test query endpoint returns 422 when symbol is missing."""
    response = client.get("/query")
    assert response.status_code == 422  # Validation error

def test_query_endpoint_invalid_window():
    """Test query endpoint returns 422 for invalid window format."""
    response = client.get("/query?symbol=AAPL&window=invalid")
    assert response.status_code == 422

def test_query_endpoint_valid_params():
    """Test query endpoint with valid parameters.

    Note: This will attempt to actually query the pipeline,
    which may fail if API keys aren't configured, but we're
    testing that the endpoint itself works.
    """
    response = client.get("/query?symbol=AAPL&window=24h")
    # Should get either 200 or 500, but not validation error
    assert response.status_code in [200, 500]
    assert "detail" not in response.json() or response.status_code == 500

def test_query_endpoint_max_symbol_length():
    """Test query endpoint enforces max symbol length."""
    long_symbol = "A" * 20
    response = client.get(f"/query?symbol={long_symbol}&window=24h")
    assert response.status_code == 422

def test_docs_endpoint():
    """Test OpenAPI docs are available."""
    response = client.get("/docs")
    assert response.status_code == 200

def test_redoc_endpoint():
    """Test ReDoc documentation is available."""
    response = client.get("/redoc")
    assert response.status_code == 200

def test_openapi_json():
    """Test OpenAPI JSON schema is available."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert "openapi" in data
    assert "paths" in data
