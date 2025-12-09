# GEMINI.md

This document provides a comprehensive overview of the Sentiment Bot project, intended for developers and future AI interactions.

## Project Overview

The Sentiment Bot is a Python-based API that aggregates and analyzes social media sentiment about financial instruments. It collects data from Reddit, performs sentiment analysis using a pre-trained FinBERT model, and stores the results in a PostgreSQL database with pgvector for efficient similarity searches. The project uses FastAPI to expose a RESTful API for querying sentiment data.

### Key Technologies

*   **Backend:** Python, FastAPI
*   **Data Storage:** PostgreSQL with pgvector, Redis for caching
*   **NLP:** `transformers` (FinBERT), `sentence-transformers`
*   **Social Media Clients:** `praw` (Reddit)
*   **Infrastructure:** Docker, Docker Compose

### Architecture

The application is structured as a monolithic service with a clear separation of concerns:

*   `app/main.py`: The FastAPI application entry point, defining the API endpoints.
*   `app/orchestration/tasks.py`: The core of the application, orchestrating the data collection, processing, and storage pipeline.
*   `app/services/`: Contains the client for interacting with the Reddit API and the symbol resolver.
*   `app/nlp/`: Handles all NLP-related tasks, including sentiment analysis, text cleaning, and embedding generation.
*   `app/storage/`: Manages database interactions.
*   `app/config.py`: Manages application configuration using Pydantic.
*   `infra/`: Contains the Docker Compose configuration for the PostgreSQL and Redis services.

## Building and Running

### Prerequisites

*   Python 3.8+
*   Docker & Docker Compose
*   API keys for the various social media platforms (optional, but recommended).

### Setup and Execution

1.  **Start the backend services:**
    ```bash
    cd infra
    docker-compose up -d
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure the environment:**
    *   Copy `.env.example` to `.env`.
    *   Edit `.env` to add your API keys and any other necessary configuration.

4.  **Run the application:**
    ```bash
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```

### Testing

The project uses `pytest` for testing. To run the tests, execute the following command:

```bash
pytest tests/
```

## Development Conventions

*   **Configuration:** All configuration is managed through environment variables, facilitated by the `pydantic-settings` library. No secrets are hard-coded.
*   **Styling:** The code follows the standard PEP 8 style guidelines.
*   **Typing:** The code uses type hints for clarity and maintainability.
*   **Testing:** The `tests/` directory contains integration and unit tests for the various components of the application.
*   **Asynchronous Operations:** The application utilizes asynchronous programming where appropriate.
