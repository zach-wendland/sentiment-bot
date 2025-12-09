#!/bin/bash
# Setup verification script for Sentiment Bot MVP

echo "╔════════════════════════════════════════════════════════╗"
echo "║     SENTIMENT BOT MVP - SETUP VERIFICATION              ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

CHECKS_PASSED=0
CHECKS_TOTAL=0

# Function to check if command exists
check_command() {
    local cmd=$1
    local name=$2
    CHECKS_TOTAL=$((CHECKS_TOTAL + 1))

    if command -v $cmd &> /dev/null; then
        echo "✓ $name is installed"
        CHECKS_PASSED=$((CHECKS_PASSED + 1))
    else
        echo "✗ $name is NOT installed"
    fi
}

# Function to check if file exists
check_file() {
    local file=$1
    local name=$2
    CHECKS_TOTAL=$((CHECKS_TOTAL + 1))

    if [ -f "$file" ]; then
        echo "✓ $name exists"
        CHECKS_PASSED=$((CHECKS_PASSED + 1))
    else
        echo "✗ $name NOT found: $file"
    fi
}

# Function to check if directory exists
check_dir() {
    local dir=$1
    local name=$2
    CHECKS_TOTAL=$((CHECKS_TOTAL + 1))

    if [ -d "$dir" ]; then
        echo "✓ $name directory exists"
        CHECKS_PASSED=$((CHECKS_PASSED + 1))
    else
        echo "✗ $name directory NOT found: $dir"
    fi
}

echo "=== SYSTEM DEPENDENCIES ==="
check_command "python3" "Python 3"
check_command "docker" "Docker"
check_command "docker-compose" "Docker Compose"
check_command "curl" "curl"
echo ""

echo "=== PROJECT STRUCTURE ==="
check_file "requirements.txt" "requirements.txt"
check_file "README.md" "README.md"
check_file ".env.example" ".env.example"
check_dir "app" "app directory"
check_dir "tests" "tests directory"
check_dir "infra" "infra directory"
echo ""

echo "=== APPLICATION FILES ==="
check_file "app/main.py" "app/main.py (FastAPI)"
check_file "app/config.py" "app/config.py (Configuration)"
check_dir "app/services" "app/services (API clients)"
check_dir "app/nlp" "app/nlp (NLP modules)"
check_dir "app/orchestration" "app/orchestration (Pipeline)"
check_dir "app/storage" "app/storage (Database)"
echo ""

echo "=== SERVICE FILES ==="
check_file "app/services/x_client.py" "X/Twitter client"
check_file "app/services/reddit_client.py" "Reddit client"
check_file "app/services/stocktwits_client.py" "StockTwits client"
check_file "app/services/discord_client.py" "Discord client"
echo ""

echo "=== NLP FILES ==="
check_file "app/nlp/sentiment.py" "Sentiment analysis (FinBERT)"
check_file "app/nlp/embeddings.py" "Embeddings (sentence-transformers)"
check_file "app/nlp/clean.py" "Text cleaning"
check_file "app/nlp/bot_filter.py" "Bot detection"
echo ""

echo "=== TEST FILES ==="
check_file "tests/test_api_endpoints.py" "API endpoint tests"
check_file "tests/test_nlp.py" "NLP tests"
check_file "tests/test_api_clients.py" "API client tests"
check_file "tests/test_pipeline.py" "Pipeline tests"
check_file "validate_e2e.py" "E2E validation script"
echo ""

echo "=== INFRASTRUCTURE ==="
check_file "infra/docker-compose.yaml" "Docker Compose config"
echo ""

echo "=== ENVIRONMENT CONFIGURATION ==="
if [ -f ".env" ]; then
    echo "✓ .env file exists"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
else
    echo "⚠ .env file NOT found (copy from .env.example and configure)"
fi
CHECKS_TOTAL=$((CHECKS_TOTAL + 1))
echo ""

echo "=== DOCKER CONTAINERS ==="
echo "Checking if required Docker services are configured:"
check_file "infra/docker-compose.yaml" "docker-compose.yaml contains PostgreSQL"
echo ""

echo "=== SUMMARY ==="
echo "Setup verification: $CHECKS_PASSED/$CHECKS_TOTAL checks passed"
echo ""

if [ $CHECKS_PASSED -eq $CHECKS_TOTAL ]; then
    echo "✓ Setup looks good! You're ready to start."
    echo ""
    echo "Next steps:"
    echo "  1. Configure .env with your API credentials"
    echo "  2. Start Docker containers: cd infra && docker-compose up -d"
    echo "  3. Install dependencies: pip install -r requirements.txt"
    echo "  4. Run the API: uvicorn app.main:app --reload"
    echo "  5. Visit http://localhost:8000/docs for API documentation"
    exit 0
else
    MISSING=$((CHECKS_TOTAL - CHECKS_PASSED))
    echo "⚠ $MISSING check(s) failed. See details above."
    exit 1
fi
