# ERA Decision & Arbitration Layer
# Production-ready Docker image
# 
# Build: docker build -t era-dal:latest .
# Run:   docker run -e OPENROUTER_API_KEY=sk-... era-dal --pool science --problem "Your question"

FROM python:3.11-slim as builder

# Set build-time variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ============================================
# Production stage
# ============================================
FROM python:3.11-slim as production

# Labels
LABEL maintainer="Eugene Kundrotas" \
      version="1.1.0" \
      description="ERA Decision & Arbitration Layer - Reliable LLM Ensemble Decision Making"

# Runtime environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN groupadd -r era && useradd -r -g era era

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY --chown=era:era . .

# Create output directory
RUN mkdir -p /app/out && chown -R era:era /app/out

# Switch to non-root user
USER era

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; from src.config import config; sys.exit(0 if config.openrouter_api_key else 1)"

# Default command
ENTRYPOINT ["python", "app.py"]
CMD ["--help"]

# ============================================
# Dashboard variant
# ============================================
FROM production as dashboard

# Expose dashboard port
EXPOSE 8080

# Override entrypoint for dashboard mode
ENTRYPOINT ["python", "-m", "src.api"]
CMD ["8080"]

# ============================================
# Development stage
# ============================================
FROM production as development

USER root

# Install development dependencies
RUN pip install pytest pytest-cov black isort mypy

# Switch back to era user
USER era

# Development command runs tests by default
CMD ["python", "-m", "pytest", "tests/", "-v"]
