# ==============================================================================
# Production Dockerfile for Air Export Pricing Recommendation Engine
# Production deployment for AWS (S3, SSM) and standalone container runtimes
# ==============================================================================
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src:/app
ENV ALLOW_SKLEARN_MISMATCH=1

WORKDIR /app

# Install minimal OS dependencies for numerical libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt boto3 gunicorn

# Copy application modules, models, and data
COPY src /app/src
COPY models /app/models
COPY data /app/data
COPY scripts /app/scripts

# Default ports and healthcheck
EXPOSE 5050

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:5050/ || exit 1

# Production Gunicorn WSGI server entrypoint
CMD ["gunicorn", "--bind", "0.0.0.0:5050", "--workers", "2", "--threads", "2", "--timeout", "60", "--access-logfile", "-", "src.app:app"]
