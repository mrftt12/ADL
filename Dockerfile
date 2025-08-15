# Multi-stage build for AutoML Framework API
FROM python:3.12-slim as base

COPY ./badproxy /etc/apt/apt.conf.d/99fixbadproxy

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY automl_framework/ ./automl_framework/
COPY config/ ./config/
COPY run_api.py .

# Create necessary directories
RUN mkdir -p /app/data/uploads /app/logs /app/checkpoints /app/models

# Create non-root user
RUN useradd --create-home --shell /bin/bash automl && \
    chown -R automl:automl /app
USER automl

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "run_api.py"]