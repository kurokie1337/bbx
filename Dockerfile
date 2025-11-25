FROM python:3.11-slim

LABEL maintainer="Blackbox Team"
LABEL description="Blackbox Workflow Engine - Universal automation platform"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/workflows /app/logs

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=info

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run API server
CMD ["python", "api_server.py"]
