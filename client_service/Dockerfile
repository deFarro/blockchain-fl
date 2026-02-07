# Pre-downloaded dataset (built once by: docker-compose build datasets)
# Requires: make build (which builds datasets first) or docker-compose build datasets
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY client_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-downloaded datasets (from shared datasets image, built once)
COPY --from=blockchain-fl-datasets:latest /app/data /app/data

# Copy shared utilities
COPY shared/ /app/shared/

# Copy client service code
COPY client_service/ /app/client_service/

# Set Python path
ENV PYTHONPATH=/app

# Default command (can be overridden in docker-compose)
CMD ["python", "-m", "client_service"]

