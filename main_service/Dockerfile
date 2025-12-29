FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY main_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared utilities
COPY shared/ /app/shared/

# Copy main service code
COPY main_service/ /app/main_service/

# Set Python path
ENV PYTHONPATH=/app

# Default command (can be overridden in docker-compose)
CMD ["python", "-m", "main_service"]

