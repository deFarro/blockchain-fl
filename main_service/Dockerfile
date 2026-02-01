FROM python:3.12-slim

WORKDIR /app

# Install system dependencies including Node.js for React build
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY main_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared utilities
COPY shared/ /app/shared/

# Copy main service code
COPY main_service/ /app/main_service/

# Build and copy dashboard
WORKDIR /app/dashboard
COPY dashboard/package*.json dashboard/vite.config.js dashboard/index.html ./
RUN npm install
COPY dashboard/src ./src
RUN npm run build

# Set Python path
WORKDIR /app
ENV PYTHONPATH=/app

# Default command (can be overridden in docker-compose)
CMD ["python", "-m", "main_service"]

