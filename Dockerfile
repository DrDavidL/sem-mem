FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml ./
COPY sem_mem/ ./sem_mem/

# Install all optional dependencies (server + app + httpx for API client)
RUN pip install --no-cache-dir -e ".[server,app]" httpx

# Copy application files
COPY server.py app.py app_api.py ./

# Create storage directory
RUN mkdir -p /app/local_memory

# Expose ports (API: 8000, Streamlit: 8501)
EXPOSE 8000 8501

# Default: run API server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
