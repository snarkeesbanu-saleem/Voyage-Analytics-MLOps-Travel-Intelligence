# =============================================================================
# Voyage Analytics Flask API Dockerfile
# =============================================================================
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.10-slim

WORKDIR /app

# Copy python dependencies from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy source and api directories
COPY config.yaml .
COPY src/ ./src
COPY api/ ./api
COPY models/ ./models
COPY pipelines/ ./pipelines
COPY mlflow_tracking/ ./mlflow_tracking

EXPOSE 5000

ENV FLASK_APP=api/app.py
ENV PYTHONPATH=/app

# Healthcheck to probe the /health endpoint of Flask app
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import requests; r=requests.get('http://localhost:5000/health'); exit(0 if r.status_code == 200 else 1)"

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "api.app:create_app()"]
