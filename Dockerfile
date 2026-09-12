FROM python:3.11-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_HOME=/app \
    PYTHONPATH=/app

WORKDIR ${APP_HOME}

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "gunicorn>=21.2.0"

COPY app/         ./app/
COPY sources/     ./sources/
COPY pipelines/   ./pipelines/
COPY run_api.py   ./
COPY run_pipeline.py ./

COPY models/.gitkeep ./models/.gitkeep
COPY data/.gitkeep ./data/.gitkeep
COPY data/raw/ ./data/raw/

USER root
RUN useradd -m --uid 1000 appuser && \
    mkdir -p ${APP_HOME}/data/processed ${APP_HOME}/models && \
    chown -R appuser:appuser ${APP_HOME}

COPY models/urgency_classifier.joblib* ./models/ 2>/dev/null || true

USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

ENV PORT=8000 \
    HOST=0.0.0.0 \
    WORKERS=1 \
    WORKER_THREADS=4 \
    WEB_CONCURRENCY=1 \
    TIMEOUT=120

ENV MODELS_DIR=${APP_HOME}/models

CMD ["sh", "-c", "gunicorn app.main:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers ${WORKERS:-1} \
    --threads ${WORKER_THREADS:-4} \
    --bind ${HOST:-0.0.0.0}:${PORT:-8000} \
    --timeout ${TIMEOUT:-120} \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    --keep-alive 5 \
    --preload"]
