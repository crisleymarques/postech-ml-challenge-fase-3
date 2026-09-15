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

COPY pyproject.toml uv.lock ./

RUN pip install --upgrade pip setuptools wheel uv && \
    uv pip install --system -r pyproject.toml && \
    uv pip install --system "gunicorn>=21.2.0"

COPY app/         ./app/
COPY sources/     ./sources/
COPY pipelines/   ./pipelines/
COPY run_api.py   ./
COPY run_pipeline.py ./

COPY models/.gitkeep ./models/.gitkeep

USER root
RUN mkdir -p ${APP_HOME}/data/raw ${APP_HOME}/data/processed ${APP_HOME}/models

RUN useradd -m --uid 1000 appuser && \
    chown -R appuser:appuser ${APP_HOME}

RUN echo "[INFO] Pastas data/raw, data/processed, models criadas vazias no build." \
    && echo "       Os arquivos reais (modelo .joblib e datasets) serao MONTADOS via volumes no docker-compose.ymll:ro" \
    && echo "       Verifique: ./models:/app/models:ro e ./data:/app/data:ro" \
    && chown -R appuser:appuser ${APP_HOME}/data ${APP_HOME}/models || true
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
