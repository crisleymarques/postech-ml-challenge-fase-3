from __future__ import annotations

import sys
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Histogram,
    Info,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

logger = logging.getLogger(__name__)

# =============================================================================
# METRICAS HTTP (Middleware)
# =============================================================================

HTTP_REQUESTS_TOTAL = Counter(
    name="http_requests_total",
    documentation="Numero total de requisicoes HTTP recebidas pela API.",
    labelnames=["method", "endpoint", "status_code"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    name="http_request_duration_seconds",
    documentation=(
        "Histograma de latencia (tempo de processamento) das requisicoes HTTP "
        "processadas pela API, em segundos."
    ),
    labelnames=["method", "endpoint"],
    buckets=(
        0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0
    ),
)

HTTP_ERRORS_TOTAL = Counter(
    name="http_errors_total",
    documentation=(
        "Numero total de respostas HTTP com status de erro (4xx ou 5xx), "
        "agrupadas por endpoint, status numerico e familia de status."
    ),
    labelnames=["method", "endpoint", "status_code", "status_family"],
)

# =============================================================================
# METRICAS ESPECIFICAS DE INFERENCIA / ML
# =============================================================================

INFERENCE_PREDICTIONS_TOTAL = Counter(
    name="inference_predictions_total",
    documentation=(
        "Numero total de inferencias (predicoes individuais) realizadas com "
        "sucesso pelo modelo, separadas por tipo de endpoint, classe predita "
        "e nome do modelo."
    ),
    labelnames=["endpoint_type", "predicted_class", "model_name"],
)

INFERENCE_LATENCY_SECONDS = Histogram(
    name="inference_latency_seconds",
    documentation=(
        "Histograma de latencia exclusiva do passo de inferencia do modelo "
        "(predict do sklearn), excluindo overhead HTTP, validacoes e serializacao."
    ),
    labelnames=["endpoint_type"],
    buckets=(0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0),
)

INFERENCE_ERRORS_TOTAL = Counter(
    name="inference_errors_total",
    documentation=(
        "Numero total de falhas durante a inferencia, separadas por tipo de "
        "endpoint e tipo de erro."
    ),
    labelnames=["endpoint_type", "error_type"],
)

INFERENCE_ITEMS_PROCESSED_TOTAL = Counter(
    name="inference_items_processed_total",
    documentation=(
        "Contador acumulado de itens individuais processados por endpoint. "
        "Para /predict sempre 1. Para /predict/batch pode ser >1."
    ),
    labelnames=["endpoint_type"],
)

# =============================================================================
# METRICAS DO MODELO / DEPLOY
# =============================================================================

MODEL_LOADED_GAUGE = Gauge(
    name="model_loaded",
    documentation=(
        "Indicador binario (0 ou 1) de se o modelo de classificacao esta "
        "corretamente carregado em memoria. 1 = carregado, 0 = nao carregado."
    ),
    labelnames=["model_name"],
)

MODEL_INFO = Info(
    name="model_metadata",
    documentation=(
        "Metadados estaticos do modelo atualmente carregado: nome, versao, "
        "data de treino, acuracia registrada, etc."
    ),
)


# =============================================================================
# HELPERS PUBLICOS
# =============================================================================

def _safe_family(status_code: int) -> str:
    code = int(status_code)
    if 400 <= code < 500:
        return "4xx"
    if 500 <= code < 600:
        return "5xx"
    return "other"


def observe_http_request(
    method: str,
    endpoint: str,
    status_code: int,
    duration_seconds: float,
) -> None:
    """Registra contador e latencia para uma requisicao HTTP completa."""
    m = (method or "GET").upper()
    ep = endpoint or "unknown"
    code = int(status_code)
    try:
        HTTP_REQUESTS_TOTAL.labels(method=m, endpoint=ep, status_code=str(code)).inc()
        HTTP_REQUEST_DURATION_SECONDS.labels(method=m, endpoint=ep).observe(
            max(0.0, float(duration_seconds))
        )
        if code >= 400:
            HTTP_ERRORS_TOTAL.labels(
                method=m,
                endpoint=ep,
                status_code=str(code),
                status_family=_safe_family(code),
            ).inc()
    except Exception as exc:  # pragma: no cover - observability nunca quebra request
        logger.debug("Falha ao observar http request metrics: %s", exc)


def observe_inference_prediction(
    endpoint_type: str,
    predicted_class: str,
    model_name: str,
    latency_seconds: float,
    n_items: int = 1,
) -> None:
    """Registra metricas de uma INFERENCIA BEM SUCEDIDA (predicao)."""
    et = endpoint_type or "unknown"
    cls = str(predicted_class)
    mn = str(model_name)
    n = max(1, int(n_items))
    try:
        INFERENCE_PREDICTIONS_TOTAL.labels(
            endpoint_type=et, predicted_class=cls, model_name=mn
        ).inc(n)
        INFERENCE_LATENCY_SECONDS.labels(endpoint_type=et).observe(
            max(0.0, float(latency_seconds))
        )
        INFERENCE_ITEMS_PROCESSED_TOTAL.labels(endpoint_type=et).inc(n)
    except Exception as exc:  # pragma: no cover
        logger.debug("Falha ao observar inference prediction metrics: %s", exc)


def observe_inference_error(
    endpoint_type: str,
    error_type: str,
) -> None:
    """Registra contador de erro de inferencia."""
    et = endpoint_type or "unknown"
    err = str(error_type)
    try:
        INFERENCE_ERRORS_TOTAL.labels(endpoint_type=et, error_type=err).inc()
    except Exception as exc:  # pragma: no cover
        logger.debug("Falha ao observar inference error metrics: %s", exc)


def update_model_loaded_state(
    loaded: bool,
    model_name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Atualiza o Gauge e Info do modelo apos carregar/descarregar."""
    mn = str(model_name or "unknown")
    try:
        MODEL_LOADED_GAUGE.labels(model_name=mn).set(1 if bool(loaded) else 0)
        info_payload: Dict[str, str] = {"model_name": mn}
        if metadata:
            for k in ("accuracy", "f1_macro", "n_train_samples", "model_version", "airflow_run_id"):
                v = metadata.get(k)
                if v is not None:
                    info_payload[str(k)] = str(v)
        MODEL_INFO.info(info_payload)
    except Exception as exc:  # pragma: no cover
        logger.debug("Falha ao atualizar model_loaded metrics: %s", exc)


def render_metrics(registry: Optional[CollectorRegistry] = None) -> bytes:
    """Retorna os bytes do scrape Prometheus (formato texto exposition 0.0.4)."""
    try:
        return generate_latest(registry or REGISTRY)
    except Exception as exc:  # pragma: no cover
        logger.exception("Falha ao renderizar metricas Prometheus: %s", exc)
        return b""


def get_prometheus_content_type() -> str:
    return CONTENT_TYPE_LATEST


# =============================================================================
# TIMER CONTEXT MANAGER (util para medir duracoes)
# =============================================================================

class _MetricTimer:
    __slots__ = ("_start", "_elapsed")

    def __init__(self) -> None:
        self._start: float = 0.0
        self._elapsed: float = 0.0

    def __enter__(self) -> "_MetricTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._elapsed = time.perf_counter() - self._start

    @property
    def elapsed_seconds(self) -> float:
        if self._elapsed:
            return float(self._elapsed)
        return float(time.perf_counter() - self._start)


def metric_timer() -> _MetricTimer:
    return _MetricTimer()
