from __future__ import annotations

import sys
import logging
import time
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Depends, Request, status
from fastapi.responses import JSONResponse, Response

from app.config import settings
from app.schemas import (
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
)
from app.services.model_service import (
    ModelService,
    get_model_service,
    InvalidTextError,
)
from app.exceptions import register_exception_handlers
from app.metrics import (
    observe_http_request,
    update_model_loaded_state,
    render_metrics,
    get_prometheus_content_type,
)

logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_svc: ModelService = get_model_service()
    try:
        if model_svc.model_file_exists():
            logger.info("Arquivo de modelo detectado. Carregando modelo na inicializacao...")
            model_svc.load_model(force=True)
            update_model_loaded_state(
                loaded=True,
                model_name=model_svc.metadata.get("model_name", settings.model_name),
                metadata=model_svc.metadata,
            )
            logger.info(
                "Modelo carregado com sucesso (versao=%s). API pronta.",
                model_svc.metadata.get("model_name", settings.model_name),
            )
        else:
            update_model_loaded_state(loaded=False, model_name=settings.model_name)
            logger.warning(
                "Arquivo de modelo NAO encontrado em %s. "
                "Execute run_pipeline.py para gerar o modelo. "
                "A API vai iniciar, mas /predict retornara 503.",
                settings.classifier_path,
            )
    except Exception as exc:
        update_model_loaded_state(loaded=False, model_name=settings.model_name)
        logger.error("Falha durante o carregamento inicial do modelo: %s", exc)
    yield
    logger.info("API encerrando...")
    model_svc.unload()
    update_model_loaded_state(loaded=False, model_name=settings.model_name)


def _resolve_route_template(request: Request) -> str:
    """Resolve o template da rota (evita alta cardinalidade com path params)."""
    try:
        route = request.scope.get("route")
        if route is not None and hasattr(route, "path"):
            return str(route.path)
    except Exception:
        pass
    raw = request.url.path or ""
    prefix = settings.api_prefix
    if raw.startswith(f"{prefix}/predict/"):
        return f"{prefix}/predict/*"
    if raw == prefix or raw == f"{prefix}/":
        return f"{prefix}/"
    return raw


def _register_prometheus_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def _http_observability_middleware(request: Request, call_next):
        # Skip o proprio /metrics (evita auto-referencia / alta cardinalidade)
        skip = (request.method.upper() == "GET" and request.url.path == "/metrics")
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = int(response.status_code)
            return response
        except Exception as exc:
            status_code = 500
            raise exc
        finally:
            if not skip:
                duration = max(0.0, time.perf_counter() - started)
                method = request.method.upper()
                endpoint = _resolve_route_template(request)
                try:
                    observe_http_request(
                        method=method,
                        endpoint=endpoint,
                        status_code=status_code,
                        duration_seconds=duration,
                    )
                except Exception as exc_observer:  # pragma: no cover
                    logger.debug("observe_http_request falhou: %s", exc_observer)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "API de triagem automatica de urgencia em laudos medicos. "
            "Recebe o texto do laudo e retorna a classificacao de urgencia "
            "(normal / atencao / urgente) utilizando um modelo NLP leve treinado "
            "sobre o Medical Abstracts TC Corpus Dataset. Exibe metricas Prometheus "
            "em /metrics."
        ),
        docs_url=settings.docs_url,
        redoc_url=settings.redoc_url,
        openapi_url=settings.openapi_url,
        lifespan=lifespan,
    )
    _register_prometheus_middleware(app)
    register_exception_handlers(app)
    _register_routes(app)
    return app


def _register_routes(app: FastAPI) -> None:
    prefix = settings.api_prefix

    @app.get(
        "/metrics",
        summary="Metricas Prometheus",
        description=(
            "Expõe todas as métricas de observabilidade (requests, latência, "
            "erros, inferências, estado do modelo) no formato texto padrão "
            "do Prometheus (exposition format 0.0.4)."
        ),
        responses={
            200: {
                "description": "Metricas Prometheus",
                "content": {"text/plain": {}},
            },
        },
        tags=["Observabilidade"],
    )
    async def metrics() -> Response:
        payload = render_metrics()
        return Response(
            content=payload,
            media_type=get_prometheus_content_type(),
            status_code=status.HTTP_200_OK,
        )

    @app.get(
        f"{prefix}/health",
        response_model=HealthResponse,
        summary="Health check do servico",
        description=(
            "Retorna o status de saude da API: se o app esta no ar, "
            "se o modelo esta carregado, e metadados do deploy."
        ),
        responses={
            200: {"description": "Health check concluido"},
            503: {"description": "Servico indisponivel / modelo nao carregado"},
        },
        tags=["Health"],
    )
    async def health(
        model_svc: ModelService = Depends(get_model_service),
    ) -> JSONResponse:
        model_loaded = model_svc.is_loaded
        model_file_ok = model_svc.model_file_exists()

        if model_loaded:
            status_str = "healthy"
            http_code = status.HTTP_200_OK
        elif model_file_ok:
            status_str = "degraded"
            http_code = status.HTTP_200_OK
        else:
            status_str = "unhealthy"
            http_code = status.HTTP_503_SERVICE_UNAVAILABLE

        meta = model_svc.metadata or None
        body = HealthResponse(
            status=status_str,
            app_name=settings.app_name,
            app_version=settings.app_version,
            model_loaded=model_loaded,
            model_path_exists=model_file_ok,
            model_name=settings.model_name,
            model_metadata=meta,
            timestamp=datetime.now(timezone.utc),
        ).model_dump(mode="json")
        return JSONResponse(status_code=http_code, content=body)

    @app.post(
        f"{prefix}/predict",
        response_model=PredictResponse,
        summary="Classificar urgencia de 1 laudo",
        description=(
            "Recebe o texto de um laudo medico e retorna sua classificacao "
            "de urgencia, com probabilidades opcionais por classe. "
            "Atualiza automaticamente as metricas Prometheus: "
            "inference_predictions_total, inference_latency_seconds, etc."
        ),
        responses={
            200: {"description": "Classificacao realizada com sucesso"},
            400: {"description": "Texto de entrada invalido"},
            422: {"description": "Payload da requisicao invalido"},
            503: {"description": "Modelo nao carregado"},
            500: {"description": "Erro interno do servidor"},
        },
        tags=["Inferência"],
    )
    async def predict(
        body: PredictRequest,
        model_svc: ModelService = Depends(get_model_service),
    ) -> PredictResponse:
        result = model_svc.predict_single(
            text=body.text,
            return_probabilities=body.return_probabilities,
        )
        resp = PredictResponse(
            predicted_label=result["predicted_label"],
            predicted_name=result["predicted_name"],
            probabilities=result["probabilities"],
            model_version=result["model_version"],
            processed_at=result["processed_at"],
            request_id=body.request_id,
        )
        return resp

    @app.post(
        f"{prefix}/predict/batch",
        response_model=BatchPredictResponse,
        summary="Classificar urgencia de varios laudos em lote",
        description=(
            "Recebe uma lista de laudos medicos e retorna a classificacao "
            "individual de cada um. Limite maximo de 100 itens por requisicao. "
            "Atualiza automaticamente as metricas Prometheus por item processado."
        ),
        responses={
            200: {"description": "Lote processado com sucesso"},
            400: {"description": "Item invalido ou excedeu o tamanho maximo do lote"},
            422: {"description": "Payload invalido"},
            503: {"description": "Modelo nao carregado"},
            500: {"description": "Erro interno"},
        },
        tags=["Inferência"],
    )
    async def predict_batch(
        body: BatchPredictRequest,
        model_svc: ModelService = Depends(get_model_service),
    ) -> BatchPredictResponse:
        raw_items = body.items  # type: ignore[assignment]
        item_list = [item.model_dump() for item in raw_items]
        raw_results = model_svc.predict_batch(item_list)
        request_ids = [item.request_id for item in raw_items]

        results = []
        for raw, req_id in zip(raw_results, request_ids):
            results.append(
                PredictResponse(
                    predicted_label=raw["predicted_label"],
                    predicted_name=raw["predicted_name"],
                    probabilities=raw["probabilities"],
                    model_version=raw["model_version"],
                    processed_at=raw["processed_at"],
                    request_id=req_id,
                )
            )

        return BatchPredictResponse(
            total_items=len(results),
            results=results,
            processed_at=datetime.now(timezone.utc),
        )

    @app.get(f"{prefix}/", include_in_schema=False)
    async def root() -> Dict[str, Any]:
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "status": "running",
            "docs": settings.docs_url,
            "redoc": settings.redoc_url,
            "health": f"{prefix}/health",
            "metrics": "/metrics",
            "endpoints": {
                "health": f"{prefix}/health",
                "metrics": "GET /metrics",
                "predict_single": f"POST {prefix}/predict",
                "predict_batch": f"POST {prefix}/predict/batch",
            },
        }


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        workers=1,
    )
