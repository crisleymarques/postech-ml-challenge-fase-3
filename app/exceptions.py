from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from app.schemas import ErrorResponse
from app.services.model_service import ModelServiceError, ModelNotLoadedError, InvalidTextError

logger = logging.getLogger(__name__)


def _build_error_body(
    error: str,
    message: str,
    details: Any = None,
    request_id: Optional[str] = None,
) -> dict:
    resp = ErrorResponse(
        error=error,
        message=message,
        details=details,
        request_id=request_id,
        timestamp=datetime.now(timezone.utc),
    )
    return resp.model_dump(mode="json")


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ModelNotLoadedError)
    async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError) -> JSONResponse:
        logger.warning("Modelo nao carregado: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=_build_error_body(
                error="SERVICE_UNAVAILABLE",
                message=(
                    "O modelo de classificacao ainda nao foi carregado. "
                    "Tente novamente em alguns instantes ou contate o administrador."
                ),
                details=None,
            ),
        )

    @app.exception_handler(InvalidTextError)
    async def invalid_text_handler(request: Request, exc: InvalidTextError) -> JSONResponse:
        logger.info("Entrada invalida: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=_build_error_body(
                error="INVALID_INPUT",
                message=str(exc),
            ),
        )

    @app.exception_handler(ModelServiceError)
    async def model_service_handler(request: Request, exc: ModelServiceError) -> JSONResponse:
        logger.error("Erro no ModelService: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_build_error_body(
                error="MODEL_SERVICE_ERROR",
                message=(
                    "Ocorreu um erro no servico de classificacao. "
                    "Tente novamente ou contate o administrador."
                ),
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        simplified = []
        for err in exc.errors():
            loc = ".".join(str(x) for x in err.get("loc", [])) or "request"
            simplified.append({"field": loc, "reason": err.get("msg", "unknown error")})
        logger.info("Erro de validacao da requisicao: %s", simplified)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_build_error_body(
                error="VALIDATION_ERROR",
                message="A requisicao contem campos invalidos ou obrigatorios ausentes.",
                details=simplified,
            ),
        )

    @app.exception_handler(ValidationError)
    async def pydantic_validation_handler(
        request: Request, exc: ValidationError
    ) -> JSONResponse:
        simplified = []
        for err in exc.errors():
            loc = ".".join(str(x) for x in err.get("loc", [])) or "body"
            simplified.append({"field": loc, "reason": err.get("msg", "unknown error")})
        logger.info("Erro de validacao pydantic: %s", simplified)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_build_error_body(
                error="VALIDATION_ERROR",
                message="Dados invalidos.",
                details=simplified,
            ),
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Erro nao tratado na rota %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_build_error_body(
                error="INTERNAL_SERVER_ERROR",
                message=(
                    "Ocorreu um erro interno no servidor durante o processamento da sua solicitacao. "
                    "Tente novamente mais tarde."
                ),
            ),
        )
