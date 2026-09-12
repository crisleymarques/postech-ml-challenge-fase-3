from __future__ import annotations

from typing import Dict, List, Optional, Any
from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict


class UrgencyProbability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normal: float = Field(ge=0.0, le=1.0, description="Probabilidade da classe Normal (0)")
    atencao: float = Field(ge=0.0, le=1.0, description="Probabilidade da classe Atenção (1)")
    urgente: float = Field(ge=0.0, le=1.0, description="Probabilidade da classe Urgente (2)")


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        ...,
        min_length=1,
        description="Texto do laudo médico / resumo a ser classificado.",
        json_schema_extra={
            "example": (
                "The patient presented with acute chest pain radiating to the left arm, "
                "associated with diaphoresis and dyspnea. ECG showed ST elevation. "
                "Diagnosis: Acute myocardial infarction."
            )
        },
    )
    return_probabilities: bool = Field(
        default=True,
        description="Se True, retorna as probabilidades por classe.",
    )
    request_id: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Identificador opcional fornecido pelo cliente.",
    )


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predicted_label: int = Field(description="Label numérico da urgência (0=normal, 1=atenção, 2=urgente)")
    predicted_name: str = Field(description="Nome da classe de urgência")
    probabilities: Optional[UrgencyProbability] = Field(
        default=None,
        description="Probabilidades por classe (apenas se return_probabilities=True e o modelo suportar predict_proba).",
    )
    model_version: str = Field(description="Versão / nome do modelo utilizado")
    processed_at: datetime = Field(description="Horário UTC da inferência")
    request_id: Optional[str] = Field(default=None, description="Echo do request_id enviado pelo cliente.")


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: List[PredictRequest] = Field(
        ...,
        min_length=1,
        description="Lista de laudos para classificação em lote.",
    )


class BatchPredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_items: int = Field(description="Número total de itens processados.")
    results: List[PredictResponse] = Field(description="Lista com o resultado de cada item.")
    processed_at: datetime = Field(description="Horário UTC da inferência em lote.")


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: str = Field(description="Tipo / título do erro.")
    message: str = Field(description="Mensagem amigável explicando o erro.")
    details: Optional[Any] = Field(default=None, description="Detalhes adicionais (opcional).")
    request_id: Optional[str] = Field(default=None)
    timestamp: datetime = Field(description="Horário UTC do erro.")


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = Field(description="Health status: 'healthy', 'degraded' ou 'unhealthy'.")
    app_name: str
    app_version: str
    model_loaded: bool = Field(description="Se True, o modelo foi carregado com sucesso.")
    model_path_exists: bool = Field(description="Se True, o arquivo do modelo existe no disco.")
    model_name: str
    model_metadata: Optional[Dict[str, Any]] = Field(default=None)
    timestamp: datetime = Field(description="Horário UTC do health check.")
