from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "API Classificação de Urgência de Laudos Médicos"
    app_version: str = "1.0.0"
    api_prefix: str = ""
    debug: bool = False

    model_name: str = "urgency_classifier"
    models_dir: Optional[str] = None
    vectorizer_name: str = "tfidf_vectorizer"

    # =============================================================================
    # OTIMIZACAO DE INFERENCIA (ONNX Runtime CPU)
    # USE_ONNX:
    #   "1"/True  -> Força carregar modelo .onnx (falha se arquivo/nao existe)
    #   "0"/False -> Desliga completamente, usa sklearn joblib (baseline)
    #   "auto"    -> DEFAULT. Tenta carregar ONNX primeiro; se indisponivel/ausente
    #                cai p/ sklearn joblib SEM interromper a API.
    # =============================================================================
    use_onnx: str = "auto"

    onnx_intra_op_num_threads: int = 1
    onnx_inter_op_num_threads: int = 0
    onnx_execution_mode: str = "sequential"
    onnx_enable_optimizations: bool = True

    request_text_min_length: int = 10
    request_text_max_length: int = 50000

    request_batch_max_items: int = 100

    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    openapi_url: str = "/openapi.json"

    @property
    def models_path(self) -> Path:
        if self.models_dir:
            return Path(self.models_dir).resolve()
        return Path(__file__).resolve().parent.parent / "models"

    @property
    def classifier_path(self) -> Path:
        return self.models_path / f"{self.model_name}.joblib"

    @property
    def classifier_onnx_path(self) -> Path:
        return self.models_path / f"{self.model_name}.onnx"


settings = Settings()
