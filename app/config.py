import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


settings = Settings()
