from __future__ import annotations

import sys
import logging
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import joblib
import pandas as pd

from sources.config import URGENCY_LABEL_TO_NAME
from app.config import settings
from app.metrics import (
    observe_inference_prediction,
    observe_inference_error,
    update_model_loaded_state,
)

logger = logging.getLogger(__name__)


class ModelServiceError(Exception):
    pass


class ModelNotLoadedError(ModelServiceError):
    pass


class InvalidTextError(ModelServiceError):
    pass


class ModelService:
    def __init__(self, config: Optional[Any] = None):
        self.config = config or settings
        self._pipeline = None
        self._metadata: Dict[str, Any] = {}
        self._loaded_at: Optional[datetime] = None

    # ---- Carregamento do modelo ----
    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(self._metadata)

    @property
    def loaded_at(self) -> Optional[datetime]:
        return self._loaded_at

    def model_file_exists(self) -> bool:
        return self.config.classifier_path.exists()

    def load_model(self, force: bool = False) -> None:
        if self.is_loaded and not force:
            return
        path = self.config.classifier_path
        if not path.exists():
            try:
                mn = str(self._metadata.get("model_name", self.config.model_name))
                update_model_loaded_state(False, mn)
            except Exception:
                pass
            raise ModelServiceError(
                f"Arquivo de modelo nao encontrado em: {path}. "
                "Execute o pipeline de treinamento primeiro (run_pipeline.py)."
            )
        try:
            artifact = joblib.load(path)
        except Exception as exc:
            try:
                mn = str(self._metadata.get("model_name", self.config.model_name))
                update_model_loaded_state(False, mn)
            except Exception:
                pass
            raise ModelServiceError(f"Falha ao carregar o modelo: {exc.__class__.__name__}") from exc

        if isinstance(artifact, dict) and "model" in artifact:
            self._pipeline = artifact.get("model")
            self._metadata = artifact.get("metadata", {}) or {}
        else:
            self._pipeline = artifact
            self._metadata = {}

        self._loaded_at = datetime.now(timezone.utc)
        mn = str(self._metadata.get("model_name", self.config.model_name))
        update_model_loaded_state(True, mn, self._metadata)
        logger.info(
            "Modelo carregado com sucesso de %s (loaded_at=%s, classes=%s)",
            path,
            self._loaded_at.isoformat(),
            list(URGENCY_LABEL_TO_NAME.values()),
        )

    def unload(self) -> None:
        mn = str(self._metadata.get("model_name", self.config.model_name))
        self._pipeline = None
        self._metadata = {}
        self._loaded_at = None
        try:
            update_model_loaded_state(False, mn)
        except Exception:
            pass

    # ---- Validações ----
    def _validate_text(self, text: str) -> None:
        if not isinstance(text, str):
            raise InvalidTextError("O campo 'text' deve ser uma string.")
        stripped = text.strip()
        if len(stripped) == 0:
            raise InvalidTextError("O campo 'text' nao pode estar vazio ou conter apenas espacos.")
        if len(stripped) < self.config.request_text_min_length:
            raise InvalidTextError(
                f"O campo 'text' deve ter pelo menos {self.config.request_text_min_length} caracteres."
            )
        if len(stripped) > self.config.request_text_max_length:
            raise InvalidTextError(
                f"O campo 'text' deve ter no maximo {self.config.request_text_max_length} caracteres."
            )

    # ---- Inferência ----
    def predict_single(
        self,
        text: str,
        return_probabilities: bool = True,
    ) -> Dict[str, Any]:
        endpoint_type = "single"
        model_name = str(self._metadata.get("model_name", self.config.model_name))
        if not self.is_loaded:
            observe_inference_error(endpoint_type, "model_not_loaded")
            raise ModelNotLoadedError("Modelo nao foi carregado ainda. Chame load_model().")
        try:
            self._validate_text(text)
        except InvalidTextError as exc:
            observe_inference_error(endpoint_type, "invalid_text")
            raise exc
        except ModelServiceError as exc:
            observe_inference_error(endpoint_type, "validation_error")
            raise exc

        text_series = pd.Series([text])
        t0 = time.perf_counter()
        try:
            y_pred = self._pipeline.predict(text_series)
            latency = max(0.0, time.perf_counter() - t0)
        except Exception as exc:
            observe_inference_error(endpoint_type, "predict_exception")
            raise ModelServiceError(f"Falha durante predict do modelo: {exc.__class__.__name__}") from exc

        label = int(y_pred[0])
        name = URGENCY_LABEL_TO_NAME.get(label, f"desconhecido_{label}")

        result: Dict[str, Any] = {
            "predicted_label": label,
            "predicted_name": name,
            "probabilities": None,
            "model_version": model_name,
            "processed_at": datetime.now(timezone.utc),
        }

        if return_probabilities and hasattr(self._pipeline, "predict_proba"):
            try:
                proba = self._pipeline.predict_proba(text_series)[0]
                classes = list(self._pipeline.classes_)
                proba_map: Dict[str, float] = {}
                for i, cls in enumerate(classes):
                    cls_name = URGENCY_LABEL_TO_NAME.get(int(cls), str(cls))
                    proba_map[cls_name] = float(proba[i])
                result["probabilities"] = {
                    "normal": proba_map.get("normal", 0.0),
                    "atencao": proba_map.get("atencao", 0.0),
                    "urgente": proba_map.get("urgente", 0.0),
                }
            except Exception:
                result["probabilities"] = None

        observe_inference_prediction(
            endpoint_type=endpoint_type,
            predicted_class=name,
            model_name=model_name,
            latency_seconds=latency,
            n_items=1,
        )
        return result

    def predict_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        endpoint_type = "batch"
        model_name = str(self._metadata.get("model_name", self.config.model_name))
        if not self.is_loaded:
            observe_inference_error(endpoint_type, "model_not_loaded")
            raise ModelNotLoadedError("Modelo nao foi carregado ainda.")
        max_items = self.config.request_batch_max_items
        n = len(items)
        if n > max_items:
            observe_inference_error(endpoint_type, "batch_too_large")
            raise InvalidTextError(
                f"Batch excedeu o limite maximo de {max_items} itens. "
                f"Recebido: {len(items)}"
            )
        results: List[Dict[str, Any]] = []
        for item in items:
            results.append(
                self.predict_single(
                    text=item["text"],
                    return_probabilities=bool(item.get("return_probabilities", True)),
                )
            )
        return results


_model_service_instance: Optional[ModelService] = None


def get_model_service() -> ModelService:
    global _model_service_instance
    if _model_service_instance is None:
        _model_service_instance = ModelService()
    return _model_service_instance
