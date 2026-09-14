from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sources.config import (
    MODELS_DIR,
    URGENCY_LABEL_TO_NAME,
    TEXT_COLUMN,
)

logger = logging.getLogger(__name__)

HAS_ONNX = False
HAS_SKL2ONNX = False

try:
    import onnxruntime as _ort  # type: ignore

    HAS_ONNX = True
except ImportError:
    _ort = None  # type: ignore[assignment]

try:
    from skl2onnx import convert_sklearn as _convert_sklearn  # type: ignore
    from skl2onnx.common.data_types import StringTensorType as _StringTensorType  # type: ignore

    HAS_SKL2ONNX = True
except ImportError:
    _convert_sklearn = None  # type: ignore[assignment]
    _StringTensorType = None  # type: ignore[assignment]

try:
    import onnx as _onnx  # type: ignore
except ImportError:
    _onnx = None  # type: ignore[assignment]


DEFAULT_ONNX_OPSET = 19
DEFAULT_ONNX_MODEL_NAME = "urgency_classifier.onnx"


def is_onnx_available() -> bool:
    return HAS_ONNX


def is_onnx_converter_available() -> bool:
    return HAS_SKL2ONNX


def get_default_onnx_path(models_dir: Path = MODELS_DIR) -> Path:
    return Path(models_dir) / DEFAULT_ONNX_MODEL_NAME


class OnnxConversionError(Exception):
    pass


class OnnxInferenceError(Exception):
    pass


class OnnxInferenceSessionWrapper:
    """
    Wrapper drop-in compatível com sklearn Pipeline.
    Expõe `predict(X)`, `predict_proba(X)` e atributos `classes_`
    (igual ao sklearn) para poder substituir o pipeline original
    no ModelService SEM quebrar a API de inferência.
    """

    def __init__(
        self,
        onnx_path: str | Path,
        intra_op_num_threads: int = 1,
        inter_op_num_threads: int = 0,
        execution_mode: str = "sequential",
        enable_optimizations: bool = True,
        graph_optimization_level: int | None = None,
    ) -> None:
        if not HAS_ONNX:
            raise OnnxInferenceError(
                "onnxruntime nao esta instalado. Instale com: pip install onnxruntime>=1.19"
            )
        path = Path(onnx_path)
        if not path.exists():
            raise OnnxInferenceError(f"Arquivo ONNX nao encontrado: {path}")

        try:
            providers = ["CPUExecutionProvider"]
            session_options = _ort.SessionOptions()
            if enable_optimizations:
                session_options.graph_optimization_level = (
                    graph_optimization_level
                    if graph_optimization_level is not None
                    else _ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
            else:
                session_options.graph_optimization_level = (
                    _ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                )
            try:
                session_options.intra_op_num_threads = int(intra_op_num_threads)
                session_options.inter_op_num_threads = int(inter_op_num_threads)
            except Exception:
                pass
            execution_mode_val = (
                _ort.ExecutionMode.ORT_SEQUENTIAL
                if execution_mode == "sequential"
                else _ort.ExecutionMode.ORT_PARALLEL
            )
            try:
                session_options.execution_mode = execution_mode_val
            except Exception:
                pass

            session_options.log_severity_level = 3  # WARN
            self._session = _ort.InferenceSession(
                str(path),
                sess_options=session_options,
                providers=providers,
            )
            self._onnx_path = path
            meta = self._session.get_inputs()
            if len(meta) < 1:
                raise OnnxInferenceError("Modelo ONNX nao tem inputs registrados")
            self._input_name = meta[0].name
            outs = self._session.get_outputs()
            self._output_names = [o.name for o in outs]
            output_label: Optional[str] = None
            output_proba: Optional[str] = None
            for o in outs:
                otype = (o.type or "").lower()
                oname = o.name.lower()
                if "probabilities" in oname or "probability" in oname:
                    output_proba = o.name
                if "output_label" in oname or "label" in oname or otype.startswith("seq"):
                    if output_label is None:
                        output_label = o.name
            if output_proba is None and len(self._output_names) >= 2:
                output_proba = self._output_names[1]
            if output_label is None and len(self._output_names) >= 1:
                output_label = self._output_names[0]
            self._output_label_name = output_label
            self._output_proba_name = output_proba
            self._classes: Optional[np.ndarray] = None
            self._num_classes: Optional[int] = None
            logger.info(
                "Sessao ONNX criada. Providers=%s, input=%s, outputs=%s, path=%s",
                self._session.get_providers(),
                self._input_name,
                self._output_names,
                self._onnx_path,
            )
        except OnnxInferenceError:
            raise
        except Exception as exc:
            raise OnnxInferenceError(
                f"Falha ao carregar sessao ONNX de {path}: {exc.__class__.__name__}: {exc}"
            ) from exc

    def _ensure_classes(self, sample_labels: np.ndarray) -> None:
        if self._classes is not None:
            return
        uniq = np.unique(np.asarray(sample_labels).astype(np.int64))
        self._classes = uniq
        self._num_classes = int(uniq.size)

    def _to_input_array(self, X: Any) -> np.ndarray:
        if isinstance(X, (pd.DataFrame, pd.Series)):
            if isinstance(X, pd.DataFrame):
                if TEXT_COLUMN in X.columns:
                    arr = X[TEXT_COLUMN].astype(str).to_numpy()
                else:
                    arr = X.iloc[:, 0].astype(str).to_numpy()
            else:
                arr = X.astype(str).to_numpy()
        elif isinstance(X, (list, tuple)):
            arr = np.asarray([str(x) for x in X], dtype=object)
        else:
            arr = np.asarray(X).astype(str)
        if arr.ndim == 0:
            arr = arr.reshape((1,))
        return arr.reshape((-1,))

    def _run_session(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        feed = {self._input_name: X}
        try:
            outs = self._session.run(self._output_names, feed)
        except Exception as exc:
            raise OnnxInferenceError(
                f"Falha durante run da sessao ONNX: {exc.__class__.__name__}: {exc}"
            ) from exc
        return dict(zip(self._output_names, outs))

    def predict(self, X: Any) -> np.ndarray:
        arr = self._to_input_array(X)
        out = self._run_session(arr)
        labels = np.asarray(out[self._output_label_name]).astype(np.int64).reshape((-1,))
        self._ensure_classes(labels)
        return labels

    @staticmethod
    def _list_of_maps_to_matrix(
        list_of_maps: Any,
        expected_classes: Optional[np.ndarray] = None,
        sample_row_out: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        import numpy as _np

        if isinstance(list_of_maps, _np.ndarray):
            try:
                flat = list_of_maps.reshape((-1,))
                if flat.size > 0 and isinstance(flat[0], dict):
                    list_of_maps = flat.tolist()
            except Exception:
                pass
        if not isinstance(list_of_maps, list):
            return _np.asarray(list_of_maps, dtype=np.float64)
        if len(list_of_maps) == 0:
            return _np.zeros((0, len(expected_classes) if expected_classes is not None else 0), dtype=np.float64)
        keys: List[int] = []
        if expected_classes is not None:
            keys = [int(x) for x in expected_classes.tolist()]
        else:
            for row in list_of_maps:
                if isinstance(row, dict):
                    keys = sorted(int(k) for k in row.keys())
                    break
        n = int(len(list_of_maps))
        nc = int(len(keys))
        out_arr = _np.zeros((n, nc), dtype=np.float64)
        key_to_idx = {int(k): j for j, k in enumerate(keys)}
        for i, row in enumerate(list_of_maps):
            if isinstance(row, dict):
                for k, v in row.items():
                    j = key_to_idx.get(int(k))
                    if j is not None:
                        out_arr[i, j] = float(v)
        return out_arr

    def predict_proba(self, X: Any) -> np.ndarray:
        arr = self._to_input_array(X)
        out = self._run_session(arr)
        if self._output_proba_name is None or self._output_proba_name not in out:
            labels = np.asarray(out[self._output_label_name]).astype(np.int64).reshape((-1,))
            self._ensure_classes(labels)
            n = int(labels.shape[0])
            nc = int(self._num_classes or (np.max(labels) + 1 if n else 1))
            proba = np.zeros((n, nc), dtype=np.float32)
            for i, lab in enumerate(labels):
                if lab < nc:
                    proba[i, int(lab)] = 1.0
            return proba
        raw_proba = out[self._output_proba_name]
        is_list_of_maps = (
            isinstance(raw_proba, list) and len(raw_proba) > 0 and isinstance(raw_proba[0], dict)
        )
        if is_list_of_maps:
            proba = self._list_of_maps_to_matrix(raw_proba, expected_classes=self.classes_)
        else:
            proba = np.asarray(raw_proba, dtype=np.float64)
        if proba.ndim == 3 and proba.shape[1] == 1:
            proba = proba.reshape((proba.shape[0], proba.shape[2]))
        if proba.ndim == 1:
            proba = np.column_stack([1.0 - proba, proba])
        labels = np.asarray(out[self._output_label_name]).astype(np.int64).reshape((-1,))
        self._ensure_classes(labels)
        proba = proba.reshape((proba.shape[0], -1))
        if proba.shape[1] < int(self._num_classes or 0):
            pad = int(self._num_classes) - int(proba.shape[1])
            proba = np.hstack([proba, np.zeros((proba.shape[0], pad), dtype=np.float64)])
        norm = np.sum(proba, axis=1, keepdims=True)
        norm[norm <= 0] = 1.0
        proba = proba / norm
        return proba

    @property
    def classes_(self) -> np.ndarray:
        if self._classes is None:
            sorted_labels = sorted(int(k) for k in URGENCY_LABEL_TO_NAME.keys())
            self._classes = np.array(sorted_labels, dtype=np.int64)
            self._num_classes = self._classes.size
        return self._classes

    def __repr__(self) -> str:
        return (
            f"OnnxInferenceSessionWrapper(path={self._onnx_path}, "
            f"intra_op_threads={getattr(self._session.get_session_options(), 'intra_op_num_threads', None)}, "
            f"input={self._input_name}, outputs={self._output_names})"
        )


def export_sklearn_pipeline_to_onnx(
    sklearn_pipeline: Any,
    output_path: str | Path,
    opset: int = DEFAULT_ONNX_OPSET,
    batch_size: Optional[int] = None,
    zipmap: bool = False,
) -> Path:
    """
    Converte um pipeline sklearn (TfidfVectorizer + Classificador) p/ ONNX.
    Exige que as libs `onnx` + `skl2onnx` estejam instaladas.
    """
    if not HAS_ONNX:
        raise OnnxConversionError(
            "onnxruntime nao instalado. Instale: pip install onnxruntime>=1.19"
        )
    if not HAS_SKL2ONNX:
        raise OnnxConversionError(
            "skl2onnx nao instalado. Instale: pip install skl2onnx>=1.17"
        )
    if sklearn_pipeline is None:
        raise OnnxConversionError("Pipeline sklearn eh None")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = None if batch_size is None else int(batch_size)
    initial_types = [
        (
            TEXT_COLUMN,
            _StringTensorType([n]),
        )
    ]
    options_convert = None
    try:
        from sklearn.linear_model import LogisticRegression as _LR

        _set_opts = {}
        # Percorre a arvore do pipeline (se for Pipeline, extrai passos). Garante zipmap=False em
        # TODO LogisticRegression (evita lista de dict no output_probability, melhora performance e
        # simplifica convert_sklearn -> array numpy direto).
        if hasattr(sklearn_pipeline, "named_steps"):
            for step_name, step_obj in sklearn_pipeline.named_steps.items():
                if isinstance(step_obj, _LR):
                    try:
                        from skl2onnx._supported_operators import cluster_tensors  # noqa: F401

                        _set_opts[step_obj] = {"zipmap": False}
                    except Exception:
                        _set_opts[step_obj] = {"zipmap": False}
        elif isinstance(sklearn_pipeline, _LR):
            _set_opts[sklearn_pipeline] = {"zipmap": False}
        if _set_opts:
            options_convert = _set_opts
        logger.info("Options skl2onnx zipmap=False setadas para %d classificador(es).", len(_set_opts))
    except Exception:
        pass
    try:
        onnx_model_bytes = _convert_sklearn(
            sklearn_pipeline,
            initial_types=initial_types,
            target_opset=opset,
            options=options_convert,
            verbose=False,
        )
    except TypeError as exc:
        raise OnnxConversionError(
            f"Falha convert_sklearn (opset={opset}): {exc}. "
            "Tente reduzir target_opset (ex: 17) ou atualizar sklearn."
        ) from exc
    except Exception as exc:
        raise OnnxConversionError(
            f"Falha convert_sklearn generica: {exc.__class__.__name__}: {exc}"
        ) from exc
    try:
        model_proto = onnx_model_bytes if isinstance(onnx_model_bytes, bytes) else None
        if model_proto is None:
            if _onnx is not None and hasattr(onnx_model_bytes, "SerializeToString"):
                model_proto = onnx_model_bytes.SerializeToString()
            else:
                try:
                    model_proto = bytes(onnx_model_bytes)
                except Exception as exc2:
                    raise OnnxConversionError(
                        f"Resultado skl2onnx tem tipo inesperado: {type(onnx_model_bytes)}: {exc2}"
                    ) from exc2
        if _onnx is not None:
            try:
                loaded = _onnx.load_from_string(model_proto)
                _onnx.checker.check_model(loaded)
                logger.info("Validacao ONNX checker OK")
            except Exception as exc:
                logger.warning(
                    "Arquivo ONNX gerado, mas o onnx.checker falhou: %s. "
                    "Isso as vezes ocorre em versoes antigas do onnx; a inferencia ainda deve funcionar.",
                    exc,
                )
        with open(out_path, "wb") as f:
            f.write(model_proto)
    except OnnxConversionError:
        raise
    except Exception as exc:
        raise OnnxConversionError(
            f"Falha ao salvar arquivo ONNX em {out_path}: {exc.__class__.__name__}: {exc}"
        ) from exc
    size_kb = out_path.stat().st_size / 1024.0
    logger.info(
        "Pipeline sklearn convertido para ONNX. Arquivo: %s (%.2f KB, opset=%s).",
        out_path,
        size_kb,
        opset,
    )
    return out_path


def predict_proba_name_labels(
    classes_arr: np.ndarray,
    probabilities_row: np.ndarray,
) -> Dict[str, float]:
    if probabilities_row is None:
        return {}
    proba = np.asarray(probabilities_row).reshape((-1,))
    out: Dict[str, float] = {}
    for i, cls in enumerate(classes_arr):
        cls_int = int(cls)
        name = URGENCY_LABEL_TO_NAME.get(cls_int, str(cls_int))
        out[name] = float(proba[i]) if i < int(proba.shape[0]) else 0.0
    return out
