from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sources.config import (
    MODELS_DIR,
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    TEST_FILE,
    URGENCY_COLUMN,
    TEXT_COLUMN,
    TARGET_COLUMN,
    CONDITION_TO_URGENCY,
    setup_dirs,
)

TEST_CSV_CANDIDATES = [
    DATA_PROCESSED_DIR / "medical_tc_test.csv",
    DATA_PROCESSED_DIR / "test_processed.csv",
    TEST_FILE,
]
LABEL_COLUMN = URGENCY_COLUMN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("optimize_model")


DEFAULT_SAMPLES = 20000


def import_deps_or_exit() -> None:
    missing: List[str] = []
    try:
        import pandas as _  # noqa: F401
    except ImportError:
        missing.append("pandas")
    try:
        import joblib  # noqa: F401
    except ImportError:
        missing.append("joblib")
    try:
        import sklearn  # noqa: F401
    except ImportError:
        missing.append("scikit-learn")
    try:
        from sources.model_onnx import (
            export_sklearn_pipeline_to_onnx,
            OnnxInferenceSessionWrapper,
            is_onnx_available,
            is_onnx_converter_available,
        )  # noqa: F401

        if not is_onnx_available():
            missing.append("onnxruntime (verifique pip install onnxruntime>=1.19)")
        if not is_onnx_converter_available():
            missing.append("skl2onnx (verifique pip install skl2onnx>=1.17)")
    except ImportError as exc:
        missing.append(f"sources.model_onnx (deps faltando: {exc})")
    if missing:
        logger.error("Dependencias faltando para otimizacao ONNX:\n- %s", "\n- ".join(missing))
        raise SystemExit(2)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "[Atividade8-Otimizacao] Exporta o pipeline sklearn salvo (joblib) "
            "para ONNX Runtime CPU, valida equivalencia (>=99% label match + delta proba "
            "medio < 0.02) no dataset teste, salva modelo otimizado e report JSON."
        )
    )
    p.add_argument(
        "--joblib-path",
        type=str,
        default=str(MODELS_DIR / "urgency_classifier.joblib"),
        help="Caminho para o modelo sklearn joblib existente (default: models/urgency_classifier.joblib)",
    )
    p.add_argument(
        "--onnx-path",
        type=str,
        default=str(MODELS_DIR / "urgency_classifier.onnx"),
        help="Caminho de saida do modelo otimizado ONNX (default: models/urgency_classifier.onnx)",
    )
    p.add_argument(
        "--test-csv",
        type=str,
        default=str(TEST_CSV_CANDIDATES[0]),
        help=(
            "CSV split de teste p/ validar equivalencia. "
            f"Ordem fallback automatica: {[str(p) for p in TEST_CSV_CANDIDATES]}"
        ),
    )
    p.add_argument(
        "--opset",
        type=int,
        default=19,
        help="Target opset ONNX (padrao 19; se houver erro de conversao, tente 17)",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help="Max amostras no teste para equivalencia (default 20000, -1 = todos)",
    )
    p.add_argument(
        "--min-accuracy-match",
        type=float,
        default=0.99,
        help="Minimo de acuracia relativa sklearn vs ONNX (default 0.99)",
    )
    p.add_argument(
        "--max-mean-abs-diff-proba",
        type=float,
        default=0.02,
        help="Diferenca maxima absoluta media de probabilidades (default 0.02)",
    )
    p.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Caminho JSON de saida do relatorio (default: benchmarks/results/onnx_parity_YYYYMMDD_HHMMSS.json)",
    )
    p.add_argument(
        "--skip-parity",
        action="store_true",
        help="Apenas exporta, sem validacao de equivalencia",
    )
    return p.parse_args()


def _load_sklearn_pipeline(joblib_path: Path) -> Tuple[Any, Dict[str, Any]]:
    import joblib

    logger.info("Carregando pipeline sklearn de %s", joblib_path)
    t0 = time.perf_counter()
    artifact = joblib.load(joblib_path)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    if isinstance(artifact, dict) and "model" in artifact:
        pipeline = artifact["model"]
        metadata = dict(artifact.get("metadata", {}) or {})
    else:
        pipeline = artifact
        metadata = {}
    logger.info("Pipeline sklearn carregado em %.2f ms (path: %s)", dt_ms, joblib_path)
    return pipeline, metadata, dt_ms  # type: ignore[return-value]


def _resolve_test_csv(test_csv: str) -> Path:
    user_provided = Path(test_csv).resolve()
    if user_provided.exists():
        return user_provided
    for cand in TEST_CSV_CANDIDATES:
        if cand.exists():
            logger.warning(
                "CSV informado %s nao encontrado. Usando fallback %s.",
                test_csv, cand,
            )
            return cand.resolve()
    raise FileNotFoundError(
        "Nenhum CSV de teste encontrado. "
        f"Candidatos tentados: {[str(x) for x in [user_provided] + TEST_CSV_CANDIDATES]}. "
        "Gere o split rodando `python run_pipeline.py` antes."
    )


def _ensure_urgency_label(df: "pd.DataFrame") -> "pd.DataFrame":
    import pandas as pd

    cols = set(df.columns)
    if LABEL_COLUMN in cols and df[LABEL_COLUMN].notna().any():
        df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors="coerce").astype(int)
        return df
    if TARGET_COLUMN in cols:
        logger.warning(
            "Coluna %s nao existe. Gerando-a a partir de %s via CONDITION_TO_URGENCY.",
            LABEL_COLUMN,
            TARGET_COLUMN,
        )
        df[LABEL_COLUMN] = df[TARGET_COLUMN].map(CONDITION_TO_URGENCY)
        df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors="coerce").astype(int)
        return df
    raise ValueError(
        f"Nao foi possivel encontrar coluna de label. Esperado {LABEL_COLUMN} ou {TARGET_COLUMN}. "
        f"Colunas encontradas: {list(df.columns)}"
    )


def _load_test_df(test_csv: str, samples: int) -> "pd.DataFrame":
    import pandas as pd

    setup_dirs()
    csv_path = _resolve_test_csv(test_csv)
    df = pd.read_csv(csv_path)
    if TEXT_COLUMN not in df.columns:
        raise ValueError(
            f"Esperado coluna de texto {TEXT_COLUMN!r} no CSV. "
            f"Recebido colunas: {list(df.columns)}"
        )
    df = _ensure_urgency_label(df)
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]).reset_index(drop=True)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)
    if samples is not None and int(samples) > 0 and len(df) > int(samples):
        df = df.sample(n=int(samples), random_state=42).reset_index(drop=True)
    logger.info("Dataset equivalencia carregado: %s (n=%d)", csv_path, len(df))
    return df


def _parity_evaluation(
    sklearn_pipeline,
    onnx_wrapper,
    df,
    min_match: float,
    max_mean_abs_diff_proba: float,
) -> Dict[str, Any]:
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
    )

    texts = df[TEXT_COLUMN].astype(str).tolist()
    y_true = df[LABEL_COLUMN].astype(int).to_numpy()
    logger.info("Executando equivalencia sklearn... (%d amostras)", len(df))
    t0 = time.perf_counter()
    y_sk = np.asarray(sklearn_pipeline.predict(pd.Series(texts))).astype(int)
    t_sk = (time.perf_counter() - t0) * 1000.0
    proba_sk = np.asarray(sklearn_pipeline.predict_proba(pd.Series(texts)), dtype=np.float64)
    logger.info("Executando equivalencia ONNX... (%d amostras)", len(df))
    t1 = time.perf_counter()
    y_on = np.asarray(onnx_wrapper.predict(texts)).astype(int)
    t_on = (time.perf_counter() - t1) * 1000.0
    proba_on = np.asarray(onnx_wrapper.predict_proba(texts), dtype=np.float64)

    if proba_sk.shape != proba_on.shape:
        logger.warning(
            "Shape proba diferente sklearn=%s vs onnx=%s. Ajustando predict_proba onnx...",
            proba_sk.shape,
            proba_on.shape,
        )
        if proba_on.shape[0] == proba_sk.shape[0] and proba_on.ndim == 3:
            proba_on = proba_on.reshape((proba_on.shape[0], -1))[:, : proba_sk.shape[1]]
        else:
            raise ValueError(
                f"Nao foi possivel alinhar shapes proba: sk={proba_sk.shape}, onx={proba_on.shape}"
            )

    y_match = np.mean(y_sk == y_on)
    mean_abs_diff_proba = float(np.mean(np.abs(proba_sk - proba_on)))
    max_abs_diff_proba = float(np.max(np.abs(proba_sk - proba_on)))
    acc_sk = float(accuracy_score(y_true, y_sk))
    acc_on = float(accuracy_score(y_true, y_on))
    f1_sk = float(f1_score(y_true, y_sk, average="macro", zero_division=0))
    f1_on = float(f1_score(y_true, y_on, average="macro", zero_division=0))
    delta_acc = float(acc_on - acc_sk)
    delta_f1 = float(f1_on - f1_sk)
    speedup_total = float("inf") if t_on <= 0 else float(t_sk / t_on)
    report = {
        "n_samples": int(len(df)),
        "accuracy_match_sklearn_vs_onnx": float(y_match),
        "mean_abs_diff_probability": float(mean_abs_diff_proba),
        "max_abs_diff_probability": float(max_abs_diff_proba),
        "accuracy_vs_ytrue_sklearn": float(acc_sk),
        "accuracy_vs_ytrue_onnx": float(acc_on),
        "delta_accuracy_onnx_minus_sklearn": delta_acc,
        "f1_macro_vs_ytrue_sklearn": f1_sk,
        "f1_macro_vs_ytrue_onnx": f1_on,
        "delta_f1_macro_onnx_minus_sklearn": delta_f1,
        "total_predict_time_sklearn_ms": float(t_sk),
        "total_predict_time_onnx_ms": float(t_on),
        "onnx_speedup_vs_sklearn_total": speedup_total,
        "constraints": {
            "min_accuracy_match_required": float(min_match),
            "max_mean_abs_diff_proba_required": float(max_mean_abs_diff_proba),
            "match_ok": bool(y_match >= float(min_match)),
            "proba_ok": bool(mean_abs_diff_proba <= float(max_mean_abs_diff_proba)),
        },
    }
    return report


def _artifact_sizes(joblib_path: Path, onnx_path: Path) -> Dict[str, Any]:
    def _kb(p: Path):
        return (p.stat().st_size / 1024.0) if p.exists() else 0.0

    return {
        "joblib_kb": float(round(_kb(joblib_path), 2)),
        "onnx_kb": float(round(_kb(onnx_path), 2)),
        "diff_pct_onnx_vs_joblib": float(
            round(((_kb(onnx_path) - _kb(joblib_path)) / _kb(joblib_path)) * 100.0, 2)
        )
        if _kb(joblib_path)
        else None,
    }


def main() -> int:
    import_deps_or_exit()
    args = _parse_args()
    joblib_path = Path(args.joblib_path).resolve()
    onnx_path = Path(args.onnx_path).resolve()
    test_csv = Path(args.test_csv).resolve()
    if not joblib_path.exists():
        logger.error("Arquivo joblib nao encontrado: %s", joblib_path)
        return 2
    pipeline, metadata_joblib, load_sk_ms = _load_sklearn_pipeline(joblib_path)
    report: Dict[str, Any] = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {
            "joblib_path": str(joblib_path),
            "onnx_path": str(onnx_path),
            "test_csv": str(test_csv),
            "opset": int(args.opset),
            "samples": int(args.samples),
        },
        "sklearn_original_metadata": metadata_joblib,
        "sklearn_load_time_ms": float(load_sk_ms),
    }
    from sources.model_onnx import export_sklearn_pipeline_to_onnx, OnnxInferenceSessionWrapper

    t_exp = time.perf_counter()
    try:
        onnx_written = export_sklearn_pipeline_to_onnx(
            pipeline,
            onnx_path,
            opset=int(args.opset),
            zipmap=False,
        )
    except Exception as exc:
        logger.error(
            "Falhou export ONNX (opset=%s): %s: %s",
            args.opset,
            exc.__class__.__name__,
            exc,
        )
        if int(args.opset) > 17:
            logger.warning(
                "Tente novamente com --opset 17 (algumas versoes skl2onnx "
                "tem problemas com opset 18/19 em combinacoes sklearn 1.5 / Python 3.13)."
            )
        return 3
    export_ms = (time.perf_counter() - t_exp) * 1000.0
    report["export"] = {
        "opset": int(args.opset),
        "output_path": str(onnx_written),
        "time_ms": float(round(export_ms, 2)),
    }
    sizes = _artifact_sizes(joblib_path, onnx_path)
    report["artifact_sizes"] = sizes
    logger.info("Tamanhos: %s", sizes)

    if not args.skip_parity:
        try:
            t0 = time.perf_counter()
            onnx_wrapper = OnnxInferenceSessionWrapper(
                onnx_path,
                intra_op_num_threads=1,
                inter_op_num_threads=0,
                execution_mode="sequential",
                enable_optimizations=True,
            )
            load_on_ms = (time.perf_counter() - t0) * 1000.0
            report["onnx_load_time_ms"] = float(round(load_on_ms, 2))
        except Exception as exc:
            logger.error("Falhou carregar sessao ONNX p/ equivalencia: %s", exc)
            return 4
        df = _load_test_df(test_csv, int(args.samples))
        try:
            parity = _parity_evaluation(
                pipeline,
                onnx_wrapper,
                df,
                float(args.min_accuracy_match),
                float(args.max_mean_abs_diff_proba),
            )
        except Exception as exc:
            logger.error("Falhou avaliacao equivalencia: %s: %s", exc.__class__.__name__, exc)
            return 5
        report["parity"] = parity
        ok_all = bool(parity["constraints"]["match_ok"] and parity["constraints"]["proba_ok"])
        report["parity_ok"] = ok_all
        logger.info(
            "[PARIDADE] match=%.4f | mean_diff_proba=%.5f | max_diff_proba=%.5f | "
            "delta_acc=%.4f | delta_f1=%.4f | speedup_total=x%.2f | RESULTADO=%s",
            parity["accuracy_match_sklearn_vs_onnx"],
            parity["mean_abs_diff_probability"],
            parity["max_abs_diff_probability"],
            parity["delta_accuracy_onnx_minus_sklearn"],
            parity["delta_f1_macro_onnx_minus_sklearn"],
            parity["onnx_speedup_vs_sklearn_total"],
            "APROVADO" if ok_all else "REPROVADO",
        )
        if not ok_all:
            logger.error(
                "Paridade NAO atendeu os requisitos (min_match=%s, max_diff_proba=%s).",
                args.min_accuracy_match,
                args.max_mean_abs_diff_proba,
            )
    else:
        report["parity_ok"] = None
        logger.warning("--skip-parity: nao validamos equivalencia.")

    if args.report_path:
        report_path = Path(args.report_path).resolve()
    else:
        bench_results = ROOT / "benchmarks" / "results"
        bench_results.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        report_path = bench_results / f"onnx_parity_{ts}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info("Relatorio de paridade salvo em %s", report_path)

    return 0 if report.get("parity_ok", True) in (True, None) else 7


if __name__ == "__main__":
    raise SystemExit(main())
