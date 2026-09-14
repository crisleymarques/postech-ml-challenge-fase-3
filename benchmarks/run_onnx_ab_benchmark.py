from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

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


def _resolve_test_csv(test_csv: str) -> Path:
    user_provided = Path(test_csv).resolve()
    if user_provided.exists():
        return user_provided
    for cand in TEST_CSV_CANDIDATES:
        if cand.exists():
            return cand.resolve()
    raise FileNotFoundError(
        "Nenhum CSV de teste encontrado. "
        f"Candidatos tentados: {[str(x) for x in [user_provided] + TEST_CSV_CANDIDATES]}. "
    )


def _ensure_urgency_label(df):
    import pandas as pd

    cols = set(df.columns)
    if LABEL_COLUMN in cols and df[LABEL_COLUMN].notna().any():
        df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors="coerce").astype(int)
        return df
    if TARGET_COLUMN in cols:
        df[LABEL_COLUMN] = df[TARGET_COLUMN].map(CONDITION_TO_URGENCY)
        df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors="coerce").astype(int)
        return df
    return df

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("onnx_ab_benchmark")


def _collect_env_info() -> Dict[str, Any]:
    try:
        import cpuinfo  # type: ignore

        cpu_brand = cpuinfo.get_cpu_info().get("brand_raw", platform.processor())
    except Exception:
        cpu_brand = platform.processor() or "unknown"
    uname = platform.uname()
    info: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform_system": uname.system,
        "platform_release": uname.release,
        "platform_version": uname.version,
        "machine_arch": uname.machine,
        "cpu_brand": cpu_brand,
        "cpu_count_logical": 1 if False else None,
    }
    try:
        import os

        info["cpu_count_logical"] = os.cpu_count() or 1
    except Exception:
        info["cpu_count_logical"] = None
    try:
        import psutil  # type: ignore

        mem = psutil.virtual_memory()
        info["ram_total_gb"] = float(round(mem.total / (1024**3), 2))
        info["ram_available_gb"] = float(round(mem.available / (1024**3), 2))
    except Exception:
        pass
    for pkg_name in ["sklearn", "onnxruntime", "skl2onnx", "onnx", "numpy", "pandas"]:
        try:
            m = __import__(pkg_name)
            info[f"pkg_{pkg_name}_version"] = getattr(m, "__version__", "unknown")
        except Exception:
            info[f"pkg_{pkg_name}_version"] = None
    return info


def _load_texts(args: argparse.Namespace) -> List[str]:
    import pandas as pd

    setup_dirs()
    if args.payloads_json and Path(args.payloads_json).exists():
        with open(args.payloads_json, "r", encoding="utf-8") as f:
            payloads = json.load(f)
        if isinstance(payloads, dict) and "cases" in payloads:
            return [str(c.get("text") or c.get("content")) for c in payloads["cases"] if c.get("text") or c.get("content")]
        if isinstance(payloads, list):
            return [str(x) if not isinstance(x, dict) else str(x.get("text", json.dumps(x))) for x in payloads]
    csv_path = _resolve_test_csv(args.test_csv)
    df = pd.read_csv(csv_path)
    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"CSV de benchmark nao tem coluna {TEXT_COLUMN!r}")
    df = _ensure_urgency_label(df)
    df = df.dropna(subset=[TEXT_COLUMN]).reset_index(drop=True)
    df = df.head(int(args.samples))
    logger.info("Textos benchmark carregados de %s (n=%d)", csv_path, len(df))
    return df[TEXT_COLUMN].astype(str).tolist()


def _load_backend(args: argparse.Namespace, backend: str):
    joblib_path = Path(args.joblib_path)
    onnx_path = Path(args.onnx_path)
    if backend == "sklearn":
        import joblib

        if not joblib_path.exists():
            raise FileNotFoundError(f"joblib nao existe: {joblib_path}")
        art = joblib.load(joblib_path)
        return art.get("model") if isinstance(art, dict) and "model" in art else art
    if backend == "onnx":
        from sources.model_onnx import OnnxInferenceSessionWrapper

        if not onnx_path.exists():
            raise FileNotFoundError(
                f"onnx nao existe: {onnx_path}. Gere-o antes usando scripts/optimize_model.py."
            )
        return OnnxInferenceSessionWrapper(
            onnx_path,
            intra_op_num_threads=int(args.onnx_intra_threads),
            inter_op_num_threads=int(args.onnx_inter_threads),
            execution_mode=str(args.onnx_execution_mode),
            enable_optimizations=not args.onnx_no_optimizations,
        )
    raise ValueError(f"Backend desconhecido: {backend}")


def _percentiles(ms_list: np.ndarray, pcts: List[int]) -> Dict[str, float]:
    arr = np.asarray(ms_list, dtype=np.float64)
    out: Dict[str, float] = {}
    for p in pcts:
        out[f"p{p}"] = float(round(float(np.percentile(arr, p)), 6))
    return out


def _bench_one(
    backend_name: str,
    pipeline,
    texts: List[str],
    warmup: int,
    n: int,
    proba: bool,
    seed: int = 42,
) -> Dict[str, Any]:
    import pandas as pd

    rng = np.random.default_rng(int(seed))
    idx_warm = rng.integers(0, len(texts), size=max(1, int(warmup)))
    idx_run = rng.integers(0, len(texts), size=max(1, int(n)))

    def _run(idx: np.ndarray, measure: bool):
        lat_ms: List[float] = []
        for j in idx.tolist():
            tx = texts[int(j)]
            if backend_name == "sklearn":
                series = pd.Series([tx])
                t0 = time.perf_counter()
                _ = pipeline.predict(series)
                if proba:
                    _ = pipeline.predict_proba(series)
                dt = time.perf_counter() - t0
            else:
                t0 = time.perf_counter()
                _ = pipeline.predict([tx])
                if proba:
                    _ = pipeline.predict_proba([tx])
                dt = time.perf_counter() - t0
            if measure:
                lat_ms.append(float(dt) * 1000.0)
        return lat_ms

    logger.info("[%s] warmup (%d itens)...", backend_name, int(warmup))
    _run(idx_warm, measure=False)
    logger.info("[%s] benchmark (%d itens, proba=%s)...", backend_name, int(n), proba)
    t_wall_0 = time.perf_counter()
    lat_ms = _run(idx_run, measure=True)
    wall_time_s = float(time.perf_counter() - t_wall_0)
    arr = np.asarray(lat_ms, dtype=np.float64)
    stats = {
        "backend": backend_name,
        "n_warmup": int(warmup),
        "n_requests": int(arr.shape[0]),
        "probabilities_enabled": bool(proba),
        "wall_time_seconds": float(round(wall_time_s, 6)),
        "throughput_rps": float(round(float(arr.shape[0]) / wall_time_s, 4)) if wall_time_s > 0 else 0.0,
        "latency_ms": {
            "min": float(round(float(np.min(arr)), 6)),
            "max": float(round(float(np.max(arr)), 6)),
            "mean": float(round(float(np.mean(arr)), 6)),
            "median": float(round(float(np.median(arr)), 6)),
            "std": float(round(float(np.std(arr)), 6)),
            **_percentiles(arr, [1, 5, 25, 50, 75, 90, 95, 99]),
        },
    }
    return stats


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "[Atividade8-Otimizacao] Benchmark A/B direto Python entre sklearn (baseline) "
            "e ONNX Runtime. Mesma metodologia baseline: warmup + N requests, mede "
            "media/mediana/P90/P95/P99 + throughput."
        )
    )
    p.add_argument("--joblib-path", type=str, default=str(MODELS_DIR / "urgency_classifier.joblib"))
    p.add_argument("--onnx-path", type=str, default=str(MODELS_DIR / "urgency_classifier.onnx"))
    p.add_argument("--test-csv", type=str, default=str(TEST_CSV_CANDIDATES[0]),
                   help=f"Ordem fallback CSV: {[str(p) for p in TEST_CSV_CANDIDATES]}")
    p.add_argument("--payloads-json", type=str, default=str(ROOT / "benchmarks" / "payloads.json"))
    p.add_argument("--samples", type=int, default=1000,
                   help="Quando nao tiver payloads.json, usa N linhas do CSV teste.")
    p.add_argument("--warmup", type=int, default=500, help="Iterações warmup descartadas (default 500).")
    p.add_argument("--requests", type=int, default=5000, help="Iterações medidas (default 5000).")
    p.add_argument("--no-probabilities", action="store_true", help="Desativa medida predict_proba (por default ON).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--onnx-intra-threads", type=int, default=1)
    p.add_argument("--onnx-inter-threads", type=int, default=0)
    p.add_argument("--onnx-execution-mode", type=str, default="sequential", choices=["sequential", "parallel"])
    p.add_argument("--onnx-no-optimizations", action="store_true")
    p.add_argument("--output", type=str, default=None,
                   help="Arquivo JSON de saida (default: benchmarks/results/ab_onnx_<timestamp>.json)")
    p.add_argument("--backend", type=str, default="both", choices=["both", "sklearn", "onnx"])
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    missing: List[str] = []
    try:
        import pandas as _  # noqa: F401
    except ImportError:
        missing.append("pandas")
    try:
        import sklearn  # noqa: F401
    except ImportError:
        missing.append("scikit-learn")
    if args.backend in {"both", "onnx"}:
        try:
            from sources.model_onnx import OnnxInferenceSessionWrapper, is_onnx_available
            if not is_onnx_available():
                missing.append("onnxruntime")
        except Exception as exc:
            missing.append(f"sources.model_onnx disponivel ({exc})")
    if missing:
        logger.error("Dependencias faltando: %s", missing)
        return 2

    texts = _load_texts(args)
    logger.info("Textos carregados p/ benchmark: %d. Usando warmup=%d requests=%d.",
                len(texts), args.warmup, args.requests)
    if len(texts) < 1:
        logger.error("Nenhum texto carregado p/ benchmark.")
        return 3
    env = _collect_env_info()

    results: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "methodology": {
            "warmup_discarded": int(args.warmup),
            "measured_requests": int(args.requests),
            "concurrency_synchronous": 1,
            "probabilities_enabled": not bool(args.no_probabilities),
            "in_process_benchmark": True,
            "seed": int(args.seed),
            "notes": (
                "Benchmark direto no Python, sem overhead HTTP/JSON/stack FastAPI. "
                "Mede latencia real de inferencia sklearn vs onnx com mesmos textos, "
                "sequencial, 1 worker."
            ),
        },
        "environment": env,
        "texts_source": (
            Path(args.payloads_json).name
            if (args.payloads_json and Path(args.payloads_json).exists())
            else Path(args.test_csv).name
        ),
    }
    runs: List[Dict[str, Any]] = []
    if args.backend in {"sklearn", "both"}:
        sk = _load_backend(args, "sklearn")
        runs.append(_bench_one("sklearn", sk, texts, args.warmup, args.requests, not args.no_probabilities, args.seed))
    if args.backend in {"onnx", "both"}:
        on = _load_backend(args, "onnx")
        runs.append(_bench_one("onnx", on, texts, args.warmup, args.requests, not args.no_probabilities, args.seed))

    if len(runs) == 2:
        base = runs[0]["latency_ms"]["mean"]
        opt = runs[1]["latency_ms"]["mean"]
        base_rps = runs[0]["throughput_rps"]
        opt_rps = runs[1]["throughput_rps"]
        base_p99 = runs[0]["latency_ms"]["p99"]
        opt_p99 = runs[1]["latency_ms"]["p99"]
        results["comparison"] = {
            "latency_mean_reduction_pct": float(round(((base - opt) / base) * 100.0, 2)) if base > 0 else None,
            "throughput_increase_pct": float(round(((opt_rps - base_rps) / base_rps) * 100.0, 2)) if base_rps > 0 else None,
            "p99_reduction_pct": float(round(((base_p99 - opt_p99) / base_p99) * 100.0, 2)) if base_p99 > 0 else None,
        }
    results["runs"] = runs
    if args.output:
        out_path = Path(args.output).resolve()
    else:
        out_dir = ROOT / "benchmarks" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        out_path = out_dir / f"ab_onnx_benchmark_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    logger.info("Benchmark salvo em %s", out_path)

    print("\n" + "=" * 88)
    print("RESULTADO A/B INFERENCIA (ms, 1 worker, sequencial)")
    print("=" * 88)
    header = ["backend", "mean", "p50", "p90", "p95", "p99", "RPS", "solicitações"]
    fmt = "{:<10}{:>12}{:>12}{:>12}{:>12}{:>12}{:>14}{:>16}"
    print(fmt.format(*header))
    for r in runs:
        lat = r["latency_ms"]
        print(fmt.format(
            r["backend"],
            f"{lat['mean']:.3f}",
            f"{lat['p50']:.3f}",
            f"{lat['p90']:.3f}",
            f"{lat['p95']:.3f}",
            f"{lat['p99']:.3f}",
            f"{r['throughput_rps']:.2f}",
            f"{r['n_requests']}",
        ))
    if results.get("comparison"):
        c = results["comparison"]
        print(f"\nGanhos ONNX vs sklearn:")
        print(f"  ↓ Latencia média : {c.get('latency_mean_reduction_pct')} %")
        print(f"  ↓ Latencia P99   : {c.get('p99_reduction_pct')} %")
        print(f"  ↑ Throughput     : {c.get('throughput_increase_pct')} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
