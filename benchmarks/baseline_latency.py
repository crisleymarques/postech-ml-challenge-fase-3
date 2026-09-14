from __future__ import annotations

import sys
import os
import json
import time
import math
import argparse
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import numpy as np  # type: ignore
    HAS_NUMPY = True
except ImportError:  # pragma: no cover
    HAS_NUMPY = False

try:
    import httpx  # type: ignore
except ImportError as exc:  # pragma: no cover
    print("ERRO: httpx nao encontrado. Instale com: pip install httpx")
    sys.exit(2)


PAYLOADS_FILE = Path(__file__).resolve().parent / "payloads.json"


def load_payloads() -> Dict[str, str]:
    if PAYLOADS_FILE.exists():
        with open(PAYLOADS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "urgente": "Acute chest pain radiating to left arm with diaphoresis. ECG with ST elevation. Diagnosed acute myocardial infarction. Urgent angioplasty required.",
        "normal": "Patient with chronic gastritis, intermittent heartburn after meals. Endoscopy: mild pangastritis. Prescribed proton pump inhibitor and dietary changes.",
        "atencao": "Patient with 6-month history of migraine. Cranial MRI without contrast normal. Outpatient neurology follow-up scheduled.",
    }


def collect_env_info(target_url: str, workers: Optional[int], cpus: Optional[str]) -> Dict[str, any]:
    cpu_brand = platform.processor() or "unknown"
    cpu_count = os.cpu_count()

    mem_bytes = None
    try:
        if platform.system() == "Windows":
            out = subprocess.check_output(
                ["wmic", "OS", "get", "TotalVisibleMemorySize", "/value"],
                text=True,
            )
            for line in out.splitlines():
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    if k == "TotalVisibleMemorySize":
                        mem_bytes = int(v) * 1024
        else:
            with open("/proc/meminfo", "r") as mf:
                for ln in mf:
                    if ln.startswith("MemTotal:"):
                        kb = int(ln.split()[1])
                        mem_bytes = kb * 1024
    except Exception:
        mem_bytes = None

    return {
        "benchmark_time_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "node": platform.node(),
        "cpu_brand": cpu_brand,
        "cpu_count_logical": cpu_count,
        "memory_total_bytes": mem_bytes,
        "memory_total_gb": round(mem_bytes / (1024**3), 2) if mem_bytes else None,
        "target_url": target_url,
        "containerized": os.environ.get("CONTAINER") == "1" or Path("/.dockerenv").exists(),
        "configured_workers": workers,
        "configured_cpus": cpus,
    }


def percentile(sorted_values: List[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if HAS_NUMPY:
        return float(np.percentile(sorted_values, pct, method="linear"))
    k = (len(sorted_values) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def compute_stats(latencies_ms: List[float]) -> Dict[str, float]:
    if not latencies_ms:
        return {"count": 0}
    sorted_l = sorted(latencies_ms)
    n = len(sorted_l)
    mean_s = sum(sorted_l) / n
    variance = sum((x - mean_s) ** 2 for x in sorted_l) / n if n > 0 else 0.0
    std = math.sqrt(variance)
    return {
        "count": n,
        "min_ms": round(sorted_l[0], 3),
        "max_ms": round(sorted_l[-1], 3),
        "avg_ms": round(mean_s, 3),
        "std_ms": round(std, 3),
        "p50_ms": round(percentile(sorted_l, 50), 3),
        "p75_ms": round(percentile(sorted_l, 75), 3),
        "p90_ms": round(percentile(sorted_l, 90), 3),
        "p95_ms": round(percentile(sorted_l, 95), 3),
        "p99_ms": round(percentile(sorted_l, 99), 3),
    }


def run_requests(
    client,
    target_url: str,
    total_requests: int,
    payloads: Dict[str, str],
    concurrency: int = 1,
    return_probabilities: bool = True,
) -> Tuple[List[float], List[int], List[str], int, int]:
    latencies_ms: List[float] = []
    statuses: List[int] = []
    errors: List[str] = []
    success = 0
    failed = 0

    keys = list(payloads.keys())
    if not keys:
        raise ValueError("Sem payloads para enviar")

    if concurrency <= 1:
        for i in range(total_requests):
            key = keys[i % len(keys)]
            payload = {
                "text": payloads[key],
                "return_probabilities": return_probabilities,
                "request_id": f"bench-{i+1:06d}",
            }
            t0 = time.perf_counter()
            try:
                resp = client.post("/predict", json=payload, timeout=60.0)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                latencies_ms.append(elapsed_ms)
                statuses.append(resp.status_code)
                if 200 <= resp.status_code < 300:
                    success += 1
                else:
                    failed += 1
                    body = resp.text if hasattr(resp, "text") else str(resp)
                    errors.append(f"HTTP {resp.status_code}: {body[:200]}")
            except Exception as e:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                latencies_ms.append(elapsed_ms)
                failed += 1
                errors.append(f"EXC {type(e).__name__}: {str(e)[:200]}")
        return latencies_ms, statuses, errors, success, failed

    # concorrente
    import asyncio

    async def one_req(client_async, idx: int):
        key = keys[idx % len(keys)]
        payload = {
            "text": payloads[key],
            "return_probabilities": return_probabilities,
            "request_id": f"bench-{idx+1:06d}",
        }
        t0 = time.perf_counter()
        try:
            resp = await client_async.post("/predict", json=payload, timeout=60.0)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if 200 <= resp.status_code < 300:
                return ("ok", elapsed_ms, resp.status_code, None)
            return ("err", elapsed_ms, resp.status_code, resp.text[:200])
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return ("err", elapsed_ms, 0, f"{type(e).__name__}: {str(e)[:200]}")

    async def run_all():
        limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
        async with httpx.AsyncClient(base_url=target_url, limits=limits, timeout=120.0) as ac:
            sem = asyncio.Semaphore(concurrency)

            async def bounded(i):
                async with sem:
                    return await one_req(ac, i)

            tasks = [bounded(i) for i in range(total_requests)]
            return await asyncio.gather(*tasks)

    results = asyncio.run(run_all())
    for status_flag, elapsed_ms, code, err in results:
        latencies_ms.append(elapsed_ms)
        if code:
            statuses.append(code)
        if status_flag == "ok":
            success += 1
        else:
            failed += 1
            errors.append(err or "unknown error")

    return latencies_ms, statuses, errors, success, failed


def print_banner(title: str, color: str = "green"):
    width = 72
    colors = {
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "cyan": "\033[96m",
        "reset": "\033[0m",
    }
    sep = "=" * width
    c = colors.get(color, "")
    r = colors["reset"]
    print(f"\n{c}{sep}{r}")
    print(f"{c}{title.center(width)}{r}")
    print(f"{c}{sep}{r}")


def print_stats(name: str, stats: Dict[str, any]):
    print(f"\n  >> {name}:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"     {k:>12} : {v:9.3f} ms")
        else:
            print(f"     {k:>12} : {v}")


def main():
    parser = argparse.ArgumentParser(
        description="Baseline de latência da API de classificação de urgência médica",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", default="http://localhost:8000", help="URL base da API")
    parser.add_argument("--warmup", type=int, default=50, help="Número de requisições de warm-up")
    parser.add_argument("--requests", type=int, default=500, help="Número total de requisições do benchmark")
    parser.add_argument("--concurrency", type=int, default=1, help="Nível de concorrência (1=sequencial)")
    parser.add_argument("--workers", type=int, default=None, help="Workers Gunicorn/Uvicorn configurados (apenas documentação)")
    parser.add_argument("--cpus", type=str, default=None, help="CPUs alocadas (documentação, ex: 1.0 ou 0-1)")
    parser.add_argument("--no-probabilities", action="store_true", help="Não requisitar probabilidades por classe")
    parser.add_argument("--output", type=str, default=None, help="Arquivo JSON para salvar resultados")
    parser.add_argument("--skip-warmup", action="store_true", help="Pular warm-up")
    parser.add_argument("--timeout-health", type=int, default=120, help="Timeout máximo aguardando /health ficar healthy (segundos)")
    args = parser.parse_args()

    target_url = args.url.rstrip("/")

    print_banner("BASELINE DE LATÊNCIA - API CLASSIFICAÇÃO DE LAUDOS", "cyan")
    print(f"  Alvo              : {target_url}")
    print(f"  Warm-up requests  : {0 if args.skip_warmup else args.warmup}")
    print(f"  Benchmark requests: {args.requests}")
    print(f"  Concorrência      : {args.concurrency}")
    print(f"  Probabilidades    : {not args.no_probabilities}")
    print(f"  Timestamp (local) : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Health check com espera
    print_banner("1) Aguardando API saudável (GET /health)", "yellow")
    deadline = time.time() + args.timeout_health
    healthy = False
    health_payload: Optional[Dict] = None
    last_status = None
    with httpx.Client(base_url=target_url, timeout=10.0) as client:
        while time.time() < deadline:
            try:
                resp = client.get("/health")
                last_status = resp.status_code
                if resp.status_code == 200:
                    try:
                        health_payload = resp.json()
                        if health_payload.get("model_loaded"):
                            healthy = True
                            print(f"  OK: API healthy! status={health_payload.get('status')}, model_loaded=True")
                            break
                        else:
                            print(f"  INFO: /health 200 mas model_loaded=False. Aguardando...")
                    except Exception:
                        healthy = True
                        print(f"  OK: /health 200 OK (modelo carregado não detectado).")
                        break
            except Exception as e:
                last_status = str(e)
                print(f"  Aguardando API... {type(e).__name__}: {str(e)[:80]}")
            time.sleep(3)
        if not healthy:
            print(f"  ERRO: API não ficou healthy dentro de {args.timeout_health}s. Último status: {last_status}")
            sys.exit(1)

    payloads = load_payloads()
    print(f"\n  {len(payloads)} payloads carregados: {list(payloads.keys())}")

    env_info = collect_env_info(target_url, workers=args.workers, cpus=args.cpus)
    model_info: Dict[str, any] = {}
    if health_payload:
        model_info = {
            "status": health_payload.get("status"),
            "model_loaded": health_payload.get("model_loaded"),
            "model_name": health_payload.get("model_name"),
            "model_metadata": health_payload.get("model_metadata"),
            "app_version": health_payload.get("app_version"),
        }

    # ========= WARM-UP =========
    warmup_stats: Optional[Dict] = None
    if not args.skip_warmup and args.warmup > 0:
        print_banner("2) WARM-UP requisições", "yellow")
        with httpx.Client(base_url=target_url, timeout=120.0) as wclient:
            w_lat, w_statuses, w_errors, w_ok, w_fail = run_requests(
                wclient,
                target_url=target_url,
                total_requests=args.warmup,
                payloads=payloads,
                concurrency=1,
                return_probabilities=not args.no_probabilities,
            )
        warmup_stats = compute_stats(w_lat)
        warmup_stats["success"] = w_ok
        warmup_stats["failed"] = w_fail
        print_stats("Warm-up", warmup_stats)
        if w_fail:
            print(f"  WARN: {w_fail} erros durante warm-up. Primeiros: {w_errors[:3]}")

    # ========= BENCHMARK =========
    print_banner("3) BENCHMARK PRINCIPAL", "green")
    t_total_start = time.perf_counter()
    if args.concurrency <= 1:
        with httpx.Client(base_url=target_url, timeout=180.0) as client:
            b_lat, b_statuses, b_errors, b_ok, b_fail = run_requests(
                client,
                target_url=target_url,
                total_requests=args.requests,
                payloads=payloads,
                concurrency=args.concurrency,
                return_probabilities=not args.no_probabilities,
            )
    else:
        b_lat, b_statuses, b_errors, b_ok, b_fail = run_requests(
            None,
            target_url=target_url,
            total_requests=args.requests,
            payloads=payloads,
            concurrency=args.concurrency,
            return_probabilities=not args.no_probabilities,
        )
    t_total_s = time.perf_counter() - t_total_start
    benchmark_stats = compute_stats(b_lat)
    benchmark_stats["success"] = b_ok
    benchmark_stats["failed"] = b_fail
    benchmark_stats["total_duration_s"] = round(t_total_s, 3)
    benchmark_stats["throughput_rps"] = round(b_ok / t_total_s, 3) if t_total_s > 0 and b_ok else 0.0
    benchmark_stats["success_rate_pct"] = round((b_ok / (b_ok + b_fail)) * 100, 2) if (b_ok + b_fail) > 0 else 0.0
    print_stats("Benchmark", benchmark_stats)
    if b_fail:
        print(f"\n  WARN: {b_fail} erros. Primeiros 5 erros:")
        for i, e in enumerate(b_errors[:5], 1):
            print(f"    [{i}] {e}")

    print_banner("4) RESULTADOS FINAIS", "green")
    print(f"  Total de requisições: {args.requests}")
    print(f"  Taxa de sucesso     : {benchmark_stats['success_rate_pct']}% ({b_ok}/{b_ok+b_fail})")
    print(f"  Throughput (RPS)    : {benchmark_stats['throughput_rps']} req/s")
    print(f"  Duração total       : {t_total_s:.2f} segundos")
    print(f"\n  Latência (ms):")
    for k in ["avg_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms", "min_ms", "max_ms"]:
        if k in benchmark_stats:
            print(f"    {k:>7}: {benchmark_stats[k]:>9.3f} ms")

    results: Dict[str, any] = {
        "benchmark_name": "baseline_latency_predict",
        "version": "1.0",
        "environment": env_info,
        "api_health_snapshot": model_info,
        "parameters": {
            "warmup_requests": 0 if args.skip_warmup else args.warmup,
            "benchmark_requests": args.requests,
            "concurrency": args.concurrency,
            "return_probabilities": not args.no_probabilities,
            "n_payloads": len(payloads),
            "payload_keys": list(payloads.keys()),
        },
        "warmup": warmup_stats,
        "benchmark": benchmark_stats,
        "errors_sample": (b_errors[:20] if b_fail else []),
    }

    output_path = None
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(__file__).resolve().parent / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"baseline_{ts}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  ✅ Resultados salvos em: {output_path}")

    # print summary table (markdown)
    print_banner("5) RESUMO MARKDOWN (copiar para documentação)", "yellow")
    print("| Métrica | Valor |")
    print("|:--------|------:|")
    for k in ["count", "success_rate_pct", "throughput_rps", "avg_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms", "min_ms", "max_ms"]:
        v = benchmark_stats.get(k, "-")
        if isinstance(v, float):
            print(f"| {k} | {v:.3f} |")
        else:
            print(f"| {k} | {v} |")

    print(f"\nArquivo JSON salvo: {output_path}")
    return 0 if b_fail == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
