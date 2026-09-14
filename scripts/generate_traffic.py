from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import List

try:
    import httpx
except ImportError as e:  # pragma: no cover
    print(
        f"[ERRO] Dependencia ausente: {e.name}. Instale: pip install httpx",
        file=sys.stderr,
    )
    sys.exit(2)


# 3 classes reais — garantem ao longo do tempo distribuicao interessante no dashboard
SAMPLE_TEXTS: List[str] = [
    # Urgente (câncer / cardiovascular)
    (
        "68-year-old female with three-month history of progressive dysphagia, 8kg "
        "unintentional weight loss, and iron-deficiency anemia. Upper endoscopy reveals "
        "a circumferential ulcerative mass 35 cm from incisors. Biopsy shows "
        "adenocarcinoma of the esophagus. Staging CT confirms perigastric lymphadenopathy."
    ),
    (
        "55 year old male with three hours of crushing retrosternal chest pain radiating "
        "to the left jaw and arm. Diaphoresis, nausea, dyspnea on minimal exertion. "
        "Past medical history: systemic arterial hypertension, dyslipidemia, type 2 "
        "diabetes, current smoker. 12-lead ECG shows ST-segment elevation of 3mm in "
        "leads V1 to V4. High-sensitivity troponin I is 3 times the upper reference "
        "limit. Diagnosis: acute anterior ST-elevation myocardial infarction. "
        "Immediate primary percutaneous coronary intervention indicated."
    ),
    (
        "A 72-year-old patient known for persistent atrial fibrillation on oral "
        "anticoagulation experienced abrupt onset right-sided hemiplegia, global "
        "aphasia, and gaze deviation two hours prior to admission. NIHSS score 18. "
        "Non-contrast head CT reveals no hemorrhage; ASPECTS 7. Large vessel occlusion "
        "of the left middle cerebral artery confirmed on CTA. Thrombectomy candidates."
    ),
    # Atenção (sistema nervoso / patologias gerais)
    (
        "40-year-old female presenting with six months of relapsing-remitting visual "
        "blurring in the left eye associated with Lhermitte sign and progressive "
        "ascending paresthesia. MRI of the cervical spine and brain shows multiple "
        "periventricular demyelinating lesions with gadolinium enhancement. CSF "
        "oligoclonal bands positive. Diagnosis: multiple sclerosis relapse."
    ),
    (
        "A 29-year-old male reports recurrent attacks of throbbing unilateral "
        "headache, photophobia, phonophobia preceded by visual aura lasting 20 minutes. "
        "Family history positive for migraine. Frequency has increased to four episodes "
        "per month over the last trimester, impairing work productivity. Neurological "
        "examination unremarkable. MRI brain: no pathological findings. "
        "Diagnostic impression: migraine with aura, chronicising pattern."
    ),
    (
        "75-year-old female with history of idiopathic Parkinson disease for ten years "
        "presents with increasing gait instability, four episodes of backward falls in "
        "the last two months and new onset visual hallucinations in the evening. Current "
        "pharmacotherapy includes levodopa/carbidopa, pramipexole, rasagiline and "
        "quetiapine. Examination: 3 Hz resting tremor bilaterally, cogwheel rigidity, "
        "positive pull test, bradyphrenia. Assessment: advanced Parkinson disease "
        "requiring therapeutic adjustment and fall prevention protocol."
    ),
    # Normal (aparelho digestivo)
    (
        "32-year-old male with a one-week history of burning epigastric pain worsening "
        "at night and two hours after meals, occasional acid regurgitation, no "
        "dysphagia, no melena, no weight loss. Social history: occasional alcohol "
        "intake, uses NSAIDs twice weekly for training-related soreness. Physical exam "
        "normal except mild epigastric tenderness. Complete blood count and basic "
        "metabolic panel within reference ranges. H. pylori stool antigen negative. "
        "Clinical diagnosis: dyspepsia secondary to NSAID use / functional "
        "gastrointestinal disorder."
    ),
    (
        "A 24-year-old female reports three days of self-limited non-bloody watery "
        "diarrhea (five to six loose stools daily), low-grade fever 37.8 C, mild "
        "crampy abdominal pain and nausea without vomiting after eating at a street "
        "food vendor. No recent travel, no antibiotic use. Vitals stable, no signs of "
        "dehydration. Abdomen soft, non-peritoneal. Stool culture not indicated per "
        "guidelines. Assessment: acute uncomplicated infectious gastroenteritis."
    ),
    (
        "45-year-old overweight (BMI 30) patient complaining of three months of "
        "intermittent right upper quadrant postprandial colicky pain lasting 30 to 60 "
        "minutes after fatty meals, occasionally radiating to the right scapula with "
        "nausea but no vomiting or jaundice. Labs: AST, ALT, GGT, alkaline "
        "phosphatase, total bilirubin within normal limits; lipase normal. Abdominal "
        "ultrasound shows multiple mobile hyperechoic gallstones within a "
        "thin-walled gallbladder, no pericholecystic fluid. Diagnosis: symptomatic "
        "uncomplicated cholelithiasis."
    ),
]

INVALID_SHORT_TEXT = "short ok"  # 7 chars < min_length 10 (force 400 invalid_text)


@dataclass
class Stats:
    started_at: float = field(default_factory=time.time)
    ok: int = 0
    err_4xx: int = 0
    err_5xx: int = 0
    err_connect: int = 0
    latencies_ms: List[float] = field(default_factory=list)
    status_codes: Counter = field(default_factory=Counter)

    def record(self, status: int, latency_ms: float) -> None:
        self.latencies_ms.append(latency_ms)
        self.status_codes[status] += 1
        if 200 <= status < 400:
            self.ok += 1
        elif 400 <= status < 500:
            self.err_4xx += 1
        elif status >= 500:
            self.err_5xx += 1

    def summary(self) -> str:
        elapsed = time.time() - self.started_at
        total = self.ok + self.err_4xx + self.err_5xx
        rps = total / elapsed if elapsed > 0 else 0.0
        p50 = p95 = p99 = 0.0
        if self.latencies_ms:
            p50 = statistics.quantiles(self.latencies_ms, n=100, method="inclusive")[49]
            p95 = statistics.quantiles(self.latencies_ms, n=100, method="inclusive")[94]
            p99 = statistics.quantiles(self.latencies_ms, n=100, method="inclusive")[98]
        return (
            f"elapsed={elapsed:5.1f}s | total={total:4d} | RPS~{rps:5.2f} | "
            f"2xx={self.ok:4d} | 4xx={self.err_4xx:3d} | 5xx={self.err_5xx:2d} | "
            f"P50={p50:6.2f}ms P95={p95:6.2f}ms P99={p99:6.2f}ms"
        )


def _payload_single(text: str, include_prob: bool = True) -> dict:
    return {"text": text, "return_probabilities": include_prob}


def _payload_batch(n: int, errors_ratio: float) -> dict:
    items = []
    for i in range(n):
        if random.random() < errors_ratio:
            items.append(_payload_single(INVALID_SHORT_TEXT))
        else:
            items.append(_payload_single(random.choice(SAMPLE_TEXTS), include_prob=bool(i % 2)))
    return {"items": items}


async def single_request(client: httpx.AsyncClient, base_url: str, stats: Stats, error_ratio: float) -> None:
    if random.random() < error_ratio:
        payload = _payload_single(INVALID_SHORT_TEXT, include_prob=False)
    else:
        payload = _payload_single(random.choice(SAMPLE_TEXTS), include_prob=random.random() > 0.5)
    try:
        t0 = time.perf_counter()
        r = await client.post(f"{base_url}/predict", json=payload, timeout=15.0)
        stats.record(r.status_code, (time.perf_counter() - t0) * 1000.0)
    except Exception:
        stats.err_connect += 1


async def batch_request(client: httpx.AsyncClient, base_url: str, stats: Stats, error_ratio: float) -> None:
    n = random.choice([3, 5, 7, 10])
    payload = _payload_batch(n, errors_ratio=error_ratio)
    try:
        t0 = time.perf_counter()
        r = await client.post(f"{base_url}/predict/batch", json=payload, timeout=30.0)
        stats.record(r.status_code, (time.perf_counter() - t0) * 1000.0)
    except Exception:
        stats.err_connect += 1


async def wait_api_ready(base_url: str, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    last_msg = ""
    async with httpx.AsyncClient(timeout=10.0) as c:
        while time.time() < deadline:
            try:
                r = await c.get(f"{base_url}/health")
                if r.status_code == 200:
                    j = r.json()
                    if j.get("status") == "healthy":
                        print("[INFO] API healthy e modelo carregado. Iniciando tráfego.")
                        return
                    if last_msg != "degraded":
                        print("[INFO] API /health OK mas modelo ainda não carregado (degraded). Aguardando...")
                        last_msg = "degraded"
            except Exception as e:
                msg = f"sem conexao: {e.__class__.__name__}"
                if msg != last_msg:
                    print(f"[INFO] Aguardando API ficar disponível — {msg}")
                    last_msg = msg
            await asyncio.sleep(2.0)
    raise RuntimeError("API não ficou saudável dentro do timeout.")


async def traffic_loop(base_url: str, duration: int, rps: float, batch_ratio: float, error_ratio: float) -> None:
    await wait_api_ready(base_url)
    stats = Stats()
    sleep_for = 1.0 / max(0.01, rps)
    deadline = time.time() + duration
    last_print = 0.0

    # Um healthcheck antes para atualizar graficos
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            while time.time() < deadline:
                loop_t0 = time.perf_counter()
                if random.random() < batch_ratio:
                    await batch_request(client, base_url, stats, error_ratio)
                else:
                    await single_request(client, base_url, stats, error_ratio)

                if time.time() - last_print >= 10.0:
                    last_print = time.time()
                    print(f"[STATS] {stats.summary()}")

                remaining = sleep_for - (time.perf_counter() - loop_t0)
                if remaining > 0:
                    await asyncio.sleep(remaining)
    finally:
        print(f"[FINAL] {stats.summary()}")
        print(f"[FINAL] Distribuição status: {json.dumps(dict(sorted(stats.status_codes.items())), ensure_ascii=False)}")
        if stats.err_connect:
            print(f"[AVISO] {stats.err_connect} erros de conexão/timeout durante a geração.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gerador de tráfego sintético para a API de urgência médica (Popula painéis Prometheus/Grafana).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", default="http://localhost:8000", help="URL base da API (sem barra final).")
    parser.add_argument("--duration", type=int, default=600, help="Duração total do tráfego (segundos).")
    parser.add_argument("--rps", type=float, default=1.0, help="Taxa alvo de requisições por segundo.")
    parser.add_argument("--batch-ratio", type=float, default=0.2, help="Fração de requests que serão POST /predict/batch (0 a 1).")
    parser.add_argument("--error-ratio", type=float, default=0.1, help="Fração de requests intencionalmente inválidos (400).")
    args = parser.parse_args()

    if not 0.0 <= args.batch_ratio <= 1.0:
        parser.error("--batch-ratio deve estar entre 0 e 1.")
    if not 0.0 <= args.error_ratio <= 1.0:
        parser.error("--error-ratio deve estar entre 0 e 1.")
    if args.rps <= 0:
        parser.error("--rps deve ser > 0.")

    url = args.url.rstrip("/")
    print(
        f"[START] Gerador de tráfego: url={url} duration={args.duration}s "
        f"rps={args.rps} batch_ratio={args.batch_ratio} error_ratio={args.error_ratio}"
    )
    try:
        asyncio.run(traffic_loop(url, args.duration, args.rps, args.batch_ratio, args.error_ratio))
    except KeyboardInterrupt:
        print("\n[CANCEL] Usuário interrompeu.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
