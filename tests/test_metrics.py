from __future__ import annotations

import sys
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402
from app.services.model_service import (  # noqa: E402
    get_model_service,
)
from app.metrics import (  # noqa: E402
    observe_http_request,
    observe_inference_prediction,
    observe_inference_error,
    update_model_loaded_state,
    render_metrics,
    get_prometheus_content_type,
)


# ---------------------------------------------------------------------------
# Helpers robustos: parse exposition format text (100% confiavel)
# ---------------------------------------------------------------------------

_METRIC_LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?:\s+(?P<ts>\d+))?$"
)


def _scrape_text() -> str:
    return render_metrics().decode("utf-8", errors="replace")


def _parse_snapshot(text: Optional[str] = None) -> Dict[Tuple, float]:
    """Converte scrape text para { (metric_name, frozenset_sorted_label_tuples) : float_value }."""
    snap: Dict[Tuple, float] = {}
    if text is None:
        text = _scrape_text()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        m = _METRIC_LINE_RE.match(line)
        if not m:
            continue
        name = m.group("name")
        value = float(m.group("value"))
        label_part = m.group("labels") or ""
        labels: List[Tuple[str, str]] = []
        if label_part:
            # parse label_key="label_val", separado por virgulas
            # Regex simples para capturar pares
            for lm in re.finditer(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"', label_part):
                labels.append((lm.group(1), lm.group(2)))
        key = (name, tuple(sorted(labels)))
        snap[key] = snap.get(key, 0.0) + value
    return snap


def _delta_snaps(
    snap_before: Dict[Tuple, float],
    snap_after: Dict[Tuple, float],
    metric_name_prefix: str,
    label_filter: Optional[Dict[str, str]] = None,
) -> float:
    """Soma deltas para todas as series de metric_name_prefix, opcionalmente com filtro de labels."""
    total = 0.0
    for k, v in snap_after.items():
        name, lbls = k
        if not name.startswith(metric_name_prefix):
            continue
        if label_filter:
            lbl_d = dict(lbls)
            if any(lbl_d.get(lk) != lv for lk, lv in label_filter.items()):
                continue
        total += v
    for k, v in snap_before.items():
        name, lbls = k
        if not name.startswith(metric_name_prefix):
            continue
        if label_filter:
            lbl_d = dict(lbls)
            if any(lbl_d.get(lk) != lv for lk, lv in label_filter.items()):
                continue
        total -= v
    return max(0.0, total)


def _any_series(snap: Dict[Tuple, float], metric_name_prefix: str, label_filter: Dict[str, str]) -> bool:
    for k in snap:
        name, lbls = k
        if not name.startswith(metric_name_prefix):
            continue
        lbl_d = dict(lbls)
        if all(lbl_d.get(lk) == lv for lk, lv in label_filter.items()):
            return True
    return False


# ---------------------------------------------------------------------------
# Testes do ENDPOINT /metrics
# ---------------------------------------------------------------------------


class TestMetricsEndpoint:
    def test_metrics_endpoint_status_200(self):
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.get("/metrics")
        assert r.status_code == 200

    def test_metrics_endpoint_content_type(self):
        ct_expected = get_prometheus_content_type()
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.get("/metrics")
        actual = r.headers.get("content-type", "")
        assert actual == ct_expected or actual.startswith("text/plain")

    def test_metrics_endpoint_body_contains_all_help_type(self):
        with TestClient(app, raise_server_exceptions=False) as client:
            body = client.get("/metrics").content.decode("utf-8")
        required_prefixes = [
            "# HELP http_requests_total",
            "# TYPE http_requests_total counter",
            "# HELP http_request_duration_seconds",
            "# TYPE http_request_duration_seconds histogram",
            "# HELP http_errors_total",
            "# TYPE http_errors_total counter",
            "# HELP inference_predictions_total",
            "# TYPE inference_predictions_total counter",
            "# HELP inference_latency_seconds",
            "# TYPE inference_latency_seconds histogram",
            "# HELP inference_errors_total",
            "# TYPE inference_errors_total counter",
            "# HELP inference_items_processed_total",
            "# TYPE inference_items_processed_total counter",
            "# HELP model_loaded",
            "# TYPE model_loaded gauge",
            "# HELP model_metadata_info",
            "# TYPE model_metadata_info gauge",
        ]
        for p in required_prefixes:
            assert p in body, f"Prefixo ausente no scrape: {p}"

    def test_metrics_endpoint_render_bytes(self):
        payload = render_metrics()
        assert isinstance(payload, bytes)
        assert len(payload) > 0


# ---------------------------------------------------------------------------
# Testes de HTTP OBSERVABILITY
# ---------------------------------------------------------------------------


class TestHttpObservability:
    def test_http_request_counter_increments(self):
        snap0 = _parse_snapshot()
        observe_http_request("GET", "/health", 200, 0.001)
        observe_http_request("GET", "/health", 200, 0.002)
        snap1 = _parse_snapshot()
        delta = _delta_snaps(snap0, snap1, "http_requests_total")
        assert delta >= 2.0, f"Esperado +2.0, delta={delta}"

    def test_http_errors_counter_4xx(self):
        snap0 = _parse_snapshot()
        observe_http_request("POST", "/predict", 422, 0.005)
        observe_http_request("POST", "/predict/batch", 400, 0.006)
        snap1 = _parse_snapshot()
        delta_any = _delta_snaps(snap0, snap1, "http_errors_total")
        delta_4xx = _delta_snaps(snap0, snap1, "http_errors_total", label_filter={"status_family": "4xx"})
        assert delta_any >= 2.0, f"Esperado +2.0 erros HTTP, delta_any={delta_any}"
        assert delta_4xx >= 2.0, f"Esperado +2.0 4xx, delta_4xx={delta_4xx}"

    def test_http_errors_counter_5xx(self):
        snap0 = _parse_snapshot()
        observe_http_request("GET", "/health", 503, 0.008)
        observe_http_request("POST", "/predict", 500, 0.008)
        snap1 = _parse_snapshot()
        delta_5xx = _delta_snaps(snap0, snap1, "http_errors_total", label_filter={"status_family": "5xx"})
        assert delta_5xx >= 2.0, f"Esperado +2.0 5xx, delta_5xx={delta_5xx}"

    def test_http_duration_histogram_count_plus_two(self):
        snap0 = _parse_snapshot()
        observe_http_request("POST", "/predict", 200, 0.015)
        observe_http_request("GET", "/health", 200, 0.0008)
        snap1 = _parse_snapshot()
        # Bucket _count é o nome com sufixo "_count" no histograma
        delta = _delta_snaps(snap0, snap1, "http_request_duration_seconds_count")
        assert delta >= 2.0, f"Esperado +2 observacoes histograma, delta={delta}"

    def test_middleware_records_3_requests(self):
        with TestClient(app, raise_server_exceptions=False) as client:
            snap0 = _parse_snapshot()
            client.get("/health")
            client.get("/health")
            client.post("/predict", json={})  # 422 validation
            snap1 = _parse_snapshot()
        d_req = _delta_snaps(snap0, snap1, "http_requests_total")
        d_4xx = _delta_snaps(snap0, snap1, "http_errors_total", {"status_family": "4xx"})
        assert d_req >= 3.0, f"middleware nao registou os 3 requests: delta={d_req}"
        assert d_4xx >= 1.0, f"middleware nao registou o 422: delta_4xx={d_4xx}"


# ---------------------------------------------------------------------------
# Testes de INFERENCE OBSERVABILITY
# ---------------------------------------------------------------------------


class TestInferenceObservability:
    SAMPLE_VALID = (
        "55 year old male with three days chest pain radiating left arm, dyspnea "
        "on exertion, history of HTN. ECG ST depression anterior leads."
    )
    SAMPLE_INVALID_SHORT = "ok"  # < min_length 10

    def test_inference_predictions_counter_plus_3(self):
        snap0 = _parse_snapshot()
        observe_inference_prediction("single", "normal", "logistic_regression", 0.001, 1)
        observe_inference_prediction("single", "atencao", "logistic_regression", 0.002, 1)
        observe_inference_prediction("batch", "urgente", "logistic_regression", 0.0015, 1)
        snap1 = _parse_snapshot()
        delta = _delta_snaps(snap0, snap1, "inference_predictions_total")
        assert delta >= 3.0, f"Esperado +3.0 predicoes, delta={delta}"

    def test_inference_items_processed_batch_plus_8(self):
        snap0 = _parse_snapshot()
        observe_inference_prediction("single", "atencao", "lr", 0.001, 1)
        observe_inference_prediction("batch", "urgente", "lr", 0.002, 7)
        snap1 = _parse_snapshot()
        delta = _delta_snaps(snap0, snap1, "inference_items_processed_total")
        assert delta >= 8.0, f"Esperado 1+7=8 itens, delta={delta}"

    def test_inference_error_types_plus_3(self):
        snap0 = _parse_snapshot()
        observe_inference_error("single", "model_not_loaded")
        observe_inference_error("single", "invalid_text")
        observe_inference_error("batch", "batch_too_large")
        snap1 = _parse_snapshot()
        delta = _delta_snaps(snap0, snap1, "inference_errors_total")
        assert delta >= 3.0, f"Esperado +3 erros inferencia, delta={delta}"

    def test_inference_latency_histogram_plus_2(self):
        snap0 = _parse_snapshot()
        observe_inference_prediction("single", "urgente", "lr", 0.005, 1)
        observe_inference_prediction("single", "urgente", "lr", 0.0001, 1)
        snap1 = _parse_snapshot()
        delta = _delta_snaps(snap0, snap1, "inference_latency_seconds_count")
        assert delta >= 2.0, f"Esperado +2 obs hist latencia, delta={delta}"

    def test_model_loaded_gauge_set_and_clear(self):
        update_model_loaded_state(True, "logistic_regression", {"accuracy": 0.68, "f1_macro": 0.66})
        snap_on = _parse_snapshot()
        found_on = False
        for (name, lbls), val in snap_on.items():
            if name.startswith("model_loaded") and dict(lbls).get("model_name") == "logistic_regression":
                if val == 1.0:
                    found_on = True
                    break
        assert found_on, "Gauge model_loaded{model_name=logistic_regression} nao foi setado para 1.0"

        update_model_loaded_state(False, "logistic_regression")
        snap_off = _parse_snapshot()
        found_off = False
        for (name, lbls), val in snap_off.items():
            if name.startswith("model_loaded") and dict(lbls).get("model_name") == "logistic_regression":
                if val == 0.0:
                    found_off = True
                    break
        assert found_off, "Gauge model_loaded nao foi resetado para 0.0"

    def test_predict_endpoint_updates_inference_counters(self):
        with TestClient(app, raise_server_exceptions=False) as client:
            svc = get_model_service()
            if svc.model_file_exists() and not svc.is_loaded:
                svc.load_model(force=True)
            if not svc.is_loaded:
                pytest.skip("Modelo joblib nao disponivel para este teste.")
                return
            snap0 = _parse_snapshot()
            r = client.post("/predict", json={"text": self.SAMPLE_VALID, "return_probabilities": False})
            assert r.status_code == 200, f"status={r.status_code} body={r.text[:200]}"
            snap1 = _parse_snapshot()
        d_pred = _delta_snaps(snap0, snap1, "inference_predictions_total")
        d_items = _delta_snaps(snap0, snap1, "inference_items_processed_total")
        assert d_pred >= 1.0, f"predict nao incrementou inference_predictions_total (delta={d_pred})"
        assert d_items >= 1.0, f"predict nao incrementou inference_items_processed_total (delta={d_items})"

    def test_predict_invalid_text_marks_inference_error(self):
        with TestClient(app, raise_server_exceptions=False) as client:
            svc = get_model_service()
            if svc.model_file_exists() and not svc.is_loaded:
                svc.load_model(force=True)
            if not svc.is_loaded:
                pytest.skip("Modelo joblib nao disponivel.")
                return
            snap0 = _parse_snapshot()
            r = client.post("/predict", json={"text": self.SAMPLE_INVALID_SHORT})
            assert r.status_code == 400
            snap1 = _parse_snapshot()
        d_err = _delta_snaps(snap0, snap1, "inference_errors_total", {"error_type": "invalid_text"})
        any_err = _any_series(snap1, "inference_errors_total", {"endpoint_type": "single", "error_type": "invalid_text"})
        assert d_err >= 1.0 or any_err, (
            f"invalid_text nao incrementou inference_errors_total (delta={d_err}, any_series={any_err})"
        )


# ---------------------------------------------------------------------------
# Testes de LABEL CARDINALITY BAIXA
# ---------------------------------------------------------------------------


class TestLowCardinalityLabels:
    def test_repeated_same_labels_no_series_creation(self):
        """30 chamadas identicas nao criam > 50 series (regra baixa cardinalidade)."""
        for _ in range(30):
            observe_http_request("GET", "/health", 200, 0.001)
        snap = _parse_snapshot()
        n_series = sum(1 for k in snap if k[0] == "http_requests_total")
        assert n_series <= 50, (
            f"Alta cardinalidade em http_requests_total: {n_series} series (>50). Suspeita de label dinamico!"
        )

    def test_no_high_cardinality_labels_in_output(self):
        body = _scrape_text()
        forbidden_keys = (
            'request_id="',
            'input_text="',
            'text="',
            'user_agent="',
            'client_ip="',
            'body="',
            'payload="',
        )
        hits = [k for k in forbidden_keys if k in body]
        assert not hits, f"Labels suspeitas de alta cardinalidade no scrape: {hits}"
