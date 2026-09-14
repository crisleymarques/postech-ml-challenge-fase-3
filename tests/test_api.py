import json
from datetime import datetime

import pytest


SAMPLE_TEXT_VALID = (
    "The patient presented with acute chest pain radiating to the left arm, "
    "associated with diaphoresis and dyspnea. ECG showed ST elevation. "
    "Diagnosis: acute myocardial infarction."
)
SAMPLE_TEXT_GASTRITE = (
    "42-year-old patient with chronic gastritis reports intermittent heartburn "
    "after meals. Endoscopy showed mild pangastritis. Prescribed proton pump inhibitor."
)


@pytest.fixture
def dummy_model_service(tmp_path, monkeypatch):
    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    import joblib

    texts = [
        "acute chest pain myocardial infarction st elevation urgent angioplasty",
        "cardiac arrest cpr icu troponin coronary artery bypass",
        "stroke patient rehabilitation mri brain headache",
        "brain tumor biopsy surgical oncology urgent metastasis",
        "colon cancer chemotherapy oncology resection",
        "chronic gastritis proton pump endoscopy ulcer diet",
        "stomach ulcer hospital discharge oral antibiotics",
        "esophageal varices sclerotherapy cirrhosis hepatic",
        "diverticulitis antibiotics oral diet rest colon",
        "migraine dizziness neurological evaluation",
        "diabetes mellitus insulin glucose monitoring",
        "hypertension medication blood pressure control",
        "pneumonia antibiotics respiratory infection fever",
        "renal failure dialysis creatinine nephrology",
    ]
    labels = [2, 2, 1, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    from sklearn.utils.validation import check_random_state
    rs = check_random_state(42)
    texts = texts * 10
    labels = labels * 10
    indices = rs.permutation(len(texts))
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=500, ngram_range=(1, 1))),
        ("clf", LogisticRegression(C=10.0, max_iter=2000, random_state=42, class_weight="balanced")),
    ])
    pipe.fit(texts, labels)

    model_file = tmp_path / "urgency_classifier.joblib"
    joblib.dump({
        "model": pipe,
        "metadata": {"model_name": "logistic_regression_dummy", "test_f1_macro": 0.9},
    }, model_file)

    from app.config import Settings
    s = Settings(models_dir=str(tmp_path), request_text_min_length=5, request_batch_max_items=3)

    from app.services.model_service import ModelService, _model_service_instance
    ms = ModelService(config=s)
    ms.load_model(force=True)
    yield ms


@pytest.fixture
def app_with_dummy(dummy_model_service, monkeypatch):
    from app.services.model_service import _model_service_instance
    monkeypatch.setattr(
        "app.services.model_service._model_service_instance",
        dummy_model_service,
    )
    from app.main import create_app
    application = create_app()
    return application


@pytest.fixture
def client(app_with_dummy):
    from fastapi.testclient import TestClient
    with TestClient(app_with_dummy) as c:
        yield c


class TestSchemaValidation:
    def test_predict_request_ok(self):
        from app.schemas import PredictRequest
        r = PredictRequest(text="Patient with chest pain", return_probabilities=False, request_id="abc123")
        assert r.text == "Patient with chest pain"
        assert r.return_probabilities is False
        assert r.request_id == "abc123"

    def test_predict_request_text_missing(self):
        from pydantic import ValidationError
        from app.schemas import PredictRequest
        with pytest.raises(ValidationError):
            PredictRequest()

    def test_predict_request_empty_string_fails(self):
        from pydantic import ValidationError
        from app.schemas import PredictRequest
        with pytest.raises(ValidationError):
            PredictRequest(text="")


class TestModelServiceUnit:
    def test_is_loaded_true(self, dummy_model_service):
        assert dummy_model_service.is_loaded is True

    def test_metadata_contains_model_name(self, dummy_model_service):
        assert "logistic_regression_dummy" in dummy_model_service.metadata.values()

    def test_predict_valid_text(self, dummy_model_service):
        result = dummy_model_service.predict_single(SAMPLE_TEXT_VALID, return_probabilities=True)
        assert result["predicted_label"] in {0, 1, 2}
        assert result["predicted_name"] in {"normal", "atencao", "urgente"}
        assert result["probabilities"] is not None
        keys = set(result["probabilities"].keys())
        assert keys == {"normal", "atencao", "urgente"}
        assert abs(sum(result["probabilities"].values()) - 1.0) < 0.01
        assert isinstance(result["processed_at"], datetime)
        assert result["model_version"]

    def test_predict_without_probabilities(self, dummy_model_service):
        result = dummy_model_service.predict_single(SAMPLE_TEXT_VALID, return_probabilities=False)
        assert result["probabilities"] is None

    def test_predict_empty_string_invalid(self, dummy_model_service):
        from app.services.model_service import InvalidTextError
        with pytest.raises(InvalidTextError):
            dummy_model_service.predict_single("    ")

    def test_predict_too_short_invalid(self, dummy_model_service):
        from app.services.model_service import InvalidTextError
        with pytest.raises(InvalidTextError):
            dummy_model_service.predict_single("abc")

    def test_predict_batch_limit_enforced(self, dummy_model_service):
        from app.services.model_service import InvalidTextError
        items = [{"text": f"Patient chest pain report number {i}", "return_probabilities": False} for i in range(10)]
        with pytest.raises(InvalidTextError):
            dummy_model_service.predict_batch(items)

    def test_predict_batch_success(self, dummy_model_service):
        items = [
            {"text": SAMPLE_TEXT_VALID, "return_probabilities": True},
            {"text": SAMPLE_TEXT_GASTRITE, "return_probabilities": False},
        ]
        results = dummy_model_service.predict_batch(items)
        assert len(results) == 2
        assert results[0]["probabilities"] is not None
        assert results[1]["probabilities"] is None

    def test_model_not_loaded_error(self, monkeypatch):
        from app.services.model_service import ModelService, ModelNotLoadedError
        ms = ModelService()
        ms.unload()
        with pytest.raises(ModelNotLoadedError):
            ms.predict_single("Patient chest pain here")


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model_path_exists"] is True
        assert "app_name" in data and "timestamp" in data

    def test_health_contains_metadata(self, client):
        r = client.get("/health")
        data = r.json()
        assert isinstance(data["model_metadata"], dict)


class TestPredictEndpoint:
    def test_predict_success(self, client):
        payload = {
            "text": SAMPLE_TEXT_VALID,
            "return_probabilities": True,
            "request_id": "req-iam-001",
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["predicted_label"] in {0, 1, 2}
        assert data["predicted_name"] in {"normal", "atencao", "urgente"}
        assert data["request_id"] == "req-iam-001"
        assert "model_version" in data
        assert "processed_at" in data
        assert data["probabilities"] is not None
        assert set(data["probabilities"].keys()) == {"normal", "atencao", "urgente"}

    def test_predict_urgent_case(self, client):
        payload = {"text": "acute myocardial infarction chest pain ST elevation urgent angioplasty"}
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["predicted_name"] == "urgente" or data["predicted_label"] == 2

    def test_predict_invalid_empty_body(self, client):
        r = client.post("/predict", json={})
        assert r.status_code == 422
        data = r.json()
        assert data["error"] == "VALIDATION_ERROR"
        assert data["details"] and isinstance(data["details"], list)
        assert "traceback" not in data
        for item in data["details"]:
            assert "field" in item and "reason" in item

    def test_predict_invalid_no_key(self, client):
        r = client.post("/predict", json={"texto_ptbr": SAMPLE_TEXT_VALID})
        assert r.status_code == 422

    def test_predict_invalid_short_text(self, client):
        r = client.post("/predict", json={"text": "oi"})
        assert r.status_code == 400
        data = r.json()
        assert data["error"] == "INVALID_INPUT"
        assert "pelo menos" in data["message"].lower() or "least" in data["message"].lower()

    def test_predict_no_probabilities(self, client):
        payload = {"text": SAMPLE_TEXT_GASTRITE, "return_probabilities": False}
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        assert r.json()["probabilities"] is None


class TestBatchPredictEndpoint:
    def test_batch_success(self, client):
        payload = {"items": [
            {"text": SAMPLE_TEXT_VALID, "return_probabilities": True},
            {"text": SAMPLE_TEXT_GASTRITE, "return_probabilities": False, "request_id": "gastrite-1"},
        ]}
        r = client.post("/predict/batch", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["total_items"] == 2
        assert len(data["results"]) == 2
        assert data["results"][0]["probabilities"] is not None
        assert data["results"][1]["probabilities"] is None
        assert data["results"][1]["request_id"] == "gastrite-1"

    def test_batch_empty_items_fails(self, client):
        r = client.post("/predict/batch", json={"items": []})
        assert r.status_code == 422

    def test_batch_limit_exceeded(self, client):
        big = {"items": [{"text": f"Patient with report {i} chest pain."} for i in range(5)]}
        r = client.post("/predict/batch", json=big)
        assert r.status_code == 400


class TestRootEndpoint:
    def test_root_200(self, client):
        r = client.get("/")
        assert r.status_code == 200
        d = r.json()
        assert d["status"] == "running"
        assert "docs" in d


class TestModelNotLoadedScenarios:
    def test_health_unhealthy_when_no_file(self, tmp_path, monkeypatch):
        from app.config import Settings
        s = Settings(models_dir=str(tmp_path), request_text_min_length=5)
        from app.services.model_service import ModelService, _model_service_instance
        ms = ModelService(config=s)
        monkeypatch.setattr("app.services.model_service._model_service_instance", ms)

        from app.main import create_app
        from fastapi.testclient import TestClient
        app = create_app()
        with TestClient(app) as client:
            r = client.get("/health")
            assert r.status_code == 503
            assert r.json()["status"] == "unhealthy"

    def test_predict_503_when_unloaded(self, tmp_path, monkeypatch):
        from app.config import Settings
        s = Settings(models_dir=str(tmp_path), request_text_min_length=5)
        from app.services.model_service import ModelService
        ms = ModelService(config=s)
        monkeypatch.setattr("app.services.model_service._model_service_instance", ms)

        from app.main import create_app
        from fastapi.testclient import TestClient
        app = create_app()
        with TestClient(app) as client:
            r = client.post("/predict", json={"text": SAMPLE_TEXT_VALID})
            assert r.status_code == 503
            data = r.json()
            assert data["error"] == "SERVICE_UNAVAILABLE"
            assert "stack" not in json.dumps(data).lower()


class TestExceptionHandling:
    def test_404_has_nice_format(self, client):
        r = client.get("/this/route/does/not/exist")
        assert r.status_code == 404

    def test_bad_content_type(self, client):
        r = client.post("/predict", data="not-json")
        assert r.status_code == 422
