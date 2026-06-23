from fastapi.testclient import TestClient

from src.service.main import _load_artifacts, app

_load_artifacts()
client = TestClient(app)

VALID_PAYLOAD = {
    "LIMIT_BAL": 200000,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 35,
    "max_delay": 0,
    "mean_delay": -0.5,
    "n_months_delayed": 0,
    "avg_bill_amt": 45000,
    "avg_pay_amt": 6000,
    "payment_to_bill_ratio": 0.8,
    "credit_utilization": 0.225,
    "bill_trend": 1200.0,
}


def test_health_ok():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_valid_response():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["proba_default"] <= 1.0
    assert body["risk_category"] in ("low", "medium", "high")


def test_predict_rejects_invalid_age():
    bad_payload = dict(VALID_PAYLOAD, AGE=5)
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_rejects_missing_field():
    bad_payload = dict(VALID_PAYLOAD)
    del bad_payload["credit_utilization"]
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_high_risk_profile_has_higher_proba_than_low_risk():
    low_risk = dict(VALID_PAYLOAD)
    high_risk = dict(
        VALID_PAYLOAD,
        max_delay=4, mean_delay=3.0, n_months_delayed=5,
        avg_pay_amt=0, payment_to_bill_ratio=0.02, credit_utilization=1.8,
    )
    proba_low = client.post("/predict", json=low_risk).json()["proba_default"]
    proba_high = client.post("/predict", json=high_risk).json()["proba_default"]
    assert proba_high > proba_low
