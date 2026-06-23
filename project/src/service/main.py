"""
FastAPI-сервис кредитного скоринга.

Запуск локально:
    uvicorn src.service.main:app --reload --port 8000
"""
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.service.schemas import CreditApplication, HealthResponse, PredictionResponse

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_ARTIFACT_PATH", "artifacts/model.joblib")
METADATA_PATH = PROJECT_ROOT / os.getenv("MODEL_METADATA_PATH", "artifacts/model_metadata.json")
LOG_LEVEL = os.getenv("CREDIT_SCORING_LOG_LEVEL", "INFO")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)-5s %(name)s %(message)s",
)
logger = logging.getLogger("credit_scoring")

_model = None
_metadata = None


def _load_artifacts():
    global _model, _metadata
    if MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
        logger.info("Модель загружена из %s", MODEL_PATH)
    else:
        _model = None
        logger.error("Файл модели не найден: %s", MODEL_PATH)

    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            _metadata = json.load(f)
    else:
        _metadata = {"feature_columns": [], "decision_threshold": 0.5, "best_model": "unknown"}
        logger.warning("Файл метаданных модели не найден: %s", METADATA_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifacts()
    yield


app = FastAPI(title="Credit Risk Scoring Service", version="1.0", lifespan=lifespan)


def risk_category(proba: float) -> str:
    if proba < 0.15:
        return "low"
    if proba < 0.35:
        return "medium"
    return "high"


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    request_id = f"{int(start * 1000) % 100000:05d}"
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("id=%s %s %s -> unhandled error", request_id, request.method, request.url.path)
        raise
    latency_ms = round((time.time() - start) * 1000, 1)
    logger.info(
        "id=%s %s %s -> status=%s latency=%sms",
        request_id, request.method, request.url.path, response.status_code, latency_ms,
    )
    return response


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=_model is not None)


@app.post("/predict", response_model=PredictionResponse)
def predict(application: CreditApplication):
    if _model is None:
        logger.error("Запрос /predict при отсутствующей модели")
        return JSONResponse(status_code=503, content={"detail": "Модель не загружена"})

    feature_columns = _metadata.get("feature_columns") or list(application.model_dump().keys())
    row = pd.DataFrame([application.model_dump()])[feature_columns]

    proba = float(_model.predict_proba(row)[0, 1])
    threshold = float(_metadata.get("decision_threshold", 0.5))
    prediction = int(proba >= threshold)

    logger.info("predict proba=%.4f threshold=%.4f prediction=%s", proba, threshold, prediction)

    return PredictionResponse(
        prediction=prediction,
        proba_default=round(proba, 4),
        risk_category=risk_category(proba),
        model_version=_metadata.get("best_model", "unknown"),
    )
