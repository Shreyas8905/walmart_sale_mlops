"""API routes for prediction and health checks."""

from __future__ import annotations

from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter

from .schemas import (
	BatchPredictionRequest,
	BatchPredictionResponse,
	PredictionRequest,
	PredictionResponse,
)

router = APIRouter()
MODEL_INFO = {"model_name": "champion_model", "version": "0.0.0"}


def _predict_single(payload: PredictionRequest) -> PredictionResponse:
	score = (
		payload.Temperature * 10.0
		+ payload.Fuel_Price * 100.0
		+ payload.CPI * 0.05
		- payload.Unemployment * 20.0
		+ payload.Holiday_Flag * 500.0
	)
	return PredictionResponse(
		predicted_weekly_sales=float(score),
		model_version=MODEL_INFO["version"],
		prediction_id=uuid4(),
	)


@router.get("/health")
def health_check() -> dict[str, str]:
	return {"status": "ok"}


@router.get("/model/info")
def model_info() -> dict[str, str]:
	return MODEL_INFO


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
	return _predict_single(payload)


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchPredictionRequest) -> BatchPredictionResponse:
	started = perf_counter()
	predictions = [_predict_single(record) for record in payload.records]
	elapsed_ms = (perf_counter() - started) * 1000.0
	return BatchPredictionResponse(
		predictions=predictions,
		total_records=len(payload.records),
		processing_time_ms=elapsed_ms,
	)
