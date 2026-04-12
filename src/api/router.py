"""API routes for prediction and health checks."""

from __future__ import annotations

from time import perf_counter
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, Request

from ..data.preprocessor import engineer_features
from ..logger import get_logger
from ..models.trainer import ModelBundle
from .schemas import (
	BatchPredictionRequest,
	BatchPredictionResponse,
	PredictionRequest,
	PredictionResponse,
)

router = APIRouter()
LOGGER = get_logger(__name__)


def _baseline_score(payload: PredictionRequest) -> float:
	return (
		payload.Temperature * 10.0
		+ payload.Fuel_Price * 100.0
		+ payload.CPI * 0.05
		- payload.Unemployment * 20.0
		+ payload.Holiday_Flag * 500.0
	)


def _get_bundle(request: Request) -> ModelBundle:
	bundle = getattr(request.app.state, "model_bundle", None)
	if bundle is None:
		return ModelBundle(model=None, preprocessing_pipeline=None, metadata={"model_name": "baseline", "version": "0.0.0"})
	return bundle


def _predict_single(payload: PredictionRequest, bundle: ModelBundle) -> float:
	if bundle.model is None or bundle.preprocessing_pipeline is None:
		return float(_baseline_score(payload))

	raw_frame = pd.DataFrame([payload.model_dump()])
	engineered = engineer_features(raw_frame)
	transformed = bundle.preprocessing_pipeline.transform(engineered)
	return float(bundle.model.predict(transformed)[0])


@router.get("/health")
def health_check() -> dict[str, str]:
	return {"status": "ok"}


@router.get("/model/info")
def model_info(request: Request) -> dict[str, str]:
	bundle = _get_bundle(request)
	return {
		"model_name": bundle.metadata.get("model_name", "baseline"),
		"version": bundle.metadata.get("version", "0.0.0"),
	}


@router.post("/predict", response_model=PredictionResponse)
def predict(request: Request, payload: PredictionRequest) -> PredictionResponse:
	request_id = getattr(request.state, "request_id", str(uuid4()))
	bundle = _get_bundle(request)
	LOGGER.info("prediction_request request_id=%s payload=%s", request_id, payload.model_dump(), extra={"request_id": request_id})
	score = _predict_single(payload, bundle)
	response = PredictionResponse(
		predicted_weekly_sales=score,
		model_version=bundle.metadata.get("version", "0.0.0"),
		prediction_id=uuid4(),
	)
	LOGGER.info("prediction_response request_id=%s response=%s", request_id, response.model_dump(), extra={"request_id": request_id})
	return response


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: Request, payload: BatchPredictionRequest) -> BatchPredictionResponse:
	request_id = getattr(request.state, "request_id", str(uuid4()))
	bundle = _get_bundle(request)
	started = perf_counter()
	predictions = [
		PredictionResponse(
			predicted_weekly_sales=_predict_single(record, bundle),
			model_version=bundle.metadata.get("version", "0.0.0"),
			prediction_id=uuid4(),
		)
		for record in payload.records
	]
	elapsed_ms = (perf_counter() - started) * 1000.0
	response = BatchPredictionResponse(
		predictions=predictions,
		total_records=len(payload.records),
		processing_time_ms=elapsed_ms,
	)
	LOGGER.info("batch_prediction_response request_id=%s total_records=%s", request_id, len(payload.records), extra={"request_id": request_id})
	return response
