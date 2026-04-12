"""Tests for API schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.schemas import BatchPredictionRequest, PredictionRequest


def test_prediction_request_valid_payload():
	payload = PredictionRequest(
		Store=1,
		Date="05-02-2010",
		Holiday_Flag=0,
		Temperature=42.31,
		Fuel_Price=2.572,
		CPI=211.0963582,
		Unemployment=8.106,
	)
	assert payload.Store == 1


def test_prediction_request_missing_field_error():
	with pytest.raises(ValidationError):
		PredictionRequest(
			Store=1,
			Date="05-02-2010",
			Holiday_Flag=0,
			Temperature=42.31,
			Fuel_Price=2.572,
			CPI=211.0963582,
		)


def test_prediction_request_wrong_types_and_range():
	with pytest.raises(ValidationError):
		PredictionRequest(
			Store="x",
			Date="bad-date",
			Holiday_Flag=5,
			Temperature="warm",
			Fuel_Price=2.572,
			CPI=211.0963582,
			Unemployment=8.106,
		)


def test_batch_prediction_max_records_limit():
	payload = {
		"records": [
			{
				"Store": 1,
				"Date": "05-02-2010",
				"Holiday_Flag": 0,
				"Temperature": 42.31,
				"Fuel_Price": 2.572,
				"CPI": 211.0963582,
				"Unemployment": 8.106,
			}
		]
		* 1001
	}
	with pytest.raises(ValidationError):
		BatchPredictionRequest(**payload)
