"""API request and response schemas."""

from __future__ import annotations

from typing import ClassVar
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
	Store: int
	Date: str
	Holiday_Flag: int = Field(ge=0, le=1)
	Temperature: float
	Fuel_Price: float
	CPI: float
	Unemployment: float


class PredictionResponse(BaseModel):
	predicted_weekly_sales: float
	model_version: str
	prediction_id: UUID


class BatchPredictionRequest(BaseModel):
	records: list[PredictionRequest]
	max_records: ClassVar[int] = 1000

	@field_validator("records")
	@classmethod
	def validate_records(cls, records: list[PredictionRequest]) -> list[PredictionRequest]:
		if not records:
			raise ValueError("records must not be empty")
		if len(records) > cls.max_records:
			raise ValueError("records cannot exceed 1000 items")
		return records


class BatchPredictionResponse(BaseModel):
	predictions: list[PredictionResponse]
	total_records: int
	processing_time_ms: float
