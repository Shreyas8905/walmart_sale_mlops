"""Data loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import get_settings
from ..logger import get_logger

LOGGER = get_logger(__name__)
EXPECTED_COLUMNS = (
	"Store",
	"Date",
	"Weekly_Sales",
	"Holiday_Flag",
	"Temperature",
	"Fuel_Price",
	"CPI",
	"Unemployment",
)
NUMERIC_COLUMNS = (
	"Store",
	"Weekly_Sales",
	"Holiday_Flag",
	"Temperature",
	"Fuel_Price",
	"CPI",
	"Unemployment",
)


def validate_schema(frame: pd.DataFrame) -> None:
	missing_columns = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
	if missing_columns:
		raise ValueError(f"Missing expected columns: {missing_columns}")

	for column in NUMERIC_COLUMNS:
		if not pd.api.types.is_numeric_dtype(frame[column]):
			raise ValueError(f"Column '{column}' must be numeric")

	if not pd.api.types.is_object_dtype(frame["Date"]) and not pd.api.types.is_string_dtype(
		frame["Date"]
	):
		raise ValueError("Column 'Date' must be a string or object dtype")


def load_raw_data(file_path: Path | None = None) -> pd.DataFrame:
	settings = get_settings()
	csv_path = file_path or (settings.paths.data_raw / "Walmart_Sales.csv")
	if not csv_path.exists():
		raise FileNotFoundError(f"Raw dataset not found: {csv_path}")

	frame = pd.read_csv(csv_path)
	validate_schema(frame)
	LOGGER.info("Loaded raw data with %s rows and %s columns", frame.shape[0], frame.shape[1])
	return frame
