"""Training pipeline CLI entry point."""

from __future__ import annotations

from src.config import get_settings
from src.data.loader import load_raw_data
from src.data.preprocessor import (
	build_preprocessing_pipeline,
	engineer_features,
	fit_transform_pipeline,
	transform_pipeline,
)
from src.data.splitter import temporal_split
from src.logger import get_logger
from src.models.trainer import train_models

from .explain_pipeline import run_explain_pipeline

LOGGER = get_logger(__name__)


def run_train_pipeline() -> int:
	settings = get_settings()
	raw_data = load_raw_data()
	x_train, x_val, x_test, y_train, y_val, y_test = temporal_split(raw_data)

	x_train_engineered = engineer_features(x_train)
	x_val_engineered = engineer_features(x_val)
	x_test_engineered = engineer_features(x_test)

	preprocessing_pipeline = build_preprocessing_pipeline()
	x_train_transformed = fit_transform_pipeline(preprocessing_pipeline, x_train_engineered)
	x_val_transformed = transform_pipeline(preprocessing_pipeline, x_val_engineered)
	x_test_transformed = transform_pipeline(preprocessing_pipeline, x_test_engineered)

	train_models(
		x_train_transformed,
		y_train,
		x_val_transformed,
		y_val,
		x_test_transformed,
		y_test,
		preprocessing_pipeline,
		settings,
	)
	run_explain_pipeline(settings)
	LOGGER.info("Training pipeline completed successfully")
	return 0


def main() -> None:
	try:
		raise SystemExit(run_train_pipeline())
	except Exception as exc:  # pragma: no cover - CLI failure path
		LOGGER.exception("Training pipeline failed: %s", exc)
		raise SystemExit(1) from exc


if __name__ == "__main__":
	main()
