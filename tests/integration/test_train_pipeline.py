"""Integration tests for the training pipeline."""

from __future__ import annotations

import json

import numpy as np

from src.config import get_settings
from src.data.preprocessor import build_preprocessing_pipeline, engineer_features, fit_transform_pipeline, transform_pipeline
from src.models.trainer import train_models


def test_train_pipeline_core_flow(sample_df):
	settings = get_settings()
	sorted_df = sample_df.sort_values("Date").reset_index(drop=True)

	train_df = sorted_df.iloc[:30]
	val_df = sorted_df.iloc[30:40]
	test_df = sorted_df.iloc[40:]

	x_train = train_df.drop(columns=[settings.project.target])
	y_train = train_df[settings.project.target]
	x_val = val_df.drop(columns=[settings.project.target])
	y_val = val_df[settings.project.target]
	x_test = test_df.drop(columns=[settings.project.target])
	y_test = test_df[settings.project.target]

	pipeline = build_preprocessing_pipeline()
	x_train_t = fit_transform_pipeline(pipeline, engineer_features(x_train))
	x_val_t = transform_pipeline(pipeline, engineer_features(x_val))
	x_test_t = transform_pipeline(pipeline, engineer_features(x_test))

	summary = train_models(x_train_t, y_train, x_val_t, y_val, x_test_t, y_test, pipeline, settings)
	assert summary.champion["model_name"]
	assert (settings.paths.models / "champion_model.pkl").exists()
	assert (settings.paths.models / "preprocessing_pipeline.pkl").exists()
	assert summary.summary_path is not None and summary.summary_path.exists()

	payload = json.loads(summary.summary_path.read_text(encoding="utf-8"))
	assert "results" in payload and payload["results"]
