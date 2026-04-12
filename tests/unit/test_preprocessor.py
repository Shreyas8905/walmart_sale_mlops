"""Tests for the preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.preprocessor import (
	build_preprocessing_pipeline,
	engineer_features,
	fit_transform_pipeline,
	load_pipeline,
	save_pipeline,
	transform_pipeline,
)


def test_engineer_features_creates_expected_columns(sample_df):
	features = sample_df.drop(columns=["Weekly_Sales"])
	engineered = engineer_features(features)
	assert "Date" not in engineered.columns
	for expected in ["Week", "Month", "Year", "DayOfYear"]:
		assert expected in engineered.columns


def test_preprocessing_pipeline_output_shape(sample_df):
	features = sample_df.drop(columns=["Weekly_Sales"])
	engineered = engineer_features(features)
	pipeline = build_preprocessing_pipeline()
	transformed = fit_transform_pipeline(pipeline, engineered)
	assert isinstance(transformed, np.ndarray)
	assert transformed.shape[0] == len(engineered)
	assert transformed.shape[1] == 10


def test_preprocessing_pipeline_pickle_round_trip(tmp_path, sample_df):
	features = sample_df.drop(columns=["Weekly_Sales"])
	engineered = engineer_features(features)
	pipeline = build_preprocessing_pipeline()
	fit_transform_pipeline(pipeline, engineered)

	save_path = tmp_path / "pipeline.pkl"
	save_pipeline(pipeline, save_path)
	loaded_pipeline = load_pipeline(save_path)

	original = transform_pipeline(pipeline, engineered)
	reloaded = transform_pipeline(loaded_pipeline, engineered)
	assert np.allclose(original, reloaded)
