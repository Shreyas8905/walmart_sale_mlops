"""Feature engineering and preprocessing pipeline helpers."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from ..config import get_settings


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
	settings = get_settings()
	engineered = frame.copy()
	date_series = pd.to_datetime(
		engineered[settings.data.date_column],
		format=settings.data.date_format,
		errors="raise",
	)
	engineered["Week"] = date_series.dt.isocalendar().week.astype(int)
	engineered["Month"] = date_series.dt.month.astype(int)
	engineered["Year"] = date_series.dt.year.astype(int)
	engineered["DayOfYear"] = date_series.dt.dayofyear.astype(int)
	engineered = engineered.drop(columns=[settings.data.date_column])
	return engineered


def build_preprocessing_pipeline() -> Pipeline:
	settings = get_settings()
	numerical_features = settings.data.numerical_features
	categorical_features = settings.data.categorical_features

	numeric_transformer = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="median")),
			("scaler", StandardScaler()),
		]
	)
	categorical_transformer = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="most_frequent")),
			(
				"encoder",
				OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
			),
		]
	)

	preprocessor = ColumnTransformer(
		transformers=[
			("numeric", numeric_transformer, numerical_features),
			("categorical", categorical_transformer, categorical_features),
		],
		remainder="drop",
	)
	return Pipeline(steps=[("preprocessor", preprocessor)])


def fit_transform_pipeline(pipeline: Pipeline, x_train: pd.DataFrame) -> np.ndarray:
	return pipeline.fit_transform(x_train)


def transform_pipeline(pipeline: Pipeline, x_frame: pd.DataFrame) -> np.ndarray:
	return pipeline.transform(x_frame)


def save_pipeline(pipeline: Pipeline, path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("wb") as file_handle:
		pickle.dump(pipeline, file_handle)


def load_pipeline(path: Path) -> Pipeline:
	with path.open("rb") as file_handle:
		return pickle.load(file_handle)
