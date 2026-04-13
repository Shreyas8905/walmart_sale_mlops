"""Shared pytest fixtures."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.config import PROJECT_ROOT, get_settings
from src.data.preprocessor import (
    build_preprocessing_pipeline,
    engineer_features,
    fit_transform_pipeline,
)


def pytest_configure(config) -> None:
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    test_log_file = logs_dir / "test_results.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.FileHandler(test_log_file, encoding="utf-8")],
        force=True,
    )


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = 50
    dates = pd.date_range("2010-01-01", periods=rows, freq="W-FRI")
    return pd.DataFrame(
        {
            "Store": rng.integers(1, 11, size=rows),
            "Date": dates.strftime("%d-%m-%Y"),
            "Weekly_Sales": rng.normal(1_500_000, 120_000, size=rows).round(2),
            "Holiday_Flag": rng.integers(0, 2, size=rows),
            "Temperature": rng.normal(65.0, 10.0, size=rows).round(2),
            "Fuel_Price": rng.normal(3.0, 0.3, size=rows).round(3),
            "CPI": rng.normal(210.0, 5.0, size=rows).round(6),
            "Unemployment": rng.normal(7.5, 1.0, size=rows).round(3),
        }
    )


@pytest.fixture
def config():
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture
def preprocessing_pipeline(sample_df):
    features = sample_df.drop(columns=["Weekly_Sales"])
    engineered = engineer_features(features)
    pipeline = build_preprocessing_pipeline()
    fit_transform_pipeline(pipeline, engineered)
    return pipeline


@pytest.fixture
def app_client():
    with TestClient(app) as client:
        yield client
