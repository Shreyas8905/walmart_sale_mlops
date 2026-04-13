"""Tests for evaluation metrics."""

from __future__ import annotations

import pytest

from src.evaluation.metrics import (
    compute_all_metrics,
    mae,
    mape,
    r2,
    rmse,
    select_champion,
)


def test_metric_functions_known_values():
    y_true = [100.0, 200.0, 300.0]
    y_pred = [110.0, 190.0, 305.0]

    assert rmse(y_true, y_pred) == pytest.approx(8.6602540378, rel=1e-6)
    assert mae(y_true, y_pred) == pytest.approx(8.3333333333, rel=1e-6)
    assert r2(y_true, y_pred) == pytest.approx(0.98875, rel=1e-6)
    assert mape(y_true, y_pred) == pytest.approx(5.5555555556, rel=1e-6)


def test_compute_all_metrics():
    metrics = compute_all_metrics([1, 2], [1, 2])
    assert set(metrics.keys()) == {"rmse", "mae", "r2", "mape"}


def test_select_champion_minimize_and_maximize():
    results = [
        {"model_name": "a", "rmse": 5.0, "r2": 0.90},
        {"model_name": "b", "rmse": 4.0, "r2": 0.88},
    ]
    assert (
        select_champion(results, metric="rmse", direction="minimize")["model_name"]
        == "b"
    )
    assert (
        select_champion(results, metric="r2", direction="maximize")["model_name"] == "a"
    )
