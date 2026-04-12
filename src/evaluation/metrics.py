"""Evaluation metrics and champion selection helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmse(y_true, y_pred) -> float:
	return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
	return float(mean_absolute_error(y_true, y_pred))


def r2(y_true, y_pred) -> float:
	return float(r2_score(y_true, y_pred))


def mape(y_true, y_pred) -> float:
	y_true_array = np.asarray(y_true, dtype=float)
	y_pred_array = np.asarray(y_pred, dtype=float)
	mask = y_true_array != 0
	if not np.any(mask):
		return 0.0
	return float(np.mean(np.abs((y_true_array[mask] - y_pred_array[mask]) / y_true_array[mask])) * 100.0)


def compute_all_metrics(y_true, y_pred) -> dict[str, float]:
	return {
		"rmse": rmse(y_true, y_pred),
		"mae": mae(y_true, y_pred),
		"r2": r2(y_true, y_pred),
		"mape": mape(y_true, y_pred),
	}


def select_champion(results: list[dict[str, Any]], metric: str, direction: str) -> dict[str, Any]:
	if not results:
		raise ValueError("results cannot be empty")
	if direction not in {"minimize", "maximize"}:
		raise ValueError("direction must be either 'minimize' or 'maximize'")

	reverse = direction == "maximize"
	return sorted(results, key=lambda item: item[metric], reverse=reverse)[0]
