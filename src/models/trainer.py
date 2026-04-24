"""Training loop and MLflow experiment orchestration."""

from __future__ import annotations

import json
import pickle
import tempfile
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from ..config import AppConfig, get_settings
from ..evaluation.metrics import compute_all_metrics, select_champion
from ..logger import get_logger
from .registry import ModelSpec, build_model_registry

try:  # pragma: no cover - optional dependency
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None

LOGGER = get_logger(__name__)


@dataclass
class TrainingArtifact:
    model_name: str
    parameters: dict[str, Any]
    metrics: dict[str, float]
    run_id: str | None
    model: Any


@dataclass
class TrainingSummary:
    results: list[dict[str, Any]] = field(default_factory=list)
    champion: dict[str, Any] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)
    summary_path: Path | None = None


@dataclass
class ModelBundle:
    model: Any | None
    preprocessing_pipeline: Any | None
    metadata: dict[str, Any] = field(default_factory=dict)


class _NullRun:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    @property
    def info(self):
        return type("Info", (), {"run_id": None})()


def _mlflow_start_run(**kwargs):
    if mlflow is None:
        return _NullRun()
    return mlflow.start_run(**kwargs)


def _iter_param_combinations(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not param_grid:
        return [{}]
    keys = list(param_grid.keys())
    values = [param_grid[key] for key in keys]
    return [dict(zip(keys, combination)) for combination in product(*values)]


def _apply_common_model_options(model: Any, random_seed: int) -> Any:
    if hasattr(model, "random_state") and getattr(model, "random_state", None) is None:
        try:
            setattr(model, "random_state", random_seed)
        except Exception:
            pass
    if hasattr(model, "max_iter") and getattr(model, "max_iter", None) is None:
        try:
            setattr(model, "max_iter", 10000)
        except Exception:
            pass
    if hasattr(model, "n_jobs") and getattr(model, "n_jobs", None) is None:
        try:
            setattr(model, "n_jobs", -1)
        except Exception:
            pass
    return model


def _instantiate_model(
    estimator_cls: type, parameters: dict[str, Any], random_seed: int
) -> Any:
    normalized_parameters = {
        key: (
            None
            if isinstance(value, str) and value.lower() in {"none", "null"}
            else value
        )
        for key, value in parameters.items()
    }
    model = estimator_cls(**normalized_parameters)
    return _apply_common_model_options(model, random_seed)


def _to_serializable_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serializable_results: list[dict[str, Any]] = []
    for result in results:
        item = {key: value for key, value in result.items() if key != "model"}
        serializable_results.append(item)
    return serializable_results


def _log_artifact(path: Path, artifact_path: str | None = None) -> None:
    if mlflow is None:
        return
    mlflow.log_artifact(str(path), artifact_path=artifact_path)


def _log_evaluation_table(rows: list[dict[str, Any]], artifact_file: str) -> None:
    if mlflow is None or not rows:
        return

    if hasattr(mlflow, "log_table"):
        normalized_columns: dict[str, list[Any]] = {}
        all_keys = sorted({key for row in rows for key in row.keys()})
        for key in all_keys:
            normalized_columns[key] = [row.get(key) for row in rows]
        mlflow.log_table(normalized_columns, artifact_file)
        return

    # Backward-compatible fallback for older MLflow versions.
    with tempfile.NamedTemporaryFile(
        suffix=".json", delete=False, mode="w", encoding="utf-8"
    ) as temp_file:
        json.dump(rows, temp_file, indent=2, default=str)
        temp_table_path = Path(temp_file.name)
    _log_artifact(temp_table_path, artifact_path=str(Path(artifact_file).parent))
    temp_table_path.unlink(missing_ok=True)


def _build_trial_table_row(
    model_name: str,
    parameters: dict[str, Any],
    metrics: dict[str, float],
    run_id: str | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model_name": model_name,
        "run_id": run_id,
    }
    row.update({f"param_{key}": value for key, value in parameters.items()})
    row.update({f"val_{key}": value for key, value in metrics.items()})
    return row


def train_models(
    x_train: np.ndarray,
    y_train,
    x_val: np.ndarray,
    y_val,
    x_test: np.ndarray,
    y_test,
    preprocessing_pipeline,
    settings: AppConfig | None = None,
) -> TrainingSummary:
    settings = settings or get_settings()
    if mlflow is not None:
        mlflow.set_tracking_uri(str(settings.mlflow.tracking_uri))
        mlflow.set_experiment(settings.mlflow.experiment_name)

    results: list[dict[str, Any]] = []
    model_registry = build_model_registry()
    trial_group = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for model_spec in model_registry:
        for index, parameters in enumerate(
            _iter_param_combinations(model_spec.param_grid)
        ):
            model = _instantiate_model(
                model_spec.estimator_cls,
                parameters,
                settings.project.random_seed,
            )
            with _mlflow_start_run(run_name=f"{model_spec.name}_{index}") as run:
                if mlflow is not None:
                    mlflow.set_tags(
                        {
                            "pipeline": "train",
                            "trial_group": trial_group,
                            "model_name": model_spec.name,
                            "project_name": settings.project.name,
                            "project_version": settings.project.version,
                        }
                    )
                    mlflow.log_params(
                        {
                            "random_seed": settings.project.random_seed,
                            **parameters,
                        }
                    )

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    model.fit(x_train, y_train)
                validation_predictions = model.predict(x_val)
                metrics = compute_all_metrics(y_val, validation_predictions)
                result = {
                    "model_name": model_spec.name,
                    "parameters": parameters,
                    **metrics,
                    "run_id": getattr(run.info, "run_id", None),
                    "model": model,
                }
                results.append(result)

                if mlflow is not None:
                    mlflow.log_metrics(
                        {
                            f"val_{metric_name}": metric_value
                            for metric_name, metric_value in metrics.items()
                        }
                    )
                    _log_evaluation_table(
                        [
                            _build_trial_table_row(
                                model_spec.name,
                                parameters,
                                metrics,
                                getattr(run.info, "run_id", None),
                            )
                        ],
                        artifact_file="tables/evaluation.json",
                    )
                    with tempfile.NamedTemporaryFile(
                        suffix=".pkl", delete=False
                    ) as temp_file:
                        pickle.dump(model, temp_file)
                        temp_model_path = Path(temp_file.name)
                    _log_artifact(temp_model_path, artifact_path="models")
                    temp_model_path.unlink(missing_ok=True)

                LOGGER.info(
                    "Trained %s with params=%s rmse=%.4f",
                    model_spec.name,
                    parameters,
                    metrics["rmse"],
                )

    champion = select_champion(
        results,
        settings.champion_selection.metric,
        settings.champion_selection.direction,
    )
    champion_model = champion["model"]

    model_path = settings.paths.models / "champion_model.pkl"
    preprocessing_path = settings.paths.models / "preprocessing_pipeline.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as model_handle:
        pickle.dump(champion_model, model_handle)
    with preprocessing_path.open("wb") as pipeline_handle:
        pickle.dump(preprocessing_pipeline, pipeline_handle)

    test_predictions = champion_model.predict(x_test)
    test_metrics = compute_all_metrics(y_test, test_predictions)

    summary = {
        "champion": {key: value for key, value in champion.items() if key != "model"},
        "test_metrics": test_metrics,
        "results": _to_serializable_results(results),
    }
    summary_path = settings.paths.reports / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    if mlflow is not None:
        champion_run_id = champion.get("run_id")
        if champion_run_id:
            client = mlflow.tracking.MlflowClient()
            client.set_tag(champion_run_id, "is_champion", "true")
            client.set_tag(
                champion_run_id,
                "champion_metric",
                settings.champion_selection.metric,
            )
            client.set_tag(
                champion_run_id,
                "champion_direction",
                settings.champion_selection.direction,
            )
            for metric_name, metric_value in test_metrics.items():
                client.log_metric(
                    champion_run_id,
                    f"test_{metric_name}",
                    metric_value,
                )
            client.log_artifact(
                champion_run_id, str(model_path), artifact_path="models"
            )
            client.log_artifact(
                champion_run_id,
                str(preprocessing_path),
                artifact_path="models",
            )
            client.log_artifact(
                champion_run_id, str(summary_path), artifact_path="reports"
            )

        with _mlflow_start_run(run_name=f"training_summary_{trial_group}"):
            mlflow.set_tags(
                {
                    "pipeline": "train",
                    "trial_group": trial_group,
                    "run_type": "summary",
                    "champion_model": champion["model_name"],
                    "champion_metric": settings.champion_selection.metric,
                    "champion_direction": settings.champion_selection.direction,
                }
            )
            mlflow.log_metrics(
                {
                    f"champion_val_{metric_name}": champion[metric_name]
                    for metric_name in ("rmse", "mae", "mape", "r2")
                    if metric_name in champion
                }
            )
            mlflow.log_metrics(
                {
                    f"test_{metric_name}": metric_value
                    for metric_name, metric_value in test_metrics.items()
                }
            )
            comparison_rows: list[dict[str, Any]] = []
            champion_run_id = champion.get("run_id")
            for result in results:
                row = _build_trial_table_row(
                    model_name=result["model_name"],
                    parameters=result["parameters"],
                    metrics={
                        "rmse": result["rmse"],
                        "mae": result["mae"],
                        "mape": result["mape"],
                        "r2": result["r2"],
                    },
                    run_id=result.get("run_id"),
                )
                row["is_champion"] = result.get("run_id") == champion_run_id
                if row["is_champion"]:
                    row.update(
                        {
                            f"test_{metric_name}": metric_value
                            for metric_name, metric_value in test_metrics.items()
                        }
                    )
                comparison_rows.append(row)

            _log_evaluation_table(
                comparison_rows,
                artifact_file="tables/model_comparison.json",
            )
            _log_artifact(summary_path, artifact_path="reports")

    LOGGER.info(
        "Selected champion %s with %s=%.4f",
        champion["model_name"],
        settings.champion_selection.metric,
        champion[settings.champion_selection.metric],
    )

    return TrainingSummary(
        results=_to_serializable_results(results),
        champion={key: value for key, value in champion.items() if key != "model"},
        test_metrics=test_metrics,
        summary_path=summary_path,
    )


def load_model_bundle(settings: AppConfig | None = None) -> ModelBundle:
    settings = settings or get_settings()
    model_path = settings.paths.models / "champion_model.pkl"
    pipeline_path = settings.paths.models / "preprocessing_pipeline.pkl"
    summary_path = settings.paths.reports / "training_summary.json"

    model = None
    preprocessing_pipeline = None
    metadata: dict[str, Any] = {
        "model_name": "baseline",
        "version": settings.project.version,
    }

    if model_path.exists():
        with model_path.open("rb") as model_handle:
            model = pickle.load(model_handle)
    if pipeline_path.exists():
        with pipeline_path.open("rb") as pipeline_handle:
            preprocessing_pipeline = pickle.load(pipeline_handle)
    if summary_path.exists():
        summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
        champion = summary_data.get("champion", {})
        metadata.update(
            {
                "model_name": champion.get("model_name", metadata["model_name"]),
                "version": settings.project.version,
                "champion_metric": champion.get(settings.champion_selection.metric),
            }
        )

    return ModelBundle(
        model=model, preprocessing_pipeline=preprocessing_pipeline, metadata=metadata
    )
