"""SHAP explainability helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..config import AppConfig, get_settings
from ..data.preprocessor import engineer_features
from ..logger import get_logger
from ..models.trainer import ModelBundle, load_model_bundle

try:  # pragma: no cover - optional dependency
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None

try:  # pragma: no cover - optional dependency
    import shap
except Exception:  # pragma: no cover
    shap = None

LOGGER = get_logger(__name__)


def _save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def generate_shap_explanations(
    x_test,
    bundle: ModelBundle | None = None,
    settings: AppConfig | None = None,
) -> dict[str, Path]:
    settings = settings or get_settings()
    bundle = bundle or load_model_bundle(settings)
    if bundle.model is None or bundle.preprocessing_pipeline is None:
        raise RuntimeError(
            "Champion model and preprocessing pipeline must exist before explainability runs"
        )

    engineered = engineer_features(x_test)
    transformed = bundle.preprocessing_pipeline.transform(engineered)
    transformed = np.asarray(transformed)
    feature_names = getattr(
        bundle.preprocessing_pipeline.named_steps["preprocessor"],
        "get_feature_names_out",
        lambda: None,
    )()
    if feature_names is None:
        feature_names = [f"feature_{index}" for index in range(transformed.shape[1])]

    model = bundle.model
    if shap is not None:
        try:
            if hasattr(
                model, "estimators_"
            ) or model.__class__.__name__.lower().startswith(
                ("randomforest", "gradientboosting", "xgb", "histgradientboosting")
            ):
                explainer = shap.TreeExplainer(model)
            elif hasattr(model, "coef_"):
                explainer = shap.LinearExplainer(
                    model, transformed, feature_perturbation="interventional"
                )
            else:
                background = shap.sample(
                    transformed,
                    min(100, len(transformed)),
                    random_state=settings.project.random_seed,
                )
                explainer = shap.KernelExplainer(model.predict, background)
        except Exception:
            background = shap.sample(
                transformed,
                min(100, len(transformed)),
                random_state=settings.project.random_seed,
            )
            explainer = shap.KernelExplainer(model.predict, background)

        shap_values = explainer.shap_values(transformed)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_array = np.asarray(shap_values)
    else:
        LOGGER.warning(
            "shap package unavailable; using deterministic fallback contribution approximation"
        )
        if hasattr(model, "feature_importances_"):
            importance = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            importance = np.abs(np.asarray(model.coef_, dtype=float)).reshape(-1)
        else:
            importance = np.ones(transformed.shape[1], dtype=float)
        denom = np.sum(np.abs(importance))
        if denom == 0:
            importance = np.ones_like(importance)
            denom = np.sum(np.abs(importance))
        importance = importance / denom
        centered = transformed - transformed.mean(axis=0)
        shap_array = centered * importance

    reports_dir = settings.paths.reports
    reports_dir.mkdir(parents=True, exist_ok=True)
    shap_values_path = reports_dir / "shap_values.npy"
    np.save(shap_values_path, shap_array)

    mean_abs = np.mean(np.abs(shap_array), axis=0)
    ranked = np.argsort(mean_abs)[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(
        [str(feature_names[index]) for index in ranked[: min(15, len(ranked))]][::-1],
        mean_abs[ranked[: min(15, len(ranked))]][::-1],
    )
    plt.xlabel("Mean |contribution|")
    plt.title("Feature importance summary")
    summary_bar_path = reports_dir / "shap_summary_bar.png"
    _save_plot(summary_bar_path)

    plt.figure(figsize=(10, 6))
    top = ranked[: min(10, len(ranked))]
    for index in top:
        plt.scatter(
            np.full(shape=shap_array.shape[0], fill_value=int(index)),
            shap_array[:, index],
            s=8,
            alpha=0.35,
        )
    plt.xticks(
        range(len(top)),
        [str(feature_names[index]) for index in top],
        rotation=45,
        ha="right",
    )
    plt.ylabel("Contribution value")
    plt.title("Contribution distribution (beeswarm-style)")
    summary_beeswarm_path = reports_dir / "shap_summary_beeswarm.png"
    _save_plot(summary_beeswarm_path)

    waterfall_path = reports_dir / "shap_waterfall_sample_0.png"
    sample_values = np.asarray(shap_array[0])
    ranked_indices = np.argsort(np.abs(sample_values))[::-1][
        : min(12, len(sample_values))
    ]
    plt.figure(figsize=(10, 6))
    plt.barh(
        [str(feature_names[index]) for index in ranked_indices][::-1],
        sample_values[ranked_indices][::-1],
    )
    plt.xlabel("SHAP value")
    plt.title("Sample 0 feature contribution")
    _save_plot(waterfall_path)

    if mlflow is not None:
        mlflow.log_artifact(str(shap_values_path), artifact_path="explainability")
        mlflow.log_artifact(str(summary_bar_path), artifact_path="explainability")
        mlflow.log_artifact(str(summary_beeswarm_path), artifact_path="explainability")
        mlflow.log_artifact(str(waterfall_path), artifact_path="explainability")

    LOGGER.info("Saved SHAP artefacts to %s", reports_dir)
    return {
        "shap_values": shap_values_path,
        "summary_bar": summary_bar_path,
        "summary_beeswarm": summary_beeswarm_path,
        "waterfall": waterfall_path,
    }
