"""LIME explainability helpers."""

from __future__ import annotations

import random
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
	from lime.lime_tabular import LimeTabularExplainer
except Exception:  # pragma: no cover
	LimeTabularExplainer = None

LOGGER = get_logger(__name__)


def _save_figure(path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	plt.tight_layout()
	plt.savefig(path, bbox_inches="tight")
	plt.close()


def generate_lime_explanations(
	x_test,
	bundle: ModelBundle | None = None,
	settings: AppConfig | None = None,
) -> list[Path]:
	settings = settings or get_settings()
	bundle = bundle or load_model_bundle(settings)
	if bundle.model is None or bundle.preprocessing_pipeline is None:
		raise RuntimeError("Champion model and preprocessing pipeline must exist before explainability runs")

	engineered = engineer_features(x_test)
	transformed = bundle.preprocessing_pipeline.transform(engineered)
	transformed = np.asarray(transformed)
	feature_names = getattr(bundle.preprocessing_pipeline.named_steps["preprocessor"], "get_feature_names_out", lambda: None)()
	if feature_names is None:
		feature_names = [f"feature_{index}" for index in range(transformed.shape[1])]

	sample_indices = list(range(min(5, len(transformed))))
	random.Random(settings.project.random_seed).shuffle(sample_indices)

	if hasattr(bundle.model, "feature_importances_"):
		importance = np.asarray(bundle.model.feature_importances_, dtype=float)
	elif hasattr(bundle.model, "coef_"):
		importance = np.abs(np.asarray(bundle.model.coef_, dtype=float)).reshape(-1)
	else:
		importance = np.ones(transformed.shape[1], dtype=float)

	explainer = None
	if LimeTabularExplainer is not None:
		explainer = LimeTabularExplainer(
			training_data=transformed,
			feature_names=list(feature_names),
			mode="regression",
			random_state=settings.project.random_seed,
		)
	else:
		LOGGER.warning("lime package unavailable; generating deterministic fallback local explanations")

	output_paths: list[Path] = []
	for sample_number, sample_index in enumerate(sample_indices[:5]):
		html_path = settings.paths.reports / f"lime_explanation_sample_{sample_number}.html"
		png_path = settings.paths.reports / f"lime_explanation_sample_{sample_number}.png"
		html_path.parent.mkdir(parents=True, exist_ok=True)

		if explainer is not None:
			explanation = explainer.explain_instance(
				transformed[sample_index],
				bundle.model.predict,
				num_features=min(10, len(feature_names)),
			)
			explanation.save_to_file(str(html_path))
			explanation.as_pyplot_figure()
		else:
			sample_values = transformed[sample_index]
			contributions = sample_values * importance
			top = np.argsort(np.abs(contributions))[::-1][: min(10, len(feature_names))]
			lines = [
				"<html><body><h2>Fallback Local Explanation</h2><ul>",
			]
			for index in top:
				lines.append(
					f"<li>{feature_names[index]}: contribution={contributions[index]:.6f}</li>"
				)
			lines.append("</ul></body></html>")
			html_path.write_text("\n".join(lines), encoding="utf-8")

			plt.figure(figsize=(10, 6))
			plt.barh(
				[str(feature_names[index]) for index in top][::-1],
				contributions[top][::-1],
			)
			plt.xlabel("Contribution")
			plt.title(f"Fallback local explanation sample {sample_number}")

		_save_figure(png_path)
		output_paths.extend([html_path, png_path])

		if mlflow is not None:
			mlflow.log_artifact(str(html_path), artifact_path="explainability")
			mlflow.log_artifact(str(png_path), artifact_path="explainability")

	LOGGER.info("Saved LIME artefacts to %s", settings.paths.reports)
	return output_paths
