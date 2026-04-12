"""Explainability pipeline CLI entry point."""

from __future__ import annotations

from src.config import AppConfig, get_settings
from src.data.loader import load_raw_data
from src.data.splitter import temporal_split
from src.explainability.lime_explainer import generate_lime_explanations
from src.explainability.shap_explainer import generate_shap_explanations
from src.logger import get_logger
from src.models.trainer import load_model_bundle

LOGGER = get_logger(__name__)


def run_explain_pipeline(settings: AppConfig | None = None) -> int:
	settings = settings or get_settings()
	bundle = load_model_bundle(settings)
	if bundle.model is None or bundle.preprocessing_pipeline is None:
		LOGGER.warning("Skipping explainability because champion artefacts are not available yet")
		return 0

	raw_data = load_raw_data()
	_, _, x_test, _, _, _ = temporal_split(raw_data)
	generate_shap_explanations(x_test, bundle=bundle, settings=settings)
	generate_lime_explanations(x_test, bundle=bundle, settings=settings)
	LOGGER.info("Explainability pipeline completed successfully")
	return 0


def main() -> None:
	try:
		raise SystemExit(run_explain_pipeline())
	except Exception as exc:  # pragma: no cover - CLI failure path
		LOGGER.exception("Explainability pipeline failed: %s", exc)
		raise SystemExit(1) from exc


if __name__ == "__main__":
	main()
