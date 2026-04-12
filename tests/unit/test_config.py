"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

from src.config import get_settings


def test_settings_loads_and_paths_are_path_objects():
	get_settings.cache_clear()
	settings = get_settings()
	assert settings.project.name == "walmart-sales-mlops"
	assert isinstance(settings.paths.data_raw, Path)
	assert isinstance(settings.paths.data_processed, Path)
	assert isinstance(settings.paths.data_splits, Path)
	assert isinstance(settings.paths.models, Path)
	assert isinstance(settings.paths.reports, Path)
