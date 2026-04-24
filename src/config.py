"""Configuration helpers for the Walmart sales project."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"


class ProjectConfig(BaseModel):
    name: str
    version: str
    target: str
    random_seed: int


class PathsConfig(BaseModel):
    data_raw: Path
    data_processed: Path
    data_splits: Path
    models: Path
    reports: Path


class MLflowConfig(BaseModel):
    tracking_uri: str
    experiment_name: str


class DataConfig(BaseModel):
    test_size: float
    val_size: float
    date_column: str
    date_format: str
    categorical_features: list[str]
    numerical_features: list[str]


class ChampionSelectionConfig(BaseModel):
    metric: str
    direction: Literal["minimize", "maximize"]


class AppConfig(BaseModel):
    project: ProjectConfig
    paths: PathsConfig
    mlflow: MLflowConfig
    data: DataConfig
    models: dict[str, dict[str, list[Any]]]
    champion_selection: ChampionSelectionConfig
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


class EnvironmentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_PATH, extra="ignore")

    mlflow_tracking_uri: str = "mlruns"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    cors_origins: str = "*"


def _ensure_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_yaml_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle) or {}


def _resolve_mlflow_tracking_uri(uri_value: str | Path) -> str:
    raw_uri = str(uri_value).strip()
    if not raw_uri:
        raw_uri = "mlruns"

    # Windows drive paths like D:\mlruns should be treated as local file paths,
    # not as URI schemes.
    if len(raw_uri) >= 2 and raw_uri[1] == ":" and raw_uri[0].isalpha():
        return _ensure_path(raw_uri).as_uri()

    parsed = urlparse(raw_uri)
    if parsed.scheme in {
        "file",
        "http",
        "https",
        "databricks",
        "databricks-uc",
        "uc",
        "postgresql",
        "mysql",
        "sqlite",
        "mssql",
    }:
        return raw_uri

    return _ensure_path(raw_uri).as_uri()


def load_config() -> AppConfig:
    """Load the YAML and environment configuration into a typed object."""

    raw_config = _load_yaml_config()
    env_settings = EnvironmentSettings()

    paths_section = raw_config.get("paths", {})
    mlflow_section = raw_config.get("mlflow", {})

    data_paths = PathsConfig(
        data_raw=_ensure_path(paths_section.get("data_raw", "data/raw")),
        data_processed=_ensure_path(
            paths_section.get("data_processed", "data/processed")
        ),
        data_splits=_ensure_path(paths_section.get("data_splits", "data/splits")),
        models=_ensure_path(paths_section.get("models", "models")),
        reports=_ensure_path(paths_section.get("reports", "reports/explainability")),
    )

    mlflow_tracking_uri = _resolve_mlflow_tracking_uri(
        env_settings.mlflow_tracking_uri or mlflow_section.get("tracking_uri", "mlruns")
    )
    _ensure_path(PROJECT_ROOT / "logs")

    cors_origins = [
        origin.strip()
        for origin in env_settings.cors_origins.split(",")
        if origin.strip()
    ]
    if not cors_origins:
        cors_origins = ["*"]

    return AppConfig(
        project=ProjectConfig(**raw_config["project"]),
        paths=data_paths,
        mlflow=MLflowConfig(
            tracking_uri=mlflow_tracking_uri,
            experiment_name=mlflow_section.get(
                "experiment_name", "walmart-weekly-sales"
            ),
        ),
        data=DataConfig(**raw_config["data"]),
        models=raw_config["models"],
        champion_selection=ChampionSelectionConfig(**raw_config["champion_selection"]),
        api_host=env_settings.api_host,
        api_port=env_settings.api_port,
        log_level=env_settings.log_level,
        cors_origins=cors_origins,
    )


@lru_cache(maxsize=1)
def get_settings() -> AppConfig:
    return load_config()
