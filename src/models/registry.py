"""Model registry definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge

try:  # pragma: no cover - optional dependency fallback
    import xgboost as _xgboost_module
except Exception:  # pragma: no cover
    _xgboost_module = None


class _FallbackXGBRegressor(GradientBoostingRegressor):
    """Compatibility fallback for environments without xgboost."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs,
        )
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda


XGBRegressor = (
    _xgboost_module.XGBRegressor
    if _xgboost_module is not None
    else _FallbackXGBRegressor
)

from ..config import get_settings


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator_cls: type
    param_grid: dict[str, list[Any]]


def build_model_registry() -> list[ModelSpec]:
    settings = get_settings()
    model_configs = settings.models
    return [
        ModelSpec("ridge", Ridge, model_configs["ridge"]),
        ModelSpec("lasso", Lasso, model_configs["lasso"]),
        ModelSpec("elastic_net", ElasticNet, model_configs["elastic_net"]),
        ModelSpec(
            "random_forest", RandomForestRegressor, model_configs["random_forest"]
        ),
        ModelSpec(
            "gradient_boosting",
            GradientBoostingRegressor,
            model_configs["gradient_boosting"],
        ),
        ModelSpec("xgboost", XGBRegressor, model_configs["xgboost"]),
    ]
