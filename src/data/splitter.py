"""Temporal dataset splitting utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import get_settings
from ..logger import get_logger

LOGGER = get_logger(__name__)


def _save_frame(frame: pd.DataFrame | pd.Series, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(frame, pd.Series):
        frame.to_frame(name=frame.name or "value").to_parquet(path, index=False)
    else:
        frame.to_parquet(path, index=False)


def temporal_split(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    settings = get_settings()
    date_column = settings.data.date_column
    target_column = settings.project.target

    frame_with_date = frame.copy()
    frame_with_date["__parsed_date"] = pd.to_datetime(
        frame_with_date[date_column],
        format=settings.data.date_format,
        errors="raise",
    )
    sorted_frame = (
        frame_with_date.sort_values("__parsed_date")
        .drop(columns=["__parsed_date"])
        .reset_index(drop=True)
    )

    total_rows = len(sorted_frame)
    test_rows = max(1, int(round(total_rows * settings.data.test_size)))
    val_rows = max(1, int(round(total_rows * settings.data.val_size)))
    train_rows = total_rows - test_rows - val_rows
    if train_rows <= 0:
        raise ValueError("Not enough rows to create train/validation/test splits")

    train_frame = sorted_frame.iloc[:train_rows].copy()
    val_frame = sorted_frame.iloc[train_rows : train_rows + val_rows].copy()
    test_frame = sorted_frame.iloc[train_rows + val_rows :].copy()

    x_train = train_frame.drop(columns=[target_column])
    x_val = val_frame.drop(columns=[target_column])
    x_test = test_frame.drop(columns=[target_column])
    y_train = train_frame[target_column].copy()
    y_val = val_frame[target_column].copy()
    y_test = test_frame[target_column].copy()

    split_dir = settings.paths.data_splits
    _save_frame(x_train, split_dir / "x_train.parquet")
    _save_frame(x_val, split_dir / "x_val.parquet")
    _save_frame(x_test, split_dir / "x_test.parquet")
    _save_frame(y_train, split_dir / "y_train.parquet")
    _save_frame(y_val, split_dir / "y_val.parquet")
    _save_frame(y_test, split_dir / "y_test.parquet")

    LOGGER.info(
        "Split data into train=%s, val=%s, test=%s rows",
        len(x_train),
        len(x_val),
        len(x_test),
    )
    return x_train, x_val, x_test, y_train, y_val, y_test
