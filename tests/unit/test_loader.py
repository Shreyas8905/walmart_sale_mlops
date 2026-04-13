"""Tests for the data loader."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.loader import load_raw_data, validate_schema


def test_validate_schema_success(sample_df):
    validate_schema(sample_df)


def test_validate_schema_missing_column(sample_df):
    broken = sample_df.drop(columns=["CPI"])
    with pytest.raises(ValueError, match="Missing expected columns"):
        validate_schema(broken)


def test_validate_schema_dtype_failure(sample_df):
    broken = sample_df.copy()
    broken["Temperature"] = broken["Temperature"].astype(str)
    with pytest.raises(ValueError, match="must be numeric"):
        validate_schema(broken)


def test_load_raw_data_round_trip(tmp_path, sample_df):
    csv_path = tmp_path / "Walmart_Sales.csv"
    sample_df.to_csv(csv_path, index=False)
    loaded = load_raw_data(csv_path)
    assert isinstance(loaded, pd.DataFrame)
    assert loaded.shape == sample_df.shape
