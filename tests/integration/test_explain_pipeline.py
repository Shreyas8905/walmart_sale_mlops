"""Integration tests for the explainability pipeline."""

from __future__ import annotations

from pipelines.explain_pipeline import run_explain_pipeline


def test_explain_pipeline_runs_without_error():
    exit_code = run_explain_pipeline()
    assert exit_code == 0
