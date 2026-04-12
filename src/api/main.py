"""FastAPI application entry point."""

from __future__ import annotations

from fastapi import FastAPI

from .router import router

app = FastAPI(
	title="Walmart Weekly Sales MLOps",
	description="API for weekly sales prediction.",
	version="1.0.0",
)
app.include_router(router, prefix="/api/v1")
