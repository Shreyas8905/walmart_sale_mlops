"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..config import get_settings
from ..logger import configure_logging, get_logger
from .router import router

LOGGER = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()
    app.state.settings = settings
    try:
        from ..models.trainer import load_model_bundle

        app.state.model_bundle = load_model_bundle(settings)
    except ModuleNotFoundError:
        app.state.model_bundle = None
    yield


app = FastAPI(
    title=get_settings().project.name,
    description="API for weekly sales prediction.",
    version=get_settings().project.version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid4()))
    request.state.request_id = request_id
    started = perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:  # pragma: no cover - defensive runtime path
        LOGGER.exception(
            "Unhandled request error request_id=%s",
            request_id,
            extra={"request_id": request_id},
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "request_id": request_id,
                "detail": str(exc),
            },
        )
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{(perf_counter() - started) * 1000.0:.3f}"
    return response


app.include_router(router, prefix="/api/v1")
