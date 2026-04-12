# Walmart Weekly Sales MLOps

Production-ready MLOps scaffold for predicting `Weekly_Sales` on the Walmart Sales dataset.

## Overview

This repository is structured as a modular regression pipeline with explicit layers for data loading, preprocessing, model training, evaluation, explainability, and FastAPI serving.

## Repository Layout

- `configs/` for YAML configuration and logging setup
- `src/` for the main application packages
- `pipelines/` for CLI entry points
- `tests/` for unit, integration, and API coverage
- `models/`, `mlruns/`, and `reports/` for generated artefacts

## Quick Start

1. Install dependencies with `make install`.
2. Train the pipeline with `make train`.
3. Start the API with `make serve`.

## Docker

Use `docker-compose up --build` to start the API, trainer, and MLflow services together.
