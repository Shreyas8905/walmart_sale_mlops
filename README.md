# Walmart Weekly Sales MLOps

[![CI Status Placeholder](https://img.shields.io/badge/ci-pending-lightgrey.svg)](#github-actions-ci)

Production-ready MLOps project for predicting Walmart `Weekly_Sales` using a modular, test-driven machine learning system.

## What This Project Delivers

This repository implements a complete machine learning operations workflow:

1. Configuration-driven data ingestion and validation.
2. Temporal splitting to prevent leakage.
3. Feature engineering from dates (`Week`, `Month`, `Year`, `DayOfYear`).
4. Standardized preprocessing using sklearn `Pipeline` and `ColumnTransformer`.
5. Multi-model experimentation with hyperparameter search.
6. Automatic metric computation and champion model selection.
7. Model and pipeline persistence for reproducible inference.
8. Explainability artifact generation (SHAP/LIME-compatible flow with deterministic fallbacks).
9. FastAPI prediction service with request IDs, latency headers, health endpoints, and batch predictions.
10. Unit, integration, and API test coverage.
11. Dockerized execution and CI automation.

## Architecture

```text
														+----------------------+
														|  data/raw CSV input  |
														+----------+-----------+
																			 |
																			 v
														 +---------+---------+
														 | src.data.loader   |
														 | schema validation |
														 +---------+---------+
																			 |
																			 v
														+----------+-----------+
														| src.data.splitter    |
														| temporal split       |
														+----------+-----------+
																			 |
						 +-------------------------+--------------------------+
						 |                         |                          |
						 v                         v                          v
 +-----------+-----------+  +----------+----------+  +------------+-----------+
 | X_train / y_train     |  | X_val / y_val       |  | X_test / y_test        |
 +-----------+-----------+  +----------+----------+  +------------+-----------+
						 |                         |                          |
						 +-------------------------+--------------------------+
																			 |
																			 v
													 +-----------+------------+
													 | src.data.preprocessor  |
													 | pipeline fit/transform |
													 +-----------+------------+
																			 |
																			 v
													+------------+-------------+
													| src.models.trainer       |
													| model grid loop + MLflow |
													+------------+-------------+
																			 |
																			 v
								 +---------------------+----------------------+
								 | champion selection + persisted artifacts   |
								 | models/champion_model.pkl                 |
								 | models/preprocessing_pipeline.pkl          |
								 +---------------------+----------------------+
																			 |
								 +---------------------+----------------------+
								 |                                            |
								 v                                            v
			+----------+-------------+                   +----------+-------------+
			| pipelines.explain      |                   | FastAPI service         |
			| SHAP/LIME artifacts    |                   | /predict /predict/batch|
			+------------------------+                   +------------------------+
```

## Dataset

1. File: `data/raw/Walmart_Sales.csv`
2. Target: `Weekly_Sales`
3. Input features: `Store`, `Date`, `Holiday_Flag`, `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`
4. Expected shape: approximately 6,435 rows and 8 columns

## Tech Stack And Tools Used

1. Core language: Python 3.11+
2. Data processing: pandas, numpy
3. ML pipeline and models: scikit-learn, xgboost (with fallback compatibility path)
4. Experiment tracking: MLflow
5. Explainability: shap, lime (with deterministic fallback generation when unavailable)
6. API framework: FastAPI + Uvicorn
7. Config and validation: pydantic, pydantic-settings, PyYAML
8. Serialization and I/O: pickle, pyarrow/parquet
9. Testing: pytest, pytest-cov, pytest-json-report, httpx/TestClient
10. Code quality: black, isort, mypy
11. Containers: Docker, docker-compose
12. CI: GitHub Actions

## Detailed Module Guide

### `src/config.py`

1. Loads and validates `configs/config.yaml`.
2. Loads environment overrides from `.env` using `BaseSettings`.
3. Converts path settings to absolute `Path` objects.
4. Creates required directories automatically.
5. Exposes cached `get_settings()` for shared app-wide configuration.

### `src/logger.py`

1. Configures project-wide logging.
2. Writes structured JSON logs to `logs/app.log` using rotating file handler.
3. Writes human-readable logs to console.
4. Supports request ID enrichment.

### `src/data/loader.py`

1. Reads raw CSV data.
2. Validates expected schema.
3. Validates numeric dtypes and date column type.
4. Logs row/column metadata.

### `src/data/preprocessor.py`

1. Engineers date-derived features (`Week`, `Month`, `Year`, `DayOfYear`).
2. Drops raw `Date` after feature extraction.
3. Builds `ColumnTransformer` pipeline.
4. Numerical path: `SimpleImputer(median)` + `StandardScaler`.
5. Categorical path: `SimpleImputer(most_frequent)` + `OrdinalEncoder`.
6. Provides fit/transform utilities and pickle save/load helpers.

### `src/data/splitter.py`

1. Parses and sorts by date for true temporal split.
2. Produces train/validation/test partitions.
3. Saves all split artifacts as parquet files in `data/splits/`.
4. Logs split sizes.

### `src/evaluation/metrics.py`

1. Computes RMSE, MAE, R2, and MAPE.
2. Provides `compute_all_metrics()` helper.
3. Implements generic champion selection based on configurable metric direction.

### `src/models/registry.py`

Registers all supported regressors and their parameter grids:

1. Ridge
2. Lasso
3. ElasticNet
4. RandomForestRegressor
5. GradientBoostingRegressor
6. XGBRegressor (with sklearn-compatible fallback class if xgboost import is unavailable)

### `src/models/trainer.py`

1. Iterates through every model and hyperparameter combination.
2. Trains and evaluates each candidate on validation data.
3. Logs parameters and metrics to MLflow when available.
4. Selects champion model by configured metric.
5. Saves champion model and preprocessing pipeline pickles.
6. Evaluates champion on test set and writes summary report.
7. Exposes loading utility for inference/runtime use.

### `src/explainability/shap_explainer.py`

1. Loads champion and transformed test data.
2. Generates SHAP-compatible artifacts:
   `shap_values.npy`, summary bar, beeswarm-style plot, sample waterfall-style plot.
3. Uses deterministic fallback contribution approximation if SHAP is unavailable.
4. Logs artifacts to MLflow when enabled.

### `src/explainability/lime_explainer.py`

1. Generates local explanations for five test samples.
2. Saves HTML and PNG explanation artifacts.
3. Uses deterministic fallback local contribution reports if LIME is unavailable.
4. Logs artifacts to MLflow when enabled.

### `src/api/schemas.py`

1. Request/response contracts for single and batch prediction.
2. Input validation for types, date format, and batch size constraints.

### `src/api/router.py`

1. Implements:
   `/health`, `/model/info`, `/predict`, `/predict/batch`.
2. Uses loaded champion model for inference when available.
3. Falls back to deterministic baseline scoring if model artifacts are absent.
4. Logs request/response details with request ID context.

### `src/api/main.py`

1. Creates FastAPI app with lifespan context loading.
2. Loads settings and model bundle once at startup.
3. Adds CORS middleware.
4. Adds request ID and response timing middleware.
5. Returns structured 500 response payloads on unhandled runtime errors.

### `pipelines/train_pipeline.py`

End-to-end orchestrator:

1. Load raw data
2. Temporal split
3. Feature engineering
4. Preprocessing fit/transform
5. Model training and champion selection
6. Explainability pipeline trigger

### `pipelines/explain_pipeline.py`

Standalone explainability orchestrator for already-trained artifacts.

## API Reference

Base prefix: `/api/v1`

1. `GET /health`
   Returns service liveness status.
2. `GET /model/info`
   Returns active model metadata (name/version).
3. `POST /predict`
   Single-record prediction.
4. `POST /predict/batch`
   Batch prediction (max 1000 records).

### Example: Single Prediction

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
	-H "Content-Type: application/json" \
	-d '{
		"Store": 1,
		"Date": "05-02-2010",
		"Holiday_Flag": 0,
		"Temperature": 42.31,
		"Fuel_Price": 2.572,
		"CPI": 211.0963582,
		"Unemployment": 8.106
	}'
```

### Example: Batch Prediction

```bash
curl -X POST "http://localhost:8000/api/v1/predict/batch" \
	-H "Content-Type: application/json" \
	-d '{
		"records": [
			{
				"Store": 1,
				"Date": "05-02-2010",
				"Holiday_Flag": 0,
				"Temperature": 42.31,
				"Fuel_Price": 2.572,
				"CPI": 211.0963582,
				"Unemployment": 8.106
			}
		]
	}'
```

## Prerequisites

1. Python 3.11+
2. pip
3. Make
4. Docker and docker-compose (for containerized mode)

## Local Quick Start

```bash
make install
make train
make serve
```

Swagger UI:

1. `http://localhost:8000/docs`

## MLflow Tracking UI

Start UI locally:

```bash
mlflow ui --backend-store-uri mlruns --host 0.0.0.0 --port 5000
```

Open dashboard:

1. `http://localhost:5000`

Run comparison workflow:

1. Open the experiment `walmart-weekly-sales`.
2. In `Runs`, use table columns for params and metrics such as `alpha`, `learning_rate`, `max_depth`, `val_rmse`, `val_mae`, `val_r2`, and `test_rmse`.
3. Filter out summary rows by adding a filter on tag `run_type != summary`.
4. Select multiple trial runs and click `Compare` to view parallel coordinates and scatter-based metric comparisons.
5. To inspect only one training execution batch, filter by tag `trial_group`.
6. The best model trial is marked with tag `is_champion=true` and includes `test_*` metrics.

## Docker Quick Start

Build and run all services:

```bash
docker-compose up --build
```

Services:

1. API: `http://localhost:8000`
2. MLflow: `http://localhost:5000`
3. Trainer: one-shot training container that exits after artifact generation

## Testing

Run complete test suite:

```bash
make test
```

Run targeted suites:

```bash
make test-unit
make test-integration
make test-api
```

Test artifacts:

1. `logs/test_results.log`
2. `reports/test_report.json` (when json-report option is used)

## Linting And Static Analysis

```bash
make lint
```

Includes:

1. black formatting
2. isort import ordering
3. mypy static type checks

## GitHub Actions CI

Workflow path: `.github/workflows/ci.yml`

Jobs:

1. `lint`: black check, isort check, mypy
2. `test`: pytest + coverage + json report artifact upload
3. `build-and-push`: Docker buildx and conditional push on `main`

## Repository Structure

```text
walmart-sales-mlops/
├── configs/
│   ├── config.yaml
│   └── logging_config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── pipelines/
│   ├── train_pipeline.py
│   └── explain_pipeline.py
├── src/
│   ├── config.py
│   ├── logger.py
│   ├── data/
│   ├── models/
│   ├── evaluation/
│   ├── explainability/
│   └── api/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── api/
├── models/
├── reports/explainability/
├── mlruns/
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── .github/workflows/ci.yml
```

## Notes

1. Raw dataset files are not modified by training pipelines.
2. Artifacts and logs are generated in configured output directories.
3. In environments where SHAP or LIME are unavailable, explainability fallback artifacts are still produced to keep pipelines operational.
