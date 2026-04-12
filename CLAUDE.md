# CLAUDE.md — Walmart Weekly Sales MLOps Project

## Project Brief

Build a **production-ready MLOps project** for predicting `Weekly_Sales` (regression) using the Walmart Sales dataset. Follow every instruction in this file exactly, in order. Commit after every major milestone — but **always show the proposed commit message to the user and wait for approval before running `git commit`**.

---

## Dataset

- **File:** `data/raw/Walmart_Sales.csv` (already placed there — do not move or alter it)
- **Target variable:** `Weekly_Sales` (continuous, regression)
- **Features:** `Store`, `Date`, `Holiday_Flag`, `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`
- **Shape:** 6,435 rows × 8 columns

---

## Guiding Principles (follow throughout)

- Clean, modular, well-documented Python code (PEP 8, type hints, docstrings)
- Every module has a single responsibility
- No magic numbers — all constants live in config files
- No hard-coded paths — all paths resolved through the config layer
- Logging over print statements everywhere
- Fail loudly with meaningful error messages
- Keep all secrets/credentials out of source code — use `.env`

---

## Step 0 — Git Initialisation

```bash
git init
git add .gitignore README.md   # created in Step 1
```

> Propose commit message to user, wait for approval, then commit.

---

## Step 1 — Project Skeleton

Create the following directory and file structure **exactly**:

```
walmart-sales-mlops/
├── CLAUDE.md                         # this file
├── README.md                         # project overview (write a proper one)
├── .gitignore                        # Python, MLflow, env, __pycache__, etc.
├── .env.example                      # template — never commit real .env
├── .env                              # gitignored — actual secrets/settings
├── pyproject.toml                    # project metadata + tool config (black, isort, mypy, pytest)
├── requirements.txt                  # pinned runtime deps
├── requirements-dev.txt              # dev/test deps (pytest, black, isort, mypy, shap, lime, etc.)
├── Makefile                          # convenience targets: train, test, serve, docker-build, etc.
│
├── configs/
│   ├── config.yaml                   # all directory paths, MLflow settings, model params grid
│   └── logging_config.yaml           # logging format, handlers, levels
│
├── data/
│   ├── raw/                          # Walmart_Sales.csv lives here (read-only)
│   ├── processed/                    # cleaned + feature-engineered data (gitignored)
│   └── splits/                       # train/val/test splits (gitignored)
│
├── src/
│   ├── __init__.py
│   ├── config.py                     # loads config.yaml + .env, exposes typed Settings object
│   ├── logger.py                     # project-wide logger factory
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                 # reads raw CSV, validates schema
│   │   ├── preprocessor.py           # feature engineering, encoding, scaling (sklearn Pipeline)
│   │   └── splitter.py               # train/val/test split logic
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── registry.py               # maps model names → (estimator class, param grid)
│   │   └── trainer.py                # experiment loop: trains all models, logs to MLflow, pickles
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py                # RMSE, MAE, R², MAPE; champion selection logic
│   │
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── shap_explainer.py         # SHAP values + summary/force/waterfall plots
│   │   └── lime_explainer.py         # LIME tabular explanations
│   │
│   └── api/
│       ├── __init__.py
│       ├── main.py                   # FastAPI app factory
│       ├── schemas.py                # Pydantic request/response models
│       └── router.py                 # /predict (single) and /predict/batch endpoints
│
├── pipelines/
│   ├── __init__.py
│   ├── train_pipeline.py             # end-to-end: ingest → preprocess → train → select champion
│   └── explain_pipeline.py           # load champion → run SHAP + LIME → save artefacts
│
├── models/                           # pickled artefacts (gitignored)
│   ├── preprocessing_pipeline.pkl
│   └── champion_model.pkl
│
├── mlruns/                           # MLflow tracking directory (gitignored)
│
├── reports/
│   └── explainability/               # SHAP/LIME plots saved here
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                   # shared fixtures (sample df, loaded config, etc.)
│   ├── unit/
│   │   ├── test_loader.py
│   │   ├── test_preprocessor.py
│   │   ├── test_metrics.py
│   │   ├── test_schemas.py
│   │   └── test_config.py
│   ├── integration/
│   │   ├── test_train_pipeline.py
│   │   └── test_explain_pipeline.py
│   └── api/
│       └── test_endpoints.py         # TestClient tests for /predict + /predict/batch
│
├── Dockerfile
├── docker-compose.yml
└── .github/
    └── workflows/
        └── ci.yml
```

After creating the skeleton, propose a commit message and wait for user approval.

---

## Step 2 — Configuration Layer

### `configs/config.yaml`

```yaml
project:
  name: walmart-sales-mlops
  version: "1.0.0"
  target: Weekly_Sales
  random_seed: 42

paths:
  data_raw: data/raw
  data_processed: data/processed
  data_splits: data/splits
  models: models
  reports: reports/explainability

mlflow:
  tracking_uri: mlruns          # local directory; override via .env for remote
  experiment_name: walmart-weekly-sales

data:
  test_size: 0.15
  val_size: 0.15
  date_column: Date
  date_format: "%d-%m-%Y"
  categorical_features:
    - Store
    - Holiday_Flag
  numerical_features:
    - Temperature
    - Fuel_Price
    - CPI
    - Unemployment
    - Week
    - Month
    - Year
    - DayOfYear

models:
  ridge:
    alpha: [0.01, 0.1, 1.0, 10.0, 100.0]
  lasso:
    alpha: [0.01, 0.1, 1.0, 10.0, 100.0]
  elastic_net:
    alpha: [0.01, 0.1, 1.0]
    l1_ratio: [0.2, 0.5, 0.8]
  random_forest:
    n_estimators: [100, 200]
    max_depth: [None, 10, 20]
    min_samples_split: [2, 5]
  gradient_boosting:
    n_estimators: [100, 200]
    learning_rate: [0.05, 0.1]
    max_depth: [3, 5]
  xgboost:
    n_estimators: [100, 200]
    learning_rate: [0.05, 0.1]
    max_depth: [3, 6]
    reg_alpha: [0.0, 0.1]    # L1
    reg_lambda: [1.0, 10.0]  # L2

champion_selection:
  metric: rmse          # lower is better
  direction: minimize
```

### `configs/logging_config.yaml`

Configure:
- Console handler (INFO level, coloured if possible)
- File handler → `logs/app.log` (DEBUG level, rotating, max 10 MB × 5 backups)
- JSON structured logging for the file handler

### `.env.example`

```dotenv
MLFLOW_TRACKING_URI=mlruns
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

### `src/config.py`

- Use **Pydantic `BaseSettings`** (pydantic-settings) to load `.env`
- Use **PyYAML** to load `configs/config.yaml`
- Expose a single `get_settings()` function (cached with `functools.lru_cache`) and a `load_config()` function
- All path values must be returned as `pathlib.Path` objects and created if they don't exist

---

## Step 3 — Data Layer

### `src/data/loader.py`

- Read `Walmart_Sales.csv` with pandas
- Validate that all expected columns exist; raise `ValueError` if not
- Validate column dtypes
- Log row/column counts
- Return a `pd.DataFrame`

### `src/data/preprocessor.py`

**Feature engineering (create these columns before any sklearn Pipeline step):**
- Parse `Date` column using the format from config → extract `Week`, `Month`, `Year`, `DayOfYear`
- Drop the original `Date` column

**sklearn `Pipeline` / `ColumnTransformer` steps:**
- Numerical features: `SimpleImputer(strategy='median')` → `StandardScaler`
- Categorical features (Store, Holiday_Flag): `SimpleImputer(strategy='most_frequent')` → `OrdinalEncoder`

**Expose:**
```python
def build_preprocessing_pipeline() -> Pipeline: ...
def fit_transform_pipeline(pipeline, X_train) -> np.ndarray: ...
def transform_pipeline(pipeline, X) -> np.ndarray: ...
def save_pipeline(pipeline, path: Path) -> None: ...   # pickle
def load_pipeline(path: Path) -> Pipeline: ...          # unpickle
```

### `src/data/splitter.py`

- Temporal split (sort by Date before splitting — no data leakage)
- Return `(X_train, X_val, X_test, y_train, y_val, y_test)` all as `pd.DataFrame`/`pd.Series`
- Save splits to `data/splits/` as parquet files
- Log split sizes

---

## Step 4 — Model Registry & Training

### `src/models/registry.py`

Register every model and its hyperparameter grid from `config.yaml`:

| Key | Estimator | Regularisation |
|---|---|---|
| `ridge` | `Ridge` | L2 |
| `lasso` | `Lasso` | L1 |
| `elastic_net` | `ElasticNet` | L1 + L2 |
| `random_forest` | `RandomForestRegressor` | Implicit (tree depth) |
| `gradient_boosting` | `GradientBoostingRegressor` | Shrinkage + depth |
| `xgboost` | `XGBRegressor` | L1 (`reg_alpha`) + L2 (`reg_lambda`) |

Return a `ModelSpec` dataclass with `name`, `estimator_cls`, `param_grid`.

### `src/models/trainer.py`

For **every model × every hyperparameter combination**:

1. Start an **MLflow run** (nested under one parent experiment run)
2. Train on `X_train` / `y_train`
3. Evaluate on `X_val` → compute RMSE, MAE, R², MAPE
4. Log all hyperparameters + metrics to MLflow
5. Log the fitted model as an MLflow artefact
6. Store results in a list

After the loop:
- Select the **champion** (lowest val RMSE)
- Log champion metadata to MLflow (tags)
- Pickle champion model → `models/champion_model.pkl`
- Pickle preprocessing pipeline → `models/preprocessing_pipeline.pkl`
- Log all pickled files as MLflow artefacts
- Final evaluation on `X_test` → log test metrics
- Log a `training_summary.json` with all run results to `reports/`

---

## Step 5 — Evaluation

### `src/evaluation/metrics.py`

```python
def rmse(y_true, y_pred) -> float: ...
def mae(y_true, y_pred) -> float: ...
def r2(y_true, y_pred) -> float: ...
def mape(y_true, y_pred) -> float: ...
def compute_all_metrics(y_true, y_pred) -> dict[str, float]: ...
def select_champion(results: list[dict], metric: str, direction: str) -> dict: ...
```

---

## Step 6 — Explainability

### `src/explainability/shap_explainer.py`

- Load champion model + preprocessing pipeline
- Compute SHAP values (use `TreeExplainer` for tree models, `LinearExplainer` for linear, `KernelExplainer` as fallback)
- Save to `reports/explainability/`:
  - `shap_summary_bar.png`
  - `shap_summary_beeswarm.png`
  - `shap_waterfall_sample_0.png` (first test sample)
  - `shap_values.npy` (raw values)
- Log all artefacts to MLflow

### `src/explainability/lime_explainer.py`

- Use `lime.lime_tabular.LimeTabularExplainer`
- Explain 5 random test samples
- Save `lime_explanation_sample_{i}.html` and `.png` for each
- Log to MLflow

### `pipelines/explain_pipeline.py`

Orchestrates both explainers end-to-end; callable from CLI.

---

## Step 7 — FastAPI Service

### `src/api/schemas.py`

```python
class PredictionRequest(BaseModel):
    Store: int
    Date: str               # format DD-MM-YYYY
    Holiday_Flag: int       # 0 or 1
    Temperature: float
    Fuel_Price: float
    CPI: float
    Unemployment: float

class PredictionResponse(BaseModel):
    predicted_weekly_sales: float
    model_version: str
    prediction_id: str      # UUID

class BatchPredictionRequest(BaseModel):
    records: list[PredictionRequest]
    max_records: ClassVar[int] = 1000

class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    total_records: int
    processing_time_ms: float
```

### `src/api/router.py`

| Method | Path | Description |
|---|---|---|
| `POST` | `/predict` | Single record prediction |
| `POST` | `/predict/batch` | Batch predictions (≤ 1000 records) |
| `GET` | `/health` | Liveness check |
| `GET` | `/model/info` | Champion model metadata |

- Load model + pipeline once at startup using FastAPI **lifespan** context manager (not deprecated `@app.on_event`)
- Return HTTP 422 for validation errors, 500 with structured error body for runtime errors
- Log every prediction request/response with a unique `request_id`
- Add response time middleware

### `src/api/main.py`

- Create FastAPI app with title, description, version from config
- Mount router with `/api/v1` prefix
- Add CORS middleware (configurable origins)
- Add request ID middleware
- Swagger UI available at `/docs`

---

## Step 8 — Pipelines (CLI entry points)

### `pipelines/train_pipeline.py`

```
python -m pipelines.train_pipeline
```

Runs: load → engineer features → split → fit preprocessing → train all models → evaluate → select champion → pickle → explain

Exit code 0 on success, 1 on failure. Log everything.

### `pipelines/explain_pipeline.py`

```
python -m pipelines.explain_pipeline
```

Runs SHAP + LIME on the already-trained champion. Can be run independently.

---

## Step 9 — Tests

### General rules

- Use `pytest` with `pytest-cov`
- Minimum 80% line coverage (enforced in CI)
- All tests must be deterministic (use `random_seed` from config)
- Test results logged to `logs/test_results.log` (configure in `conftest.py` with a `pytest` plugin or fixture)
- Use `pytest-json-report` to also emit `reports/test_report.json`

### `tests/conftest.py`

Provide shared fixtures:
- `sample_df` — 50-row synthetic DataFrame matching the Walmart schema
- `config` — loaded config object
- `preprocessing_pipeline` — a fitted pipeline on `sample_df`
- `app_client` — FastAPI `TestClient`

### Unit tests

| File | What to test |
|---|---|
| `test_loader.py` | Schema validation, missing column error, dtype checks |
| `test_preprocessor.py` | Feature engineering output columns, pipeline transform shape, pickle round-trip |
| `test_metrics.py` | RMSE/MAE/R²/MAPE against known values, champion selection logic |
| `test_schemas.py` | Pydantic validation — valid input, missing fields, wrong types, out-of-range |
| `test_config.py` | Settings loads correctly, paths are `Path` objects |

### Integration tests

| File | What to test |
|---|---|
| `test_train_pipeline.py` | Full pipeline run on `sample_df`; champion pickle exists after run; MLflow run created |
| `test_explain_pipeline.py` | SHAP/LIME run without error; output files created |

### API tests (`tests/api/test_endpoints.py`)

- `GET /health` → 200
- `GET /model/info` → 200, has `model_name` and `version`
- `POST /predict` valid payload → 200, response has `predicted_weekly_sales`
- `POST /predict` missing field → 422
- `POST /predict` wrong type → 422
- `POST /predict/batch` valid 5-record batch → 200, `total_records == 5`
- `POST /predict/batch` empty records → 422
- `POST /predict/batch` > 1000 records → 422

---

## Step 10 — Dockerfile & Docker Compose

### `Dockerfile`

```dockerfile
# Multi-stage build
# Stage 1: builder — install deps
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: runtime
FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /install /usr/local
COPY src/ src/
COPY configs/ configs/
COPY models/ models/        # champion_model.pkl + preprocessing_pipeline.pkl
COPY .env.example .env      # safe defaults; override at runtime
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `docker-compose.yml`

Services:

| Service | Image | Purpose |
|---|---|---|
| `api` | built from `Dockerfile` | FastAPI prediction service |
| `mlflow` | `ghcr.io/mlflow/mlflow:latest` | MLflow tracking UI |
| `trainer` | same build context | Runs `train_pipeline.py` then exits |

- `api` depends_on `trainer` (for models to exist) or mounts a pre-built `models/` volume
- `mlflow` exposes port 5000
- `api` exposes port 8000
- Use a named volume for `mlruns/` shared between services
- Single command to run everything:

```bash
docker-compose up --build
```

---

## Step 11 — Makefile

Provide at minimum:

```makefile
install:        pip install -r requirements.txt -r requirements-dev.txt
train:          python -m pipelines.train_pipeline
explain:        python -m pipelines.explain_pipeline
serve:          uvicorn src.api.main:app --reload --port 8000
test:           pytest tests/ -v --cov=src --cov-report=term-missing
test-unit:      pytest tests/unit/ -v
test-integration: pytest tests/integration/ -v
test-api:       pytest tests/api/ -v
lint:           black src/ tests/ && isort src/ tests/ && mypy src/
docker-build:   docker build -t walmart-sales-mlops:latest .
docker-up:      docker-compose up --build
docker-down:    docker-compose down -v
clean:          find . -type d -name __pycache__ -exec rm -rf {} +
```

---

## Step 12 — GitHub Actions CI

### `.github/workflows/ci.yml`

Trigger: `push` and `pull_request` on `main` and `develop` branches.

Jobs (run in order):

#### `lint`
- `actions/checkout`
- Setup Python 3.11
- Install dev deps
- Run `black --check`, `isort --check`, `mypy`

#### `test`
- `actions/checkout`
- Setup Python 3.11
- Cache pip
- Install all deps
- Run `pytest tests/unit tests/api -v --cov=src --cov-report=xml --json-report --json-report-file=reports/test_report.json`
- Upload coverage to Codecov
- Upload `reports/test_report.json` as a workflow artefact

#### `build-and-push`
- Depends on `lint` and `test` passing
- `docker/setup-buildx-action`
- `docker/login-action` (secrets: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`)
- `docker/build-push-action`
  - Push only on `main` branch
  - Tags: `<dockerhub_username>/walmart-sales-mlops:latest` and `:<github_sha>`

> **Note:** CD is intentionally not implemented. Deployment is a separate concern.

---

## Step 13 — README.md

Write a proper README with:
- Project overview and architecture diagram (ASCII is fine)
- Prerequisites (Python 3.11+, Docker)
- Quick start (clone → `make install` → `make train` → `make serve`)
- Docker quick start (`docker-compose up --build`)
- API reference (endpoints, example curl commands)
- MLflow UI instructions
- Running tests
- Project structure tree
- CI/CD badge placeholder

---

## Commit Checkpoints

After each step below, **show the user the proposed commit message and wait for their approval** before running `git commit -m "..."`:

| After Step | Suggested message (show to user, they may edit) |
|---|---|
| 0 | `chore: initialise git repository with .gitignore and README skeleton` |
| 1 | `feat: scaffold full project directory structure` |
| 2 | `feat: add configuration layer (config.yaml, logging, pydantic settings)` |
| 3 | `feat: implement data loading, feature engineering, and train/val/test splitting` |
| 4 | `feat: implement model registry and MLflow experiment training loop` |
| 5 | `feat: add evaluation metrics and champion model selection` |
| 6 | `feat: add SHAP and LIME explainability pipelines` |
| 7 | `feat: implement FastAPI prediction service (single + batch endpoints)` |
| 8 | `feat: add end-to-end train and explain pipeline CLI scripts` |
| 9 | `test: add unit, integration, and API test suites with coverage reporting` |
| 10 | `feat: dockerise application with multi-stage Dockerfile and docker-compose` |
| 11 | `chore: add Makefile convenience targets` |
| 12 | `ci: add GitHub Actions workflow (lint, test, build-and-push)` |
| 13 | `docs: write comprehensive README` |
| Final | `chore: final cleanup, pin all dependencies, verify docker-compose up` |

---

## Definition of Done

- [ ] `make train` runs to completion, champion model pickled, MLflow UI shows all runs
- [ ] `make explain` generates SHAP + LIME plots in `reports/explainability/`
- [ ] `make serve` starts the API; `curl localhost:8000/docs` shows Swagger UI
- [ ] `make test` passes with ≥ 80% coverage
- [ ] `docker-compose up --build` starts all services in one command
- [ ] GitHub Actions CI pipeline passes on push to `main`
- [ ] All commits have user-approved messages