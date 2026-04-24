install:
	pip install -r requirements.txt -r requirements-dev.txt

train:
	python -m pipelines.train_pipeline

mlflow-ui:
	mlflow ui --backend-store-uri mlruns --host 0.0.0.0 --port 5000

explain:
	python -m pipelines.explain_pipeline

serve:
	uvicorn src.api.main:app --reload --port 8000

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-api:
	pytest tests/api/ -v

lint:
	black src/ tests/
	isort src/ tests/
	mypy src/

docker-build:
	docker build -t walmart-sales-mlops:latest .

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
