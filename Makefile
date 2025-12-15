.PHONY: help setup install uninstall up down restart clean logs status train api airflow mlflow minio lint format test test-cov

# Default SOCRATA token (user should override in .env)
SOCRATA_TOKEN ?= YOUR_SOCRATA_TOKEN_HERE

help:
	@echo "=========================================="
	@echo "  Chicago Crime Arrest Prediction - MLOps"
	@echo "=========================================="
	@echo ""
	@echo "Installation:"
	@echo "  make install   - Full installation from scratch"
	@echo "  make uninstall - Complete cleanup for fresh install"
	@echo "  make setup     - Create directories and .env file"
	@echo ""
	@echo "Services:"
	@echo "  make up        - Start all services"
	@echo "  make down      - Stop all services"
	@echo "  make restart   - Restart all services"
	@echo "  make clean     - Stop and remove all containers, networks, and volumes"
	@echo "  make logs      - Follow logs from all services"
	@echo "  make status    - Show status of all services"
	@echo ""
	@echo "ML Pipeline:"
	@echo "  make train     - Train XGBoost model and register in MLflow"
	@echo "  make champion  - Set latest model version as champion"
	@echo "  make reload    - Reload model in API"
	@echo ""
	@echo "Web Interfaces:"
	@echo "  make airflow   - Open Airflow UI (http://localhost:8080)"
	@echo "  make mlflow    - Open MLflow UI (http://localhost:5001)"
	@echo "  make minio     - Open MinIO UI (http://localhost:9001)"
	@echo "  make api       - Open API Docs (http://localhost:8800/docs)"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint      - Run ruff linter"
	@echo "  make format    - Format Python code"
	@echo "  make test      - Run tests"
	@echo "  make test-cov  - Run tests with coverage"

# Setup directories and permissions
setup:
	@echo "Creating directories..."
	@mkdir -p airflow/config airflow/dags airflow/logs airflow/plugins
	@echo "Setting permissions..."
	@chmod 777 airflow/logs || sudo chmod 777 airflow/logs
	@echo "Creating .env file..."
	@if [ ! -f .env ]; then \
		echo "AIRFLOW_UID=0" > .env; \
		echo "SOCRATA_APP_TOKEN=$(SOCRATA_TOKEN)" >> .env; \
		echo "DATA_REPO_BUCKET_NAME=data" >> .env; \
		echo ".env file created. Please edit SOCRATA_APP_TOKEN if needed."; \
	else \
		echo ".env file already exists, skipping."; \
	fi
	@echo "Setup complete!"

# Full installation from scratch
install: setup
	@echo "Building all containers..."
	docker-compose --profile all build
	@echo "Starting all services..."
	docker-compose --profile all up -d
	@echo ""
	@echo "=========================================="
	@echo "  Installation complete!"
	@echo "=========================================="
	@echo ""
	@echo "Next steps:"
	@echo "  1. Wait ~2 minutes for services to be healthy"
	@echo "  2. Run: make status"
	@echo "  3. Open Airflow: make airflow - http://localhost:8080 (airflow/airflow)"
	@echo "  4. Trigger DAG 'etl_with_taskflow'"
	@echo "  5. After ETL completes, run: make train"
	@echo "  6. Then run: make champion"
	@echo ""

# Complete uninstall for fresh install
uninstall:
	@echo "Stopping all services..."
	docker-compose --profile all down -v --remove-orphans
	@echo "Removing logs..."
	@rm -rf airflow/logs/*
	@echo "Removing .env file..."
	@rm -f .env
	@echo "Cleaning Docker cache..."
	docker system prune -f
	@echo ""
	@echo "=========================================="
	@echo "  Uninstall complete!"
	@echo "=========================================="
	@echo ""
	@echo "Run 'make install' to reinstall from scratch."
	@echo ""

# Start all services
up:
	docker-compose --profile all up -d

# Stop all services
down:
	docker-compose --profile all down

# Restart all services
restart:
	docker-compose --profile all restart

# Clean everything (removes data!)
clean:
	docker-compose --profile all down -v --remove-orphans
	@rm -rf airflow/logs/*
	@echo "All containers, networks, volumes, and logs removed"

# Follow logs
logs:
	docker-compose --profile all logs -f

# Show status
status:
	docker-compose --profile all ps

# Train model
train:
	@echo "Training XGBoost model..."
	docker-compose run --rm trainer python train_xgboost.py

# Set champion alias
champion:
	@echo "Setting champion alias..."
	@docker-compose run --rm trainer python -c "\
import mlflow; \
mlflow.set_tracking_uri('http://mlflow:5000'); \
client = mlflow.tracking.MlflowClient(); \
versions = client.search_model_versions('name=\"xgboost_chicago_crimes\"'); \
latest = max(versions, key=lambda x: int(x.version)); \
client.set_registered_model_alias('xgboost_chicago_crimes', 'champion', latest.version); \
print(f'Champion alias set to version {latest.version}')"

# Reload model in API
reload:
	@echo "Reloading model in API..."
	@curl -s -X POST http://localhost:8800/model/reload | python3 -m json.tool

# Open web interfaces
airflow:
	@echo "Opening Airflow UI..."
	@open http://localhost:8080 || xdg-open http://localhost:8080 2>/dev/null || echo "Open http://localhost:8080"

mlflow:
	@echo "Opening MLflow UI..."
	@open http://localhost:5001 || xdg-open http://localhost:5001 2>/dev/null || echo "Open http://localhost:5001"

minio:
	@echo "Opening MinIO UI..."
	@open http://localhost:9001 || xdg-open http://localhost:9001 2>/dev/null || echo "Open http://localhost:9001"

api:
	@echo "Opening API Docs..."
	@open http://localhost:8800/docs || xdg-open http://localhost:8800/docs 2>/dev/null || echo "Open http://localhost:8800/docs"

# Code quality
lint:
	@echo "Running ruff linter..."
	ruff check airflow/

format:
	@echo "Formatting Python code..."
	ruff format airflow/ dockerfiles/
	ruff check --fix airflow/ dockerfiles/

# Testing
test:
	@echo "Running tests..."
	uv run pytest

test-cov:
	@echo "Running tests with coverage..."
	uv run pytest --cov=airflow/dags/etl_helpers --cov-report=term-missing
