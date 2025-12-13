# MLflow Training Scripts

Scripts for model training, experiment tracking, and model management with MLflow.

## Quick Start

```bash
# Train XGBoost model (from docker-compose)
docker-compose run trainer python mlflow_xgboost_poc_docker.py

# Set champion alias
docker-compose run trainer python -c "
import mlflow
mlflow.set_tracking_uri('http://mlflow:5000')
client = mlflow.tracking.MlflowClient()
client.set_registered_model_alias('xgboost_chicago_crimes', 'champion', '1')
"
```

## Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `mlflow_xgboost_poc_docker.py` | Train XGBoost with hyperparameter tuning | `docker-compose run trainer python mlflow_xgboost_poc_docker.py` |
| `champion_challenger.py` | Model promotion and A/B testing | `python champion_challenger.py` |
| `predictor.py` | Load and use models for prediction | Import as module |
| `verify_mlflow_setup.py` | Verify MLflow infrastructure | `python verify_mlflow_setup.py` |
| `train_multiple_models.py` | Train and compare multiple algorithms | `python train_multiple_models.py` |

---

## mlflow_xgboost_poc_docker.py

Main training script for XGBoost classifier.

### Features

- Loads data from MinIO (`s3://data/ml-ready-data/`)
- Trains XGBoost with BayesSearchCV hyperparameter tuning
- Logs metrics, parameters, and artifacts to MLflow
- Registers model in MLflow Model Registry

### Output

- **Experiment**: `chicago_crimes_xgboost`
- **Model**: `xgboost_chicago_crimes`
- **Artifacts**: Confusion matrix, feature importance, classification report

### Metrics (Typical Results)

| Metric | Train | Test |
|--------|-------|------|
| Accuracy | 92.8% | 88.7% |
| MCC | 0.858 | 0.539 |
| AUC | 98.5% | 86.9% |
| F1 | 92.8% | 88.0% |

---

## champion_challenger.py

Model lifecycle management using MLflow aliases.

### Usage

```python
from champion_challenger import ChampionChallengerManager

manager = ChampionChallengerManager(tracking_uri="http://localhost:5001")

# List registered models
models = manager.list_registered_models()

# Set champion
manager.set_alias("xgboost_chicago_crimes", "champion", "1")

# Compare models
comparison = manager.compare_models(
    champion_model_name="xgboost_chicago_crimes",
    challenger_model_name="random_forest_chicago_crimes",
)

# Promote challenger to champion
manager.promote_challenger("xgboost_chicago_crimes")

# Rollback
manager.rollback_champion("xgboost_chicago_crimes")
```

### Aliases

| Alias | Purpose |
|-------|---------|
| `champion` | Production model (loaded by FastAPI) |
| `challenger` | Candidate model for comparison |
| `previous_champion` | Backup for rollback |

---

## predictor.py

Load models from registry and make predictions.

```python
from predictor import ChicagoCrimePredictor

predictor = ChicagoCrimePredictor(
    tracking_uri="http://localhost:5001",
    model_name="xgboost_chicago_crimes"
)

# Load champion model
predictor.load_model()

# Predict
result = predictor.predict(features_df)
print(f"Prediction: {result['prediction']}, Probability: {result['probability']}")
```

---

## Configuration

Environment variables (set automatically in Docker):

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow server URL |
| `MINIO_ENDPOINT` | `http://s3:9000` | MinIO endpoint |
| `MINIO_BUCKET` | `data` | Data bucket name |
| `MINIO_PREFIX` | `ml-ready-data` | Data prefix |
| `AWS_ACCESS_KEY_ID` | `minio` | MinIO access key |
| `AWS_SECRET_ACCESS_KEY` | `minio123` | MinIO secret key |

---

## Local Development

To run scripts locally (outside Docker):

```bash
# Set environment variables
export MLFLOW_TRACKING_URI=http://localhost:5001
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minio
export AWS_SECRET_ACCESS_KEY=minio123

# Run training
python mlflow_xgboost_poc.py  # Uses local files
```

---

## Workflow

```
1. ETL Pipeline (Airflow)
   └── Generates: s3://data/ml-ready-data/train_*.csv, test_*.csv

2. Model Training
   └── mlflow_xgboost_poc_docker.py
       ├── Loads data from MinIO
       ├── Trains XGBoost
       ├── Logs to MLflow
       └── Registers model

3. Model Promotion
   └── champion_challenger.py
       └── Sets "champion" alias

4. Model Serving
   └── FastAPI loads model with "champion" alias
```

---

## MLflow UI

Access experiments and models: http://localhost:5001

**Key Views:**
- Experiments → `chicago_crimes_xgboost`
- Models → `xgboost_chicago_crimes`
- Compare runs → Select multiple runs → Compare
