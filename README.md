# Chicago Crime Arrest Prediction - MLOps Pipeline

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Airflow 2.10](https://img.shields.io/badge/airflow-2.10-green.svg)](https://airflow.apache.org/)
[![MLflow 2.10](https://img.shields.io/badge/mlflow-2.10-orange.svg)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/fastapi-0.115-teal.svg)](https://fastapi.tiangolo.com/)

End-to-end MLOps pipeline for predicting arrest probability in Chicago crime incidents.

## 👥 Authors

- Daniel Eduardo Peñaranda Peralta
- Jorge Adrián Alvarez
- María Belén Cattaneo
- Nicolás Valentín Ciarrapico
- Sabrina Daiana Pryszczuk

---

## 📋 Table of Contents

- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Services](#-services)
- [ETL Pipeline](#-etl-pipeline)
- [Model Training](#-model-training)
- [API Reference](#-api-reference)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA SOURCES                                   │
│  ┌─────────────────────┐    ┌─────────────────────┐                     │
│  │ Chicago Data Portal │    │   Police Stations   │                     │
│  │   (Socrata API)     │    │      (Static)       │                     │
│  └──────────┬──────────┘    └──────────┬──────────┘                     │
└─────────────┼──────────────────────────┼────────────────────────────────┘
              │                          │
              ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATION (Airflow)                           │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐     │
│  │Download│→│ Merge  │→│Enrich  │→│ Split  │→│Outliers│→│ Encode │     │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────────┘     │
│                                                              │           │
│  ┌────────┐ ┌────────┐ ┌────────┐                           ▼           │
│  │Summary │←│Features│←│Balance │←──────────────────────────┘           │
│  └────────┘ └────────┘ └────────┘                                       │
└─────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         STORAGE (MinIO S3)                               │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ s3://data/ml-ready-data/train_YYYY-MM-DD.csv                    │    │
│  │ s3://data/ml-ready-data/test_YYYY-MM-DD.csv                     │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    EXPERIMENT TRACKING (MLflow)                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │   XGBoost       │  │ Random Forest   │  │    Other...     │         │
│  │   MCC: 0.54     │  │   MCC: 0.52     │  │                 │         │
│  └────────┬────────┘  └─────────────────┘  └─────────────────┘         │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │              MODEL REGISTRY (champion alias)                     │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         SERVING (FastAPI)                                │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  POST /predict       - Single prediction                        │    │
│  │  POST /predict/batch - Batch predictions (up to 1000)           │    │
│  │  GET  /model/info    - Model metadata and metrics               │    │
│  │  GET  /health        - Health check (200/503)                   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/engine/install/) (with Docker Compose)
- 8GB RAM minimum
- 10GB disk space

### Installation

```bash
# 1. Clone repository
git clone <repository-url>
cd MLOPS-main

# 2. Create directories
mkdir -p airflow/{config,dags,logs,plugins}
chmod 777 airflow/logs

# 3. Configure environment
cat > .env << EOF
AIRFLOW_UID=$(id -u)
SOCRATA_APP_TOKEN=your_token_here
DATA_REPO_BUCKET_NAME=data
EOF

# 4. Start services
make install && make up

# 5. Verify (all services should be "healthy")
docker ps
```

### Get Socrata API Token

1. Go to [Chicago Data Portal](https://data.cityofchicago.org/)
2. Sign in → My Profile → Developer Settings
3. Create New App Token
4. Copy token to `.env` file

### Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| **Airflow** | http://localhost:8080 | `airflow` / `airflow` |
| **MLflow** | http://localhost:5001 | - |
| **MinIO** | http://localhost:9001 | `minio` / `minio123` |
| **API Docs** | http://localhost:8800/docs | - |

---

## 🔧 Services

| Service | Purpose | Port |
|---------|---------|------|
| **Airflow** | ETL orchestration | 8080 |
| **MLflow** | Experiment tracking & model registry | 5001 |
| **MinIO** | S3-compatible object storage | 9000/9001 |
| **PostgreSQL** | Metadata storage | 5432 |
| **FastAPI** | Model serving REST API | 8800 |
| **Valkey** | Celery broker (Redis fork) | 6379 |

---

## 📊 ETL Pipeline

The `etl_with_taskflow` DAG processes Chicago crime data through 11 stages:

| Stage | Description | Output |
|-------|-------------|--------|
| 1. Setup S3 | Create bucket, set lifecycle | - |
| 2. Download | Fetch from Socrata API | `0-raw-data/` |
| 3. Merge | Rolling 12-month window | `1-merged-data/` |
| 4. Enrich | Add geospatial + temporal features | `2-enriched-data/` |
| 5. Split | Train/test (80/20 stratified) | `3-split-data/` |
| 6. Outliers | Remove outliers (±3σ) | `4-outliers/` |
| 7. Encode | Frequency, cyclic, one-hot encoding | `5-encoded/` |
| 8. Scale | StandardScaler normalization | `6-scaled/` |
| 9. Balance | SMOTE + RandomUnderSampler | `7-balanced/` |
| 10. Features | Mutual Information selection | `ml-ready-data/` |
| 11. Summary | Log pipeline metrics to MLflow | - |

### Run ETL

**Manual trigger:**
```bash
# Via Airflow UI
open http://localhost:8080  # DAG: etl_with_taskflow → Play

# Via CLI
docker-compose run airflow-cli dags trigger etl_with_taskflow
```

**Schedule:** Monthly (1st day at 00:00)

---

## 🧪 Model Training

### Train XGBoost Model

```bash
docker-compose run trainer python mlflow_xgboost_poc_docker.py
```

### Assign Champion Alias

```bash
docker-compose run trainer python -c "
import mlflow
mlflow.set_tracking_uri('http://mlflow:5000')
client = mlflow.tracking.MlflowClient()
client.set_registered_model_alias('xgboost_chicago_crimes', 'champion', '1')
print('Champion alias set!')
"
```

### View Experiments

Open MLflow UI: http://localhost:5001

---

## 🌐 API Reference

### Health Check

```bash
curl http://localhost:8800/health
# 200: {"status": "healthy", "model_loaded": true, ...}
# 503: {"status": "degraded", "model_loaded": false, ...}
```

### Single Prediction

```bash
curl -X POST http://localhost:8800/predict \
  -H "Content-Type: application/json" \
  -d '{
    "primary_type_freq": 0.123,
    "location_description_freq": 0.045,
    "beat_freq": 0.012,
    "ward_freq": 0.034,
    "community_area_freq": 0.028,
    "day_of_week_sin": 0.781,
    "x_coordinate_standardized": 1.234,
    "longitude_standardized": -0.567,
    "latitude_standardized": 0.890,
    "y_coordinate_standardized": -1.123,
    "distance_crime_to_police_station_standardized": 0.345
  }'
```

**Response:**
```json
{
  "prediction": true,
  "probability": 0.87,
  "model_version": "1",
  "timestamp": "2025-12-13T15:30:00Z"
}
```

### Batch Prediction

```bash
curl -X POST http://localhost:8800/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"instances": [{"primary_type_freq": 0.1, ...}, {"primary_type_freq": 0.2, ...}]}'
```

### Model Info

```bash
curl http://localhost:8800/model/info
```

### Reload Model

```bash
curl -X POST http://localhost:8800/model/reload
```

---

## ⚙ Configuration

### Environment Variables

Create `.env` in project root:

```bash
# Required
AIRFLOW_UID=50000
SOCRATA_APP_TOKEN=your_token
DATA_REPO_BUCKET_NAME=data

# Optional (with defaults)
PG_USER=airflow
PG_PASSWORD=airflow
MINIO_ACCESS_KEY=minio
MINIO_SECRET_ACCESS_KEY=minio123
MLFLOW_PORT=5001
FASTAPI_PORT=8800
```

### ETL Configuration

See `airflow/dags/etl_helpers/config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `SPLIT_TEST_SIZE` | 0.2 | Test set proportion |
| `OUTLIER_STD_THRESHOLD` | 3 | Standard deviations for outlier removal |
| `MI_THRESHOLD` | 0.05 | Mutual Information threshold for feature selection |
| `SMOTE_SAMPLING_STRATEGY` | 0.5 | SMOTE oversampling ratio |

---

## 🛠 Commands

```bash
make help      # Show all commands
make up        # Start all services
make down      # Stop all services
make restart   # Restart all services
make install   # Rebuild containers
make clean     # Remove all containers and volumes
make logs      # Follow logs
make status    # Show service status
make lint      # Run linter
make format    # Format code
```

---

## 🔧 Troubleshooting

### Port 5000 conflict (macOS)

AirPlay uses port 5000. MLflow is configured to use port 5001 by default.

### Permission denied on airflow/logs

```bash
chmod 777 airflow/logs
```

### Model not loading in API

```bash
# 1. Check if model is registered
curl http://localhost:8800/health

# 2. Reload model
curl -X POST http://localhost:8800/model/reload

# 3. Check MLflow for champion alias
docker-compose run trainer python -c "
import mlflow
mlflow.set_tracking_uri('http://mlflow:5000')
client = mlflow.tracking.MlflowClient()
model = client.get_registered_model('xgboost_chicago_crimes')
print('Aliases:', model.aliases)
"
```

### ETL fails at balance_data

This was fixed. Ensure NaN values are handled in `data_splitter.py`.

---

## 📁 Project Structure

```
MLOPS-main/
├── airflow/
│   ├── dags/
│   │   ├── etl_process_taskflow.py    # Main ETL DAG
│   │   └── etl_helpers/               # ETL modules
│   │       ├── config.py              # Centralized configuration
│   │       ├── data_loader.py         # Socrata API client
│   │       ├── data_enrichment.py     # Geospatial features
│   │       ├── data_splitter.py       # Train/test split
│   │       ├── data_encoding.py       # Feature encoding
│   │       ├── data_scaling.py        # Normalization
│   │       ├── data_balancing.py      # SMOTE balancing
│   │       ├── feature_selection.py   # MI-based selection
│   │       ├── monitoring.py          # MLflow logging
│   │       └── minio_utils.py         # S3 operations
│   └── secrets/                        # Airflow secrets
├── dockerfiles/
│   ├── airflow/                        # Airflow image
│   ├── fastapi/                        # API image
│   ├── mlflow/                         # MLflow image
│   ├── postgres/                       # PostgreSQL image
│   └── trainer/                        # Training image
├── mlflow_scripts/
│   ├── mlflow_xgboost_poc_docker.py   # XGBoost training script
│   ├── champion_challenger.py          # Model promotion
│   └── predictor.py                    # Prediction utilities
├── docker-compose.yaml                 # Service definitions
├── Makefile                            # Command shortcuts
└── README.md                           # This file
```

---

## 📄 License

Apache License 2.0

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.
