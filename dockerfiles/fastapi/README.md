# FastAPI Prediction Service

REST API for serving Chicago Crime arrest predictions using MLflow Model Registry.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Service                           │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Startup (lifespan)                        ││
│  │  1. Connect to MLflow tracking server                        ││
│  │  2. Load model with "champion" alias from registry           ││
│  │  3. Download model.pkl artifact from MinIO                   ││
│  └─────────────────────────────────────────────────────────────┘│
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                      Endpoints                               ││
│  │  GET  /           → API info                                 ││
│  │  GET  /health     → Health check (200/503)                   ││
│  │  POST /predict    → Single prediction                        ││
│  │  POST /predict/batch → Batch predictions                     ││
│  │  GET  /model/info → Model metadata                           ││
│  │  POST /model/reload → Hot reload model                       ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `app.py` | FastAPI application with all endpoints |
| `Dockerfile` | Container image definition |
| `requirements.txt` | Python dependencies |

## Configuration

Environment variables (set in `docker-compose.yaml`):

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow server URL |
| `MODEL_NAME` | `xgboost_chicago_crimes` | Registered model name |
| `MODEL_ALIAS` | `champion` | Model alias to load |
| `AWS_ACCESS_KEY_ID` | `minio` | MinIO access key |
| `AWS_SECRET_ACCESS_KEY` | `minio123` | MinIO secret key |
| `MLFLOW_S3_ENDPOINT_URL` | `http://s3:9000` | MinIO endpoint |

## API Endpoints

### GET /health

Returns service health status.

**Response Codes:**
- `200`: Service healthy, model loaded
- `503`: Service degraded, model not loaded

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "xgboost_chicago_crimes",
  "model_version": "1"
}
```

### POST /predict

Single prediction request.

**Request:**
```json
{
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
}
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

**Response Codes:**
- `200`: Prediction successful
- `422`: Validation error (missing/invalid fields)
- `500`: Prediction failed
- `503`: Model not loaded

### POST /predict/batch

Batch prediction (up to 1000 instances).

**Request:**
```json
{
  "instances": [
    {"primary_type_freq": 0.1, ...},
    {"primary_type_freq": 0.2, ...}
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {"prediction": true, "probability": 0.87},
    {"prediction": false, "probability": 0.23}
  ],
  "model_version": "1",
  "timestamp": "2025-12-13T15:30:00Z",
  "count": 2
}
```

### GET /model/info

Returns model metadata and metrics from MLflow.

```json
{
  "name": "xgboost_chicago_crimes",
  "version": "1",
  "alias": "champion",
  "run_id": "abc123...",
  "status": "READY",
  "metrics": {
    "test_accuracy": 0.887,
    "test_precision": 0.878,
    "test_recall": 0.887,
    "test_f1": 0.880,
    "test_mcc": 0.539,
    "test_auc": 0.869
  },
  "description": "XGBoost classifier for Chicago crime arrest prediction"
}
```

### POST /model/reload

Hot reload model from MLflow registry without restarting the service.

```json
{
  "status": "success",
  "message": "Model reloaded: xgboost_chicago_crimes v1"
}
```

## Local Development

```bash
# Build image
docker build -t fastapi-prediction .

# Run locally (requires MLflow and MinIO)
docker run -p 8800:8800 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5001 \
  -e AWS_ENDPOINT_URL_S3=http://host.docker.internal:9000 \
  fastapi-prediction

# Test endpoint
curl http://localhost:8800/health
```

## Input Features

The model expects 11 preprocessed features:

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `primary_type_freq` | float | [0, 1] | Frequency encoding of crime type |
| `location_description_freq` | float | [0, 1] | Frequency encoding of location |
| `beat_freq` | float | [0, 1] | Frequency encoding of police beat |
| `ward_freq` | float | [0, 1] | Frequency encoding of ward |
| `community_area_freq` | float | [0, 1] | Frequency encoding of area |
| `day_of_week_sin` | float | [-1, 1] | Cyclic encoding of day |
| `x_coordinate_standardized` | float | - | Standardized X coordinate |
| `y_coordinate_standardized` | float | - | Standardized Y coordinate |
| `latitude_standardized` | float | - | Standardized latitude |
| `longitude_standardized` | float | - | Standardized longitude |
| `distance_crime_to_police_station_standardized` | float | - | Standardized distance |

## OpenAPI Documentation

- Swagger UI: http://localhost:8800/docs
- ReDoc: http://localhost:8800/redoc
- OpenAPI JSON: http://localhost:8800/openapi.json

