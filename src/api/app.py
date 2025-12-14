"""
FastAPI service for Chicago Crime Arrest Prediction.
Loads model from MLflow Model Registry and serves predictions.
"""

import os
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

import mlflow
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = "xgboost_chicago_crimes"
MODEL_ALIAS = "champion"

model = None
model_version = None
model_info = None


class CrimeFeatures(BaseModel):
    """Input features for prediction (7 features after preprocessing)."""
    iucr_freq: float = Field(..., description="Frequency encoding of IUCR code")
    primary_type_freq: float = Field(..., description="Frequency encoding of primary crime type")
    location_description_freq: float = Field(..., description="Frequency encoding of location")
    day_of_week_sin: float = Field(..., description="Sine encoding of day of week")
    x_coordinate_standardized: float = Field(..., description="Standardized X coordinate")
    y_coordinate_standardized: float = Field(..., description="Standardized Y coordinate")
    distance_crime_to_police_station_standardized: float = Field(..., description="Standardized distance to nearest police station")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "iucr_freq": 0.03,
                "primary_type_freq": 0.1,
                "location_description_freq": 0.05,
                "day_of_week_sin": 0.5,
                "x_coordinate_standardized": 0.12,
                "y_coordinate_standardized": -0.34,
                "distance_crime_to_police_station_standardized": 0.56
            }]
        }
    }


class PredictionResponse(BaseModel):
    """Response for single prediction."""
    prediction: bool
    probability: float
    model_version: str
    timestamp: str


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""
    instances: list[CrimeFeatures] = Field(..., max_length=1000)


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    predictions: list[dict]
    count: int
    model_version: str
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Model information response."""
    name: str
    version: str
    alias: str
    run_id: Optional[str] = None
    metrics: Optional[dict] = None


def load_model_from_registry():
    """Load model from MLflow Model Registry using champion alias."""
    global model, model_version, model_info
    
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info(f"Connecting to MLflow at {MLFLOW_TRACKING_URI}")
        
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        logger.info(f"Loading model from {model_uri}")
        
        model = mlflow.pyfunc.load_model(model_uri)
        
        client = mlflow.tracking.MlflowClient()
        model_version_info = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        model_version = model_version_info.version
        
        run = client.get_run(model_version_info.run_id)
        model_info = {
            "name": MODEL_NAME,
            "version": model_version,
            "alias": MODEL_ALIAS,
            "run_id": model_version_info.run_id,
            "metrics": run.data.metrics
        }
        
        logger.info(f"Model loaded successfully: {MODEL_NAME} v{model_version}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None
        model_version = None
        model_info = None
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    load_model_from_registry()
    yield


app = FastAPI(
    title="Chicago Crime Arrest Prediction API",
    description="""
## 🚔 Arrest Prediction API

This API serves a Machine Learning model that predicts the probability 
of a reported crime in Chicago resulting in an arrest.

### Model
- **Type**: XGBoost Classifier  
- **Training Accuracy**: ~94%
- **Test Accuracy**: ~90%
- **Test AUC**: ~0.89
- **Test MCC**: ~0.59

### Input Features (7 total)
| Feature | Description |
|---------|-------------|
| `iucr_freq` | Frequency encoding of IUCR crime code |
| `primary_type_freq` | Frequency encoding of primary crime type |
| `location_description_freq` | Frequency encoding of location description |
| `day_of_week_sin` | Sine encoding of day of week (cyclical) |
| `x_coordinate_standardized` | Standardized X coordinate (State Plane) |
| `y_coordinate_standardized` | Standardized Y coordinate (State Plane) |
| `distance_crime_to_police_station_standardized` | Standardized distance to nearest police station |

### Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check and model status |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch predictions (max 1000) |
| GET | `/model/info` | Model metadata and metrics |
| POST | `/model/reload` | Hot reload model from registry |

### Model Registry
The model is loaded from **MLflow Model Registry** using the `champion` alias.
    """,
    version="1.0.0",
    lifespan=lifespan,
    contact={
        "name": "MLOps Team - CEIA",
        "url": "https://github.com/your-repo/MLOPS",
    },
    license_info={
        "name": "MIT",
    },
)


@app.get("/", tags=["Info"])
def root():
    """API root endpoint with basic information."""
    return {
        "service": "Chicago Crime Arrest Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        model_name=MODEL_NAME if model else None,
        model_version=model_version if model else None,
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
def predict(features: CrimeFeatures):
    """Make a single prediction."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later or call /model/reload"
        )
    
    try:
        df = pd.DataFrame([features.model_dump()])
        
        if hasattr(model, '_model_impl') and hasattr(model._model_impl, 'feature_names_in_'):
            expected_features = list(model._model_impl.feature_names_in_)
            df = df[expected_features]
        
        prediction_proba = model.predict(df)
        
        if isinstance(prediction_proba, np.ndarray):
            prob = float(prediction_proba[0])
        else:
            prob = float(prediction_proba.iloc[0])
        
        prediction = prob >= 0.5
        
        return PredictionResponse(
            prediction=prediction,
            probability=round(prob, 4),
            model_version=model_version or "unknown",
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
def predict_batch(request: BatchPredictionRequest):
    """Make batch predictions (max 1000 instances)."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later or call /model/reload"
        )
    
    try:
        df = pd.DataFrame([f.model_dump() for f in request.instances])
        
        if hasattr(model, '_model_impl') and hasattr(model._model_impl, 'feature_names_in_'):
            expected_features = list(model._model_impl.feature_names_in_)
            df = df[expected_features]
        
        predictions_proba = model.predict(df)
        
        results = []
        for prob in predictions_proba:
            prob_val = float(prob)
            results.append({
                "prediction": prob_val >= 0.5,
                "probability": round(prob_val, 4)
            })
        
        return BatchPredictionResponse(
            predictions=results,
            count=len(results),
            model_version=model_version or "unknown",
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
def get_model_info():
    """Get information about the currently loaded model."""
    if model_info is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    return ModelInfoResponse(**model_info)


@app.post("/model/reload", tags=["Model"])
def reload_model():
    """Reload model from MLflow registry."""
    success = load_model_from_registry()
    
    if success:
        return {
            "status": "success",
            "message": f"Model reloaded: {MODEL_NAME} v{model_version}",
            "timestamp": datetime.utcnow().isoformat()
        }
    else:
        raise HTTPException(
            status_code=500,
            detail="Failed to reload model. Check logs for details."
        )
