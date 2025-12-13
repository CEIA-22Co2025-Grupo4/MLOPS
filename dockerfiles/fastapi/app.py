"""
Chicago Crime Arrest Prediction API

REST API for serving ML model predictions using FastAPI and MLflow.
"""

import os
import pickle
import tempfile
import logging
from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager

import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "xgboost_chicago_crimes")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")

model = None
model_version = None
model_info = None
client = None


class CrimeFeatures(BaseModel):
    """Input features for crime arrest prediction."""
    
    primary_type_freq: float = Field(
        ...,
        description="Frequency encoding of primary crime type",
        ge=0,
        le=1,
        json_schema_extra={"example": 0.123}
    )
    location_description_freq: float = Field(
        ...,
        description="Frequency encoding of location description",
        ge=0,
        le=1,
        json_schema_extra={"example": 0.045}
    )
    beat_freq: float = Field(
        ...,
        description="Frequency encoding of police beat",
        ge=0,
        le=1,
        json_schema_extra={"example": 0.012}
    )
    ward_freq: float = Field(
        ...,
        description="Frequency encoding of ward",
        ge=0,
        le=1,
        json_schema_extra={"example": 0.034}
    )
    community_area_freq: float = Field(
        ...,
        description="Frequency encoding of community area",
        ge=0,
        le=1,
        json_schema_extra={"example": 0.028}
    )
    day_of_week_sin: float = Field(
        ...,
        description="Cyclic sine encoding of day of week",
        ge=-1,
        le=1,
        json_schema_extra={"example": 0.781}
    )
    x_coordinate_standardized: float = Field(
        ...,
        description="Standardized X coordinate",
        json_schema_extra={"example": 1.234}
    )
    longitude_standardized: float = Field(
        ...,
        description="Standardized longitude",
        json_schema_extra={"example": -0.567}
    )
    latitude_standardized: float = Field(
        ...,
        description="Standardized latitude",
        json_schema_extra={"example": 0.890}
    )
    y_coordinate_standardized: float = Field(
        ...,
        description="Standardized Y coordinate",
        json_schema_extra={"example": -1.123}
    )
    distance_crime_to_police_station_standardized: float = Field(
        ...,
        description="Standardized distance to nearest police station",
        json_schema_extra={"example": 0.345}
    )


class PredictionResponse(BaseModel):
    """Response for single prediction."""
    
    prediction: bool = Field(..., description="Arrest prediction (True/False)")
    probability: float = Field(..., description="Probability of arrest (0-1)")
    model_version: str = Field(..., description="Model version used")
    timestamp: str = Field(..., description="Prediction timestamp (ISO format)")


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""
    
    instances: List[CrimeFeatures] = Field(
        ...,
        description="List of crime instances to predict",
        min_length=1,
        max_length=1000
    )


class BatchPredictionItem(BaseModel):
    """Single item in batch prediction response."""
    
    prediction: bool
    probability: float


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    
    predictions: List[BatchPredictionItem]
    model_version: str
    timestamp: str
    count: int


class ModelInfoResponse(BaseModel):
    """Model information response."""
    
    name: str
    version: str
    alias: str
    run_id: Optional[str]
    status: str
    metrics: dict
    description: str


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status: 'healthy' or 'degraded'")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded")
    model_name: Optional[str] = Field(None, description="Name of loaded model")
    model_version: Optional[str] = Field(None, description="Version of loaded model")


class ErrorResponse(BaseModel):
    """Standard error response."""
    
    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code for programmatic handling")


class ReloadResponse(BaseModel):
    """Model reload response."""
    
    status: str = Field(..., description="Operation status")
    message: str = Field(..., description="Status message")


class APIInfo(BaseModel):
    """API information."""
    
    name: str = "Chicago Crime Arrest Prediction API"
    version: str = "1.0.0"
    description: str = "ML API for predicting arrest probability in Chicago crimes"
    model_name: Optional[str] = None
    model_version: Optional[str] = None


def setup_mlflow():
    """Configure MLflow environment."""
    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "minio")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "minio123")
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://s3:9000")
    os.environ["AWS_ENDPOINT_URL_S3"] = os.getenv("AWS_ENDPOINT_URL_S3", "http://s3:9000")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def load_model_from_registry():
    """Load model from MLflow Model Registry."""
    global model, model_version, model_info, client
    
    setup_mlflow()
    client = MlflowClient()
    
    logger.info(f"Loading model '{MODEL_NAME}' with alias '{MODEL_ALIAS}'...")
    
    try:
        model_details = client.get_registered_model(MODEL_NAME)
        version = model_details.aliases.get(MODEL_ALIAS)
        
        if not version:
            logger.warning(f"No model with alias '{MODEL_ALIAS}', trying latest version")
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            if not versions:
                raise ValueError(f"No versions found for model '{MODEL_NAME}'")
            version = max(versions, key=lambda v: int(v.version)).version
        
        model_version_obj = client.get_model_version(MODEL_NAME, version)
        run_id = model_version_obj.run_id
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = client.download_artifacts(run_id, "model/model.pkl", tmpdir)
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        
        model_version = version
        model_info = {
            "name": MODEL_NAME,
            "version": version,
            "alias": MODEL_ALIAS,
            "run_id": run_id,
            "status": model_version_obj.status,
        }
        
        logger.info(f"Model loaded successfully: {MODEL_NAME} v{version}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    success = load_model_from_registry()
    if not success:
        logger.warning("Model not loaded at startup. Some endpoints may not work.")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Chicago Crime Arrest Prediction API",
    description="""
## 🚔 Arrest Prediction API

This API serves a Machine Learning model that predicts the probability 
of a reported crime in Chicago resulting in an arrest.

### Model
- **Type**: XGBoost Classifier
- **Metrics**: MCC ~0.58, AUC ~0.89
- **Features**: 11 processed features

### Endpoints
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions (up to 1000)
- `GET /model/info` - Model information and metrics
- `GET /health` - Health check
    """,
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", response_model=APIInfo, tags=["Info"])
async def get_api_info():
    """Get API information and status."""
    return APIInfo(
        model_name=MODEL_NAME if model else None,
        model_version=model_version if model else None,
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    responses={
        200: {"description": "Service is healthy and model is loaded"},
        503: {"model": HealthResponse, "description": "Service is degraded (model not loaded)"},
    },
)
async def health_check():
    """
    Check API and model health status.
    
    Returns 200 if healthy, 503 if degraded (model not loaded).
    """
    response = HealthResponse(
        status="healthy" if model else "degraded",
        model_loaded=model is not None,
        model_name=MODEL_NAME if model else None,
        model_version=model_version if model else None,
    )
    
    if model is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response.model_dump(),
        )
    
    return response


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Predictions"],
    responses={
        200: {"description": "Prediction successful"},
        422: {"description": "Validation error in request body"},
        500: {"model": ErrorResponse, "description": "Prediction failed"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
async def predict(features: CrimeFeatures):
    """
    Predict arrest probability for a single crime incident.
    
    Returns the binary prediction and probability score.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check /health endpoint.",
        )
    
    try:
        df = pd.DataFrame([features.model_dump()])
        if hasattr(model, 'feature_names_in_'):
            df = df[model.feature_names_in_]
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0, 1]
        
        return PredictionResponse(
            prediction=bool(prediction),
            probability=float(probability),
            model_version=model_version,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Predictions"],
    responses={
        200: {"description": "Batch prediction successful"},
        422: {"description": "Validation error in request body"},
        500: {"model": ErrorResponse, "description": "Prediction failed"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict arrest probability for multiple crime incidents.
    
    Accepts up to 1000 instances per request.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check /health endpoint.",
        )
    
    try:
        df = pd.DataFrame([f.model_dump() for f in request.instances])
        if hasattr(model, 'feature_names_in_'):
            df = df[model.feature_names_in_]
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]
        
        results = [
            BatchPredictionItem(prediction=bool(p), probability=float(prob))
            for p, prob in zip(predictions, probabilities)
        ]
        
        return BatchPredictionResponse(
            predictions=results,
            model_version=model_version,
            timestamp=datetime.utcnow().isoformat() + "Z",
            count=len(results),
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    tags=["Model"],
    responses={
        200: {"description": "Model information retrieved successfully"},
        500: {"model": ErrorResponse, "description": "Failed to fetch model info"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
async def get_model_info():
    """
    Get information about the currently deployed model.
    
    Returns model metadata and performance metrics.
    """
    if model is None or model_info is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check /health endpoint.",
        )
    
    try:
        run = client.get_run(model_info["run_id"])
        metrics = run.data.metrics
        
        return ModelInfoResponse(
            name=model_info["name"],
            version=model_info["version"],
            alias=model_info["alias"],
            run_id=model_info["run_id"],
            status=model_info["status"],
            metrics={
                "test_accuracy": metrics.get("test_accuracy"),
                "test_precision": metrics.get("test_precision"),
                "test_recall": metrics.get("test_recall"),
                "test_f1": metrics.get("test_f1"),
                "test_mcc": metrics.get("test_mcc"),
                "test_auc": metrics.get("test_auc"),
            },
            description="XGBoost classifier for Chicago crime arrest prediction",
        )
    except Exception as e:
        logger.error(f"Error fetching model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch model info: {str(e)}",
        )


@app.post(
    "/model/reload",
    response_model=ReloadResponse,
    tags=["Model"],
    responses={
        200: {"description": "Model reloaded successfully"},
        500: {"model": ErrorResponse, "description": "Failed to reload model"},
    },
)
async def reload_model():
    """
    Reload the model from MLflow Model Registry.
    
    Use this endpoint to update the model without restarting the service.
    """
    success = load_model_from_registry()
    
    if success:
        return ReloadResponse(
            status="success",
            message=f"Model reloaded: {MODEL_NAME} v{model_version}",
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reload model",
        )
