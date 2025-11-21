"""
MLFlow Training Helper Functions
Reusable functions for training and tracking ML models with MLFlow
"""

import os
import sys
import time
import pickle
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import io

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient


# MLFlow Configuration
MLFLOW_TRACKING_URI = "http://localhost:5001"


def setup_mlflow_environment():
    """
    Configure MLFlow environment variables and tracking URI
    
    Returns:
        str: Configured tracking URI
    """
    # Configure MinIO/S3 environment variables
    os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
    os.environ['AWS_ENDPOINT_URL_S3'] = 'http://localhost:9000'
    
    # Disable MLFlow console emojis to avoid Windows encoding issues
    os.environ['MLFLOW_ENABLE_EMOJI'] = 'false'
    
    # Set tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    return MLFLOW_TRACKING_URI


def get_or_create_experiment(experiment_name: str, tags: Optional[Dict[str, str]] = None) -> str:
    """
    Get existing experiment or create new one
    
    Args:
        experiment_name: Name of the experiment
        tags: Optional tags for the experiment
        
    Returns:
        str: Experiment ID
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            tags=tags or {}
        )
        print(f"[INFO] Created experiment '{experiment_name}' (ID: {experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        print(f"[INFO] Using existing experiment '{experiment_name}' (ID: {experiment_id})")
    
    return experiment_id


def calculate_classification_metrics(y_true: np.ndarray, 
                                     y_pred: np.ndarray, 
                                     y_pred_proba: Optional[np.ndarray] = None,
                                     prefix: str = "") -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (for AUC)
        prefix: Prefix for metric names (e.g., 'train_', 'test_')
        
    Returns:
        Dict with calculated metrics
    """
    metrics = {
        f'{prefix}accuracy': accuracy_score(y_true, y_pred),
        f'{prefix}precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        f'{prefix}recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        f'{prefix}f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        f'{prefix}mcc': matthews_corrcoef(y_true, y_pred)
    }
    
    # Add AUC if probabilities are provided
    if y_pred_proba is not None:
        try:
            metrics[f'{prefix}auc'] = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            # Handle case where only one class is present
            metrics[f'{prefix}auc'] = 0.0
    
    return metrics


def create_confusion_matrix_plot(y_true: np.ndarray, 
                                 y_pred: np.ndarray, 
                                 labels: Optional[List[str]] = None,
                                 title: str = "Confusion Matrix") -> plt.Figure:
    """
    Create confusion matrix visualization
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels for display
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels or ['Class 0', 'Class 1']
    )
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(title)
    plt.tight_layout()
    
    return fig


def create_feature_importance_plot(model: Any, 
                                   feature_names: List[str], 
                                   top_n: int = 20,
                                   title: str = "Feature Importance") -> plt.Figure:
    """
    Create feature importance visualization
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to display
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute")
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=importance_df, x='importance', y='feature', ax=ax, 
                hue='feature', palette='viridis', legend=False)
    ax.set_title(title)
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    plt.tight_layout()
    
    return fig


def train_and_log_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    experiment_name: str,
    run_name: str,
    model_name: str,
    tags: Optional[Dict[str, str]] = None,
    log_feature_importance: bool = True,
    register_model: bool = True
) -> Tuple[Any, str, Dict[str, float]]:
    """
    Train a model and log everything to MLFlow
    
    Args:
        model: Scikit-learn compatible model instance
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        experiment_name: MLFlow experiment name
        run_name: Name for this run
        model_name: Name for model registration
        tags: Optional tags for the run
        log_feature_importance: Whether to log feature importance plot
        register_model: Whether to register model in Model Registry
        
    Returns:
        Tuple of (trained_model, run_id, test_metrics)
    """
    
    print("\n" + "="*70)
    print(f"TRAINING {model.__class__.__name__} WITH MLFLOW")
    print("="*70)
    
    # Setup MLFlow
    setup_mlflow_environment()
    
    # Get or create experiment
    experiment_id = get_or_create_experiment(
        experiment_name,
        tags={"project": "chicago_crimes", "framework": "sklearn"}
    )
    
    # Disable autologging to have full control
    mlflow.sklearn.autolog(disable=True)
    
    # Start MLFlow run
    run = mlflow.start_run(experiment_id=experiment_id, run_name=run_name)
    
    try:
        print(f"\n[*] Started MLFlow run: {run.info.run_id}")
        
        # Log tags
        run_tags = {
            "model_type": model.__class__.__name__,
            "task": "classification",
            "dataset": "Chicago Crimes 2024"
        }
        if tags:
            run_tags.update(tags)
        mlflow.set_tags(run_tags)
        
        # Log model parameters
        if hasattr(model, 'get_params'):
            params = model.get_params()
            mlflow.log_params(params)
            print(f"[*] Logged {len(params)} parameters")
        
        # Train model
        print("\n[*] Training model...")
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        print(f"  [OK] Training completed in {train_time:.2f} seconds")
        
        # Make predictions
        print("\n[*] Making predictions...")
        pred_start = time.time()
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Get probabilities if available
        y_pred_proba_train = None
        y_pred_proba_test = None
        if hasattr(model, 'predict_proba'):
            y_pred_proba_train = model.predict_proba(X_train)[:, 1]
            y_pred_proba_test = model.predict_proba(X_test)[:, 1]
        
        pred_time = time.time() - pred_start
        print(f"  [OK] Predictions completed in {pred_time:.2f} seconds")
        
        # Calculate metrics
        print("\n[*] Calculating metrics...")
        train_metrics = calculate_classification_metrics(
            y_train, y_pred_train, y_pred_proba_train, prefix='train_'
        )
        test_metrics = calculate_classification_metrics(
            y_test, y_pred_test, y_pred_proba_test, prefix='test_'
        )
        
        # Timing metrics
        timing_metrics = {
            'training_time_seconds': train_time,
            'prediction_time_seconds': pred_time,
            'total_time_seconds': train_time + pred_time
        }
        
        # Log all metrics
        all_metrics = {**train_metrics, **test_metrics, **timing_metrics}
        mlflow.log_metrics(all_metrics)
        
        # Print metrics
        print("\n  Training Metrics:")
        for key, value in train_metrics.items():
            print(f"    {key}: {value:.4f}")
        
        print("\n  Test Metrics:")
        for key, value in test_metrics.items():
            print(f"    {key}: {value:.4f}")
        
        # Create and log visualizations
        print("\n[*] Creating visualizations...")
        
        # Confusion matrix
        fig_cm = create_confusion_matrix_plot(
            y_test, y_pred_test, 
            labels=['No Arrest', 'Arrest'],
            title=f'Confusion Matrix - {model.__class__.__name__}'
        )
        mlflow.log_figure(fig_cm, "confusion_matrix.png")
        plt.close(fig_cm)
        
        # Feature importance (if available)
        if log_feature_importance and hasattr(model, 'feature_importances_'):
            try:
                fig_fi = create_feature_importance_plot(
                    model, X_train.columns.tolist(), top_n=20,
                    title=f'Feature Importance - {model.__class__.__name__}'
                )
                mlflow.log_figure(fig_fi, "feature_importance.png")
                plt.close(fig_fi)
            except Exception as e:
                print(f"  [WARN] Could not create feature importance plot: {e}")
        
        print("  [OK] Visualizations logged")
        
        # Log classification report
        print("\n[*] Logging classification report...")
        report = classification_report(
            y_test, y_pred_test, 
            target_names=['No Arrest', 'Arrest']
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(report)
            report_path = f.name
        
        mlflow.log_artifact(report_path, artifact_path="reports")
        os.remove(report_path)
        
        # Log model as pickle artifact
        print("\n[*] Logging model...")
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            mlflow.log_artifact(str(model_path), artifact_path="model")
        
        print("  [OK] Model logged")
        
        # Register model if requested
        if register_model:
            print(f"\n[*] Registering model '{model_name}'...")
            try:
                client = MlflowClient()
                model_uri = f"runs:/{run.info.run_id}/model"
                
                # Create registered model if it doesn't exist
                try:
                    client.create_registered_model(model_name)
                    print(f"  [OK] Created registered model '{model_name}'")
                except:
                    print(f"  [INFO] Model '{model_name}' already exists")
                
                # Create model version
                model_version = client.create_model_version(
                    name=model_name,
                    source=model_uri,
                    run_id=run.info.run_id
                )
                print(f"  [OK] Registered as version {model_version.version}")
                
            except Exception as e:
                print(f"  [WARN] Could not register model: {e}")
        
        print(f"\n  [OK] Run ID: {run.info.run_id}")
        print(f"  [OK] View run at: {MLFLOW_TRACKING_URI}/#/experiments/{experiment_id}/runs/{run.info.run_id}")
        
        # End run properly
        print("\n[*] Ending MLFlow run...")
        try:
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            mlflow.end_run()
            sys.stdout = old_stdout
            print("  [OK] Run ended successfully")
        except Exception as e:
            sys.stdout = old_stdout
            print(f"  [WARN] Run end had output issues (ignored): {type(e).__name__}")
        
        return model, run.info.run_id, test_metrics
    
    except Exception as e:
        # Make sure run is ended even if there's an error
        try:
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            mlflow.end_run(status="FAILED")
            sys.stdout = old_stdout
        except:
            sys.stdout = old_stdout
        raise e


def compare_models(experiment_name: str, 
                   metric: str = "test_mcc",
                   top_n: int = 10) -> pd.DataFrame:
    """
    Compare models from an experiment based on a metric
    
    Args:
        experiment_name: Name of the experiment
        metric: Metric to sort by (default: test_mcc)
        top_n: Number of top runs to return
        
    Returns:
        DataFrame with run information sorted by metric
    """
    setup_mlflow_environment()
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=top_n
    )
    
    # Select relevant columns
    columns = ['run_id', 'start_time', 'status']
    columns += [col for col in runs.columns if col.startswith('metrics.')]
    columns += [col for col in runs.columns if col.startswith('params.')]
    columns += [col for col in runs.columns if col.startswith('tags.')]
    
    return runs[columns]


if __name__ == "__main__":
    print("MLFlow Training Helper Functions")
    print("="*70)
    print("\nThis module provides reusable functions for training models with MLFlow.")
    print("\nMain functions:")
    print("  - setup_mlflow_environment()")
    print("  - get_or_create_experiment()")
    print("  - train_and_log_model()")
    print("  - calculate_classification_metrics()")
    print("  - create_confusion_matrix_plot()")
    print("  - create_feature_importance_plot()")
    print("  - compare_models()")
    print("\nImport this module to use these functions in your training scripts.")
