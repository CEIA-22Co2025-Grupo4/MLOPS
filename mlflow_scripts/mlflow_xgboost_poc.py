"""
XGBoost POC with MLFlow Integration
Proof of Concept for Chicago Crimes arrest prediction using MLFlow tracking
"""

import os
import sys
import time
import zipfile
from pathlib import Path
from contextlib import redirect_stdout
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

from skopt import BayesSearchCV
from skopt.space import Real, Integer

import xgboost as xgb
from xgboost import XGBClassifier

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature


# Configuration
MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "chicago_crimes_xgboost"

# Get project root directory (parent of mlflow_scripts)
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets"
TRAIN_FILE = "chicago_crimes_and_stations_2024_final.csv"
TEST_FILE = "chicago_crimes_and_stations_2024_final_test.csv"


def setup_mlflow():
    """Configure MLFlow connection and environment variables"""
    print("\n[*] Setting up MLFlow configuration...")
    
    # Configure MinIO/S3 environment variables
    os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
    os.environ['AWS_ENDPOINT_URL_S3'] = 'http://localhost:9000'
    
    # Disable MLFlow console emojis to avoid Windows encoding issues
    os.environ['MLFLOW_ENABLE_EMOJI'] = 'false'
    
    # Set tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"  [OK] MLFlow tracking URI: {MLFLOW_TRACKING_URI}")
    
    # Create or get experiment
    try:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                EXPERIMENT_NAME,
                tags={
                    "project": "chicago_crimes",
                    "model_type": "classification",
                    "framework": "xgboost",
                    "phase": "POC"
                }
            )
            print(f"  [OK] Created experiment '{EXPERIMENT_NAME}' (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            print(f"  [OK] Using existing experiment '{EXPERIMENT_NAME}' (ID: {experiment_id})")
        
        return experiment_id
    
    except Exception as e:
        print(f"  [ERROR] Failed to setup experiment: {e}")
        raise


def extract_dataset(zip_path):
    """Extract CSV from zip file if needed"""
    zip_path = Path(zip_path)
    csv_path = zip_path.with_suffix('')
    
    if csv_path.exists():
        print(f"  [OK] Dataset already extracted: {csv_path.name}")
        return csv_path
    
    print(f"  [*] Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(zip_path.parent)
    print(f"  [OK] Extracted to: {csv_path.name}")
    return csv_path


def load_data():
    """Load training and test datasets"""
    print("\n[*] Loading datasets...")
    
    # Handle train file
    train_zip = DATASET_DIR / f"{TRAIN_FILE}.zip"
    train_csv = DATASET_DIR / TRAIN_FILE
    
    if not train_csv.exists() and train_zip.exists():
        train_csv = extract_dataset(train_zip)
    elif not train_csv.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_csv}")
    
    # Handle test file
    test_zip = DATASET_DIR / f"{TEST_FILE}.zip"
    test_csv = DATASET_DIR / TEST_FILE
    
    if not test_csv.exists() and test_zip.exists():
        test_csv = extract_dataset(test_zip)
    elif not test_csv.exists():
        raise FileNotFoundError(f"Test dataset not found: {test_csv}")
    
    # Load datasets
    print(f"  [*] Reading {train_csv.name}...")
    df_train = pd.read_csv(train_csv)
    print(f"  [OK] Train set: {df_train.shape[0]} rows, {df_train.shape[1]} columns")
    
    print(f"  [*] Reading {test_csv.name}...")
    df_test = pd.read_csv(test_csv)
    print(f"  [OK] Test set: {df_test.shape[0]} rows, {df_test.shape[1]} columns")
    
    # Split features and target
    target_col = "Arrest_tag"
    
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]
    
    print(f"\n  [INFO] Target distribution (train):")
    print(f"    Class 0 (No arrest): {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.2f}%)")
    print(f"    Class 1 (Arrest): {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.2f}%)")
    
    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(y_true, y_pred, model_name="XGBoost"):
    """Create confusion matrix visualization"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['No Arrest', 'Arrest']
    )
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    
    return fig


def plot_feature_importance(model, feature_names, top_n=20):
    """Create feature importance plot"""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=importance_df, x='importance', y='feature', ax=ax, palette='viridis')
    ax.set_title(f'Top {top_n} Feature Importances - XGBoost')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    plt.tight_layout()
    
    return fig


def train_xgboost_with_mlflow(X_train, X_test, y_train, y_test, experiment_id):
    """Train XGBoost model with MLFlow tracking"""
    
    print("\n" + "="*70)
    print("TRAINING XGBOOST MODEL WITH MLFLOW")
    print("="*70)
    
    # Disable MLFlow autologging completely
    mlflow.xgboost.autolog(disable=True)
    
    # Start MLFlow run manually (we'll handle closing it properly)
    run = mlflow.start_run(experiment_id=experiment_id, run_name="xgboost_poc_v3")
    
    try:
        
        print(f"\n[*] Started MLFlow run: {run.info.run_id}")
        
        # Log additional tags
        mlflow.set_tags({
            "model": "XGBoost",
            "type": "classification",
            "target": "Arrest_tag",
            "dataset": "Chicago Crimes 2024",
            "author": "MLOps Team"
        })
        
        # Model parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': ['aucpr', 'logloss'],
            'random_state': 1337,
            'n_jobs': -1
        }
        
        xgb_model = xgb.XGBClassifier(**params)

        search_spaces = {
            'max_depth': Integer(3, 16),                         
            'learning_rate': Real(0.001, 1, prior='log-uniform'),  
            'n_estimators': Integer(50, 200),                   
            'subsample': Real(0.7, 1.0),                        
            'colsample_bytree': Real(0.3, 0.9),                 
            'gamma': Real(0, 2),                                
            'min_child_weight': Integer(0, 15),                  
            'reg_lambda': Real(0.01, 1, prior='log-uniform'), 
            'reg_alpha': Real(0.01, 1, prior='log-uniform')      
        }

        opt = BayesSearchCV(
            estimator = xgb_model,
            search_spaces = search_spaces,
            scoring = 'average_precision',
            cv = 3,
            n_iter = 40,        
            n_jobs = -1,        
            verbose = 0,
            random_state = 1337
        )

        opt.fit(X_train, y_train)
        
        opt_params_as_dict = dict(opt.best_params_)
        
        for key in opt_params_as_dict:
            params[key] = round(opt_params_as_dict[key], 2)

        # Log parameters explicitly (autolog will also log them)
        mlflow.log_params(params)
        mlflow.log_metric("best_score", opt.best_score_)
        
        # Train model
        print("\n[*] Training XGBoost model...")
        start_time = time.time()
        
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        print(f"  [OK] Training completed in {train_time:.2f} seconds")
        
        # Predictions
        print("\n[*] Making predictions...")
        pred_start = time.time()
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        y_pred_proba_train = model.predict_proba(X_train)[:, 1]
        y_pred_proba_test = model.predict_proba(X_test)[:, 1]
        
        pred_time = time.time() - pred_start
        print(f"  [OK] Predictions completed in {pred_time:.2f} seconds")
        
        # Calculate metrics for both train and test
        print("\n[*] Calculating metrics...")
        
        # Training metrics
        train_metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'train_precision': precision_score(y_train, y_pred_train, average='weighted'),
            'train_recall': recall_score(y_train, y_pred_train, average='weighted'),
            'train_f1': f1_score(y_train, y_pred_train, average='weighted'),
            'train_auc': roc_auc_score(y_train, y_pred_proba_train),
            'train_mcc': matthews_corrcoef(y_train, y_pred_train)
        }
        
        # Test metrics
        test_metrics = {
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'test_precision': precision_score(y_test, y_pred_test, average='weighted'),
            'test_recall': recall_score(y_test, y_pred_test, average='weighted'),
            'test_f1': f1_score(y_test, y_pred_test, average='weighted'),
            'test_auc': roc_auc_score(y_test, y_pred_proba_test),
            'test_mcc': matthews_corrcoef(y_test, y_pred_test)
        }
        
        # Timing metrics
        timing_metrics = {
            'training_time_seconds': train_time,
            'prediction_time_seconds': pred_time,
            'total_time_seconds': train_time + pred_time
        }
        
        # Log all metrics
        mlflow.log_metrics({**train_metrics, **test_metrics, **timing_metrics})
        
        # Print metrics
        print("\n  Training Metrics:")
        for key, value in train_metrics.items():
            print(f"    {key}: {value:.4f}")
        
        print("\n  Test Metrics:")
        for key, value in test_metrics.items():
            print(f"    {key}: {value:.4f}")
        
        print("\n  Timing:")
        for key, value in timing_metrics.items():
            print(f"    {key}: {value:.4f}s")
        
        # Create and log visualizations
        print("\n[*] Creating visualizations...")
        
        # Confusion matrix
        fig_cm = plot_confusion_matrix(y_test, y_pred_test, "XGBoost")
        mlflow.log_figure(fig_cm, "confusion_matrix.png")
        plt.close(fig_cm)
        
        # Feature importance
        fig_fi = plot_feature_importance(model, X_train.columns, top_n=20)
        mlflow.log_figure(fig_fi, "feature_importance.png")
        plt.close(fig_fi)
        
        print("  [OK] Visualizations logged")
        
        # Log classification report as text artifact
        print("\n[*] Logging classification report...")
        report = classification_report(y_test, y_pred_test, 
                                      target_names=['No Arrest', 'Arrest'])
        
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")
        os.remove("classification_report.txt")
        
        # Create model signature
        print("\n[*] Creating model signature...")
        signature = infer_signature(X_train, y_pred_proba_test)
        
        # Log model using pyfunc to avoid new API endpoints
        print("[*] Logging model to MLFlow...")
        try:
            # Use pyfunc with a simple wrapper to avoid logged-models API
            import tempfile
            import shutil
            
            with tempfile.TemporaryDirectory() as tmpdir:
                # Save model using pickle
                import pickle
                model_path = Path(tmpdir) / "model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                
                # Log as artifact
                mlflow.log_artifact(str(model_path), artifact_path="model")
            
            print(f"  [OK] Model logged to MLFlow")
            
        except Exception as e:
            print(f"  [WARN] Failed to log with pyfunc, trying sklearn: {e}")
            # Fallback to sklearn if needed
            model_info = mlflow.sklearn.log_model(
                model,
                artifact_path="model_sklearn",
                signature=signature,
                input_example=X_train.head(5)
            )
            print(f"  [OK] Model logged via sklearn")
        
        # Register model manually (compatible with older MLFlow server)
        print("[*] Registering model...")
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            
            model_name = "xgboost_chicago_crimes"
            model_uri = f"runs:/{run.info.run_id}/model"
            
            # Create or update registered model
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
            print(f"  [INFO] Model is still logged and can be registered later")
        print(f"  [OK] Run ID: {run.info.run_id}")
        print(f"  [OK] View run at: {MLFLOW_TRACKING_URI}/#/experiments/{experiment_id}/runs/{run.info.run_id}")
        
        # Manually end the run to avoid emoji error
        print("\n[*] Ending MLFlow run...")
        try:
            # Redirect stdout temporarily to catch emoji errors
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


def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print("XGBOOST POC WITH MLFLOW - CHICAGO CRIMES ARREST PREDICTION")
    print("="*70)
    
    try:
        # Setup MLFlow
        experiment_id = setup_mlflow()
        
        # Load data
        X_train, X_test, y_train, y_test = load_data()
        
        # Train model with MLFlow
        model, run_id, metrics = train_xgboost_with_mlflow(
            X_train, X_test, y_train, y_test, experiment_id
        )
        
        # Final summary
        print("\n" + "="*70)
        print("POC COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"""
Summary:
  - Experiment: {EXPERIMENT_NAME} (ID: {experiment_id})
  - Run ID: {run_id}
  - Model: XGBoost Classifier
  - Test Accuracy: {metrics['test_accuracy']:.4f}
  - Test MCC: {metrics['test_mcc']:.4f}
  - Test AUC: {metrics['test_auc']:.4f}
  
Next Steps:
  1. View results in MLFlow UI: {MLFLOW_TRACKING_URI}
  2. Compare with other model runs
  3. Register best model version
  4. Proceed to PHASE 3: Create training helper function
""")
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] POC failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
