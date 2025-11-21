"""
MLFlow Model Predictor for Production Deployment
Provides a simple interface to load and use models from MLFlow Model Registry
"""

import os
import sys
import pickle
import tempfile
from typing import Dict, List, Union, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

from mlflow_training_helper import setup_mlflow_environment


class ChicagoCrimePredictor:
    """
    Production-ready predictor for Chicago Crime Arrest classification
    
    Features:
    - Load models by alias (champion, challenger, etc.)
    - Make predictions with validation
    - Get model metadata and information
    - Log predictions to MLFlow (optional)
    - Support for batch predictions
    """
    
    def __init__(
        self, 
        model_name: str,
        alias: str = "champion",
        tracking_uri: str = "http://localhost:5001",
        validate_input: bool = True
    ):
        """
        Initialize predictor with model from MLFlow Model Registry
        
        Args:
            model_name: Name of the registered model in MLFlow
            alias: Model alias to load (default: "champion")
            tracking_uri: MLFlow tracking server URI
            validate_input: Whether to validate input features
        """
        # Setup MLFlow environment
        setup_mlflow_environment()
        mlflow.set_tracking_uri(tracking_uri)
        
        self.model_name = model_name
        self.alias = alias
        self.tracking_uri = tracking_uri
        self.validate_input = validate_input
        
        # Initialize client
        self.client = MlflowClient()
        
        # Load model
        print(f"[*] Loading model '{model_name}' with alias '{alias}'...")
        self.model, self.model_info = self._load_model()
        print(f"  [OK] Model loaded successfully")
        print(f"  [INFO] Model version: {self.model_info['version']}")
        print(f"  [INFO] Run ID: {self.model_info['run_id']}")
        
        # Get expected features
        self.expected_features = self._get_expected_features()
        
    def _load_model(self) -> Tuple[any, Dict]:
        """
        Load model from MLFlow Model Registry by alias
        
        Returns:
            Tuple of (model, model_info)
        """
        try:
            # Get model details
            model_details = self.client.get_registered_model(self.model_name)
            
            # Get version from alias
            version = model_details.aliases.get(self.alias)
            
            if not version:
                raise ValueError(f"No model version found with alias '{self.alias}'")
            
            # Get model version details
            model_version = self.client.get_model_version(self.model_name, version)
            run_id = model_version.run_id
            
            # Load model from artifacts
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = self.client.download_artifacts(run_id, "model/model.pkl", tmpdir)
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            
            # Get model info
            model_info = {
                'name': self.model_name,
                'alias': self.alias,
                'version': version,
                'run_id': run_id,
                'status': model_version.status,
                'creation_time': model_version.creation_timestamp
            }
            
            return model, model_info
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _get_expected_features(self) -> List[str]:
        """
        Get expected feature names from model
        
        Returns:
            List of feature names
        """
        if hasattr(self.model, 'feature_names_in_'):
            return list(self.model.feature_names_in_)
        elif hasattr(self.model, 'get_booster'):
            # XGBoost model - get feature names from booster
            try:
                return self.model.get_booster().feature_names
            except:
                pass
        
        # If we can't get features from model, raise error
        raise AttributeError(
            "Could not determine expected features from model. "
            "Please provide features manually or ensure model has feature_names_in_ attribute."
        )
    
    def _validate_features(self, X: pd.DataFrame) -> None:
        """
        Validate input features
        
        Args:
            X: Input features DataFrame
            
        Raises:
            ValueError: If features are invalid
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        # Check for missing features
        missing_features = set(self.expected_features) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Check for extra features (warning only)
        extra_features = set(X.columns) - set(self.expected_features)
        if extra_features:
            print(f"  [WARN] Extra features will be ignored: {extra_features}")
        
        # Check for null values
        null_counts = X[self.expected_features].isnull().sum()
        if null_counts.any():
            null_features = null_counts[null_counts > 0].to_dict()
            raise ValueError(f"Null values found in features: {null_features}")
    
    def predict(
        self, 
        X: Union[pd.DataFrame, Dict, List[Dict]],
        return_proba: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions on input data
        
        Args:
            X: Input features (DataFrame, dict, or list of dicts)
            return_proba: Whether to return probabilities
            
        Returns:
            Predictions array, or tuple of (predictions, probabilities)
        """
        # Convert input to DataFrame if needed
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif isinstance(X, list):
            X = pd.DataFrame(X)
        
        # Validate input
        if self.validate_input:
            self._validate_features(X)
        
        # Select only expected features in correct order
        X_processed = X[self.expected_features]
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        
        if return_proba:
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X_processed)[:, 1]
                return predictions, probabilities
            else:
                print("  [WARN] Model does not support probability predictions")
                return predictions, None
        
        return predictions
    
    def predict_proba(self, X: Union[pd.DataFrame, Dict, List[Dict]]) -> np.ndarray:
        """
        Predict probabilities for positive class (Arrest)
        
        Args:
            X: Input features (DataFrame, dict, or list of dicts)
            
        Returns:
            Array of probabilities for positive class
        """
        # Convert input to DataFrame if needed
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif isinstance(X, list):
            X = pd.DataFrame(X)
        
        # Validate input
        if self.validate_input:
            self._validate_features(X)
        
        # Select only expected features
        X_processed = X[self.expected_features]
        
        # Get probabilities
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_processed)[:, 1]
            return probabilities
        else:
            raise AttributeError("Model does not support probability predictions")
    
    def predict_with_explanation(
        self, 
        X: Union[pd.DataFrame, Dict]
    ) -> Dict:
        """
        Make prediction with detailed explanation
        
        Args:
            X: Input features (single sample)
            
        Returns:
            Dictionary with prediction, probability, and metadata
        """
        # Convert to DataFrame if needed
        if isinstance(X, dict):
            X_df = pd.DataFrame([X])
        else:
            X_df = X.head(1) if len(X) > 1 else X
        
        # Make prediction
        prediction = self.predict(X_df)[0]
        
        # Get probability if available
        probability = None
        if hasattr(self.model, 'predict_proba'):
            probability = self.predict_proba(X_df)[0]
        
        # Build explanation
        explanation = {
            'prediction': int(prediction),
            'prediction_label': 'Arrest' if prediction == 1 else 'No Arrest',
            'probability': float(probability) if probability is not None else None,
            'confidence': float(probability) if probability is not None else None,
            'model_info': {
                'name': self.model_name,
                'alias': self.alias,
                'version': self.model_info['version'],
                'run_id': self.model_info['run_id']
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Add feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(
                self.expected_features,
                self.model.feature_importances_
            ))
            # Sort by importance
            feature_importance = dict(sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            explanation['feature_importance'] = feature_importance
        
        return explanation
    
    def get_model_info(self) -> Dict:
        """
        Get detailed model information
        
        Returns:
            Dictionary with model metadata
        """
        # Get run metrics
        run = self.client.get_run(self.model_info['run_id'])
        metrics = run.data.metrics
        params = run.data.params
        
        info = {
            'model_name': self.model_name,
            'alias': self.alias,
            'version': self.model_info['version'],
            'run_id': self.model_info['run_id'],
            'status': self.model_info['status'],
            'creation_time': self.model_info['creation_time'],
            'model_type': params.get('model_type', 'Unknown'),
            'metrics': {
                'test_accuracy': metrics.get('test_accuracy'),
                'test_mcc': metrics.get('test_mcc'),
                'test_auc': metrics.get('test_auc'),
                'test_f1': metrics.get('test_f1'),
                'test_precision': metrics.get('test_precision'),
                'test_recall': metrics.get('test_recall')
            },
            'expected_features': self.expected_features,
            'tracking_uri': self.tracking_uri
        }
        
        return info
    
    def batch_predict(
        self, 
        X: pd.DataFrame,
        batch_size: int = 1000,
        return_proba: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions in batches (for large datasets)
        
        Args:
            X: Input features DataFrame
            batch_size: Number of samples per batch
            return_proba: Whether to return probabilities
            
        Returns:
            Predictions array, or tuple of (predictions, probabilities)
        """
        n_samples = len(X)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        print(f"[*] Processing {n_samples} samples in {n_batches} batches...")
        
        predictions_list = []
        probabilities_list = [] if return_proba else None
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            batch = X.iloc[start_idx:end_idx]
            
            if return_proba:
                batch_pred, batch_proba = self.predict(batch, return_proba=True)
                predictions_list.append(batch_pred)
                if batch_proba is not None:
                    probabilities_list.append(batch_proba)
            else:
                batch_pred = self.predict(batch)
                predictions_list.append(batch_pred)
            
            if (i + 1) % 10 == 0:
                print(f"  [*] Processed {end_idx}/{n_samples} samples...")
        
        predictions = np.concatenate(predictions_list)
        
        if return_proba and probabilities_list:
            probabilities = np.concatenate(probabilities_list)
            return predictions, probabilities
        
        return predictions
    
    def log_prediction(
        self,
        X: Union[pd.DataFrame, Dict],
        prediction: int,
        probability: Optional[float] = None,
        experiment_name: str = "production_predictions"
    ) -> str:
        """
        Log prediction to MLFlow for monitoring
        
        Args:
            X: Input features
            prediction: Model prediction
            probability: Prediction probability
            experiment_name: MLFlow experiment name
            
        Returns:
            Run ID of logged prediction
        """
        # Get or create experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
        
        # Start run
        with mlflow.start_run(experiment_id=experiment_id, run_name="prediction") as run:
            # Log model info
            mlflow.set_tags({
                'model_name': self.model_name,
                'model_alias': self.alias,
                'model_version': self.model_info['version'],
                'prediction_type': 'single'
            })
            
            # Log prediction
            mlflow.log_metrics({
                'prediction': float(prediction),
                'probability': float(probability) if probability is not None else 0.0
            })
            
            # Log input features (as params - limited to 500 chars)
            if isinstance(X, dict):
                for key, value in X.items():
                    mlflow.log_param(f"input_{key}", str(value)[:500])
            
            return run.info.run_id


def main():
    """Example usage of ChicagoCrimePredictor"""
    
    print("\n" + "="*70)
    print("CHICAGO CRIME PREDICTOR - EXAMPLE USAGE")
    print("="*70)
    
    try:
        # Initialize predictor with champion model
        print("\n[*] Initializing predictor with champion model...")
        predictor = ChicagoCrimePredictor(
            model_name="xgboost_chicago",
            alias="champion"
        )
        
        # Get model info
        print("\n[*] Model Information:")
        info = predictor.get_model_info()
        print(f"  Model: {info['model_name']} (version {info['version']})")
        print(f"  Type: {info['model_type']}")
        print(f"  Test MCC: {info['metrics']['test_mcc']:.4f}")
        print(f"  Test Accuracy: {info['metrics']['test_accuracy']:.4f}")
        
        # Example prediction with dict
        print("\n[*] Example 1: Single prediction with dict")
        sample_crime = {
            'Primary Type_ENCODED': 5,
            'Location Description_ENCODED': 10,
            'Domestic': 0,
            'Latitude': 41.8781,
            'Longitude': -87.6298,
            'distance_to_nearest_station': 0.5,
            'Month': 6
        }
        
        prediction = predictor.predict(sample_crime)
        probability = predictor.predict_proba(sample_crime)
        
        print(f"  Prediction: {prediction[0]} ({'Arrest' if prediction[0] == 1 else 'No Arrest'})")
        print(f"  Probability: {probability[0]:.4f}")
        
        # Example with explanation
        print("\n[*] Example 2: Prediction with explanation")
        explanation = predictor.predict_with_explanation(sample_crime)
        print(f"  Prediction: {explanation['prediction_label']}")
        print(f"  Confidence: {explanation['confidence']:.4f}")
        print(f"  Model: {explanation['model_info']['name']} v{explanation['model_info']['version']}")
        
        if 'feature_importance' in explanation:
            print("\n  Top 3 Important Features:")
            for i, (feature, importance) in enumerate(list(explanation['feature_importance'].items())[:3], 1):
                print(f"    {i}. {feature}: {importance:.4f}")
        
        print("\n[OK] Predictor is working correctly!")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
