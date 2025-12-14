"""
Champion/Challenger Model Management System
Implements model promotion and A/B testing framework using MLFlow Model Registry
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import zipfile

import pandas as pd
import numpy as np
from mlflow.tracking import MlflowClient
import mlflow

from mlflow_training_helper import (
    setup_mlflow_environment,
    calculate_classification_metrics
)


class ChampionChallengerManager:
    """
    Manages Champion/Challenger model lifecycle in MLFlow Model Registry
    
    Features:
    - Assign champion/challenger aliases
    - Compare model performance
    - Automatic promotion based on metrics
    - Rollback capability
    - A/B testing support
    """
    
    def __init__(self, tracking_uri: str = "http://localhost:5001"):
        """
        Initialize Champion/Challenger Manager
        
        Args:
            tracking_uri: MLFlow tracking server URI
        """
        setup_mlflow_environment()
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        self.tracking_uri = tracking_uri
    
    def list_registered_models(self) -> list:
        """
        List all registered models in MLFlow
        
        Returns:
            List of registered model names
        """
        models = self.client.search_registered_models()
        return [model.name for model in models]
    
    def get_model_versions(self, model_name: str) -> pd.DataFrame:
        """
        Get all versions of a registered model
        
        Args:
            model_name: Name of the registered model
            
        Returns:
            DataFrame with version information
        """
        versions = self.client.search_model_versions(f"name='{model_name}'")
        
        version_data = []
        for version in versions:
            # Get aliases for this version
            model_details = self.client.get_registered_model(model_name)
            aliases = []
            for alias, ver in model_details.aliases.items():
                if ver == version.version:
                    aliases.append(alias)
            
            version_data.append({
                'version': version.version,
                'run_id': version.run_id,
                'status': version.status,
                'aliases': ', '.join(aliases) if aliases else 'None',
                'creation_time': version.creation_timestamp
            })
        
        return pd.DataFrame(version_data)
    
    def set_alias(self, model_name: str, alias: str, version: str) -> None:
        """
        Set an alias for a model version
        
        Args:
            model_name: Name of the registered model
            alias: Alias to assign (e.g., 'champion', 'challenger')
            version: Model version number
        """
        self.client.set_registered_model_alias(model_name, alias, version)
        print(f"[OK] Set alias '{alias}' for {model_name} version {version}")
    
    def delete_alias(self, model_name: str, alias: str) -> None:
        """
        Delete an alias from a model
        
        Args:
            model_name: Name of the registered model
            alias: Alias to delete
        """
        self.client.delete_registered_model_alias(model_name, alias)
        print(f"[OK] Deleted alias '{alias}' from {model_name}")
    
    def get_model_by_alias(self, model_name: str, alias: str):
        """
        Load a model by its alias
        
        Args:
            model_name: Name of the registered model
            alias: Alias of the model (e.g., 'champion')
            
        Returns:
            Loaded model object
        """
        model_uri = f"models:/{model_name}@{alias}"
        
        try:
            # Try loading as sklearn model first
            model = mlflow.sklearn.load_model(model_uri)
            return model
        except:
            # Fallback to pyfunc
            model = mlflow.pyfunc.load_model(model_uri)
            return model
    
    def get_run_metrics(self, run_id: str) -> Dict[str, float]:
        """
        Get metrics from a specific run
        
        Args:
            run_id: MLFlow run ID
            
        Returns:
            Dictionary of metrics
        """
        run = self.client.get_run(run_id)
        return run.data.metrics
    
    def compare_models(
        self,
        champion_model_name: str,
        challenger_model_name: str,
        champion_alias: str = "champion",
        challenger_alias: str = "challenger",
        X_test: pd.DataFrame = None,
        y_test: pd.Series = None
    ) -> Dict:
        """
        Compare champion and challenger models
        
        Args:
            champion_model_name: Name of the champion registered model
            challenger_model_name: Name of the challenger registered model
            champion_alias: Alias for champion model
            challenger_alias: Alias for challenger model
            X_test: Test features (optional, for live comparison)
            y_test: Test labels (optional, for live comparison)
            
        Returns:
            Dictionary with comparison results
        """
        print(f"\n{'='*70}")
        print(f"COMPARING {champion_alias.upper()} vs {challenger_alias.upper()}")
        print(f"{'='*70}")
        
        # Get model details
        champion_model_details = self.client.get_registered_model(champion_model_name)
        challenger_model_details = self.client.get_registered_model(challenger_model_name)
        
        # Get champion and challenger versions
        champion_version = champion_model_details.aliases.get(champion_alias)
        challenger_version = challenger_model_details.aliases.get(challenger_alias)
        
        if not champion_version:
            raise ValueError(f"No model with alias '{champion_alias}' found")
        if not challenger_version:
            raise ValueError(f"No model with alias '{challenger_alias}' found")
        
        print(f"\n[INFO] Champion: {champion_model_name} version {champion_version}")
        print(f"[INFO] Challenger: {challenger_model_name} version {challenger_version}")
        
        # Get run IDs
        champion_model_version = self.client.get_model_version(champion_model_name, champion_version)
        challenger_model_version = self.client.get_model_version(challenger_model_name, challenger_version)
        
        champion_run_id = champion_model_version.run_id
        challenger_run_id = challenger_model_version.run_id
        
        # Get metrics from original runs
        champion_metrics = self.get_run_metrics(champion_run_id)
        challenger_metrics = self.get_run_metrics(challenger_run_id)
        
        # Display comparison
        print(f"\n{'Metric':<25} {'Champion':<15} {'Challenger':<15} {'Winner':<10}")
        print("-" * 70)
        
        comparison = {}
        key_metrics = ['test_mcc', 'test_accuracy', 'test_auc', 'test_f1', 
                      'test_precision', 'test_recall']
        
        for metric in key_metrics:
            champ_val = champion_metrics.get(metric, 0)
            chall_val = challenger_metrics.get(metric, 0)
            
            winner = "Challenger" if chall_val > champ_val else "Champion"
            if chall_val == champ_val:
                winner = "Tie"
            
            print(f"{metric:<25} {champ_val:<15.4f} {chall_val:<15.4f} {winner:<10}")
            
            comparison[metric] = {
                'champion': champ_val,
                'challenger': chall_val,
                'winner': winner,
                'improvement': chall_val - champ_val
            }
        
        # If test data provided, do live comparison
        if X_test is not None and y_test is not None:
            print(f"\n[*] Performing live comparison on provided test data...")
            
            # Load models - need to load pickled models from artifacts
            print(f"[*] Loading champion model from {champion_model_name}...")
            print(f"[*] Loading challenger model from {challenger_model_name}...")
            
            # Load models using pickle from artifacts
            import pickle
            import tempfile
            
            # Get champion model artifact
            champion_artifact_uri = f"runs:/{champion_run_id}/model/model.pkl"
            with tempfile.TemporaryDirectory() as tmpdir:
                champion_path = self.client.download_artifacts(champion_run_id, "model/model.pkl", tmpdir)
                with open(champion_path, 'rb') as f:
                    champion_model = pickle.load(f)
            
            # Get challenger model artifact
            with tempfile.TemporaryDirectory() as tmpdir:
                challenger_path = self.client.download_artifacts(challenger_run_id, "model/model.pkl", tmpdir)
                with open(challenger_path, 'rb') as f:
                    challenger_model = pickle.load(f)
            
            # Make predictions
            champion_pred = champion_model.predict(X_test)
            challenger_pred = challenger_model.predict(X_test)
            
            # Get probabilities if available
            champion_proba = None
            challenger_proba = None
            
            if hasattr(champion_model, 'predict_proba'):
                champion_proba = champion_model.predict_proba(X_test)[:, 1]
            if hasattr(challenger_model, 'predict_proba'):
                challenger_proba = challenger_model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            champion_live_metrics = calculate_classification_metrics(
                y_test, champion_pred, champion_proba, prefix='live_'
            )
            challenger_live_metrics = calculate_classification_metrics(
                y_test, challenger_pred, challenger_proba, prefix='live_'
            )
            
            print("\nLive Test Results:")
            print("-" * 70)
            for metric in ['live_mcc', 'live_accuracy', 'live_auc', 'live_f1']:
                champ_val = champion_live_metrics.get(metric, 0)
                chall_val = challenger_live_metrics.get(metric, 0)
                winner = "Challenger" if chall_val > champ_val else "Champion"
                print(f"{metric:<25} {champ_val:<15.4f} {chall_val:<15.4f} {winner:<10}")
            
            comparison['live_metrics'] = {
                'champion': champion_live_metrics,
                'challenger': challenger_live_metrics
            }
        
        return comparison
    
    def promote_challenger(
        self,
        model_name: str,
        champion_alias: str = "champion",
        challenger_alias: str = "challenger",
        backup_alias: str = "previous_champion"
    ) -> None:
        """
        Promote challenger to champion
        
        Args:
            model_name: Name of the registered model
            champion_alias: Current champion alias
            challenger_alias: Challenger alias to promote
            backup_alias: Alias for backing up current champion
        """
        print(f"\n{'='*70}")
        print("PROMOTING CHALLENGER TO CHAMPION")
        print(f"{'='*70}")
        
        model_details = self.client.get_registered_model(model_name)
        
        # Get current versions
        current_champion = model_details.aliases.get(champion_alias)
        current_challenger = model_details.aliases.get(challenger_alias)
        
        if not current_champion:
            raise ValueError(f"No current champion found with alias '{champion_alias}'")
        if not current_challenger:
            raise ValueError(f"No challenger found with alias '{challenger_alias}'")
        
        print(f"\n[*] Current champion: version {current_champion}")
        print(f"[*] Challenger: version {current_challenger}")
        
        # Backup current champion
        print(f"\n[*] Backing up current champion as '{backup_alias}'...")
        self.set_alias(model_name, backup_alias, current_champion)
        
        # Promote challenger to champion
        print(f"[*] Promoting challenger (v{current_challenger}) to champion...")
        self.set_alias(model_name, champion_alias, current_challenger)
        
        # Remove challenger alias from promoted model
        print(f"[*] Removing challenger alias...")
        self.delete_alias(model_name, challenger_alias)
        
        print(f"\n[OK] Promotion completed!")
        print(f"     New champion: version {current_challenger}")
        print(f"     Previous champion backed up as: {backup_alias}")
    
    def rollback_champion(
        self,
        model_name: str,
        champion_alias: str = "champion",
        backup_alias: str = "previous_champion"
    ) -> None:
        """
        Rollback to previous champion
        
        Args:
            model_name: Name of the registered model
            champion_alias: Champion alias
            backup_alias: Backup alias to restore from
        """
        print(f"\n{'='*70}")
        print("ROLLING BACK TO PREVIOUS CHAMPION")
        print(f"{'='*70}")
        
        model_details = self.client.get_registered_model(model_name)
        
        backup_version = model_details.aliases.get(backup_alias)
        
        if not backup_version:
            raise ValueError(f"No backup found with alias '{backup_alias}'")
        
        print(f"\n[*] Restoring version {backup_version} as champion...")
        self.set_alias(model_name, champion_alias, backup_version)
        
        print(f"[OK] Rollback completed!")
        print(f"     Champion restored to version {backup_version}")


def setup_initial_champion_challenger(
    model_name_champion: str,
    model_name_challenger: str,
    champion_version: str = "1",
    challenger_version: str = "1"
):
    """
    Initial setup of champion and challenger models
    
    Args:
        model_name_champion: Model to set as champion
        model_name_challenger: Model to set as challenger
        champion_version: Version of champion model
        challenger_version: Version of challenger model
    """
    manager = ChampionChallengerManager()
    
    print("\n" + "="*70)
    print("INITIAL CHAMPION/CHALLENGER SETUP")
    print("="*70)
    
    print(f"\n[*] Setting up champion: {model_name_champion}")
    manager.set_alias(model_name_champion, "champion", champion_version)
    
    print(f"[*] Setting up challenger: {model_name_challenger}")
    manager.set_alias(model_name_challenger, "challenger", challenger_version)
    
    print("\n[OK] Setup completed!")
    print(f"     Champion: {model_name_champion} (version {champion_version})")
    print(f"     Challenger: {model_name_challenger} (version {challenger_version})")


def main():
    """Main execution for demonstration"""
    
    print("\n" + "="*70)
    print("CHAMPION/CHALLENGER MANAGEMENT SYSTEM")
    print("="*70)
    
    manager = ChampionChallengerManager()
    
    # List all registered models
    print("\n[*] Registered models:")
    models = manager.list_registered_models()
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    
    # Example: Setup XGBoost as champion and Random Forest as challenger
    print("\n" + "="*70)
    print("EXAMPLE: Setting up XGBoost (champion) vs Random Forest (challenger)")
    print("="*70)
    
    try:
        # Set aliases
        manager.set_alias("xgboost_chicago", "champion", "1")
        manager.set_alias("random_forest_chicago", "challenger", "1")
        
        # Show model versions
        print("\n[*] XGBoost versions:")
        xgb_versions = manager.get_model_versions("xgboost_chicago")
        print(xgb_versions.to_string(index=False))
        
        print("\n[*] Random Forest versions:")
        rf_versions = manager.get_model_versions("random_forest_chicago")
        print(rf_versions.to_string(index=False))
        
        # Compare models (using stored metrics)
        comparison = manager.compare_models(
            champion_model_name="xgboost_chicago",
            challenger_model_name="random_forest_chicago",
            champion_alias="champion",
            challenger_alias="challenger"
        )
        
        # Decision logic
        print("\n" + "="*70)
        print("PROMOTION DECISION")
        print("="*70)
        
        mcc_improvement = comparison['test_mcc']['improvement']
        print(f"\nMCC Improvement: {mcc_improvement:+.4f}")
        
        if mcc_improvement > 0:
            print("[INFO] Challenger performs better than champion")
            print("[INFO] Recommendation: PROMOTE challenger")
        else:
            print("[INFO] Champion still performs better")
            print("[INFO] Recommendation: KEEP current champion")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("For full functionality, use the ChampionChallengerManager class")
    print("="*70)


if __name__ == "__main__":
    sys.exit(main())
