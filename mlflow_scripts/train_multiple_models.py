"""
Train Multiple Models Example
Demonstrates how to use mlflow_training_helper to train multiple models
"""

import sys
from pathlib import Path
import zipfile

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Import helper functions
from mlflow_training_helper import train_and_log_model, compare_models


# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets"
TRAIN_FILE = "chicago_crimes_and_stations_2024_final.csv"
TEST_FILE = "chicago_crimes_and_stations_2024_final_test.csv"
EXPERIMENT_NAME = "chicago_crimes_all_models"


def extract_dataset(zip_path):
    """Extract CSV from zip file if needed"""
    zip_path = Path(zip_path)
    csv_path = zip_path.with_suffix('')
    
    if csv_path.exists():
        return csv_path
    
    print(f"[*] Extracting {zip_path.name}...")
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
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    
    print(f"  [OK] Train set: {df_train.shape[0]} rows, {df_train.shape[1]} columns")
    print(f"  [OK] Test set: {df_test.shape[0]} rows, {df_test.shape[1]} columns")
    
    # Split features and target
    target_col = "Arrest_tag"
    
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]
    
    return X_train, X_test, y_train, y_test


def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print("TRAINING MULTIPLE MODELS WITH MLFLOW")
    print("="*70)
    
    try:
        # Load data
        X_train, X_test, y_train, y_test = load_data()
        
        # Define models to train
        models = [
            {
                'model': LogisticRegression(max_iter=1000, random_state=42),
                'name': 'logistic_regression_chicago',
                'run_name': 'logistic_regression_v1',
                'log_feature_importance': False
            },
            {
                'model': DecisionTreeClassifier(max_depth=10, random_state=42),
                'name': 'decision_tree_chicago',
                'run_name': 'decision_tree_v1',
                'log_feature_importance': True
            },
            {
                'model': RandomForestClassifier(n_estimators=20, max_depth=10, random_state=42, n_jobs=-1),
                'name': 'random_forest_chicago',
                'run_name': 'random_forest_v1',
                'log_feature_importance': True
            },
            {
                'model': AdaBoostClassifier(
                    estimator=DecisionTreeClassifier(max_depth=5),
                    n_estimators=20,
                    random_state=42
                ),
                'name': 'adaboost_chicago',
                'run_name': 'adaboost_v1',
                'log_feature_importance': True
            },
            {
                'model': XGBClassifier(
                    max_depth=6,
                    learning_rate=0.1,
                    n_estimators=100,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                ),
                'name': 'xgboost_chicago_v2',
                'run_name': 'xgboost_v2',
                'log_feature_importance': True
            }
        ]
        
        # Train all models
        results = []
        
        for i, model_config in enumerate(models, 1):
            print(f"\n{'='*70}")
            print(f"MODEL {i}/{len(models)}: {model_config['name']}")
            print(f"{'='*70}")
            
            try:
                model, run_id, metrics = train_and_log_model(
                    model=model_config['model'],
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    experiment_name=EXPERIMENT_NAME,
                    run_name=model_config['run_name'],
                    model_name=model_config['name'],
                    tags={'model_family': model_config['model'].__class__.__name__},
                    log_feature_importance=model_config['log_feature_importance'],
                    register_model=True
                )
                
                results.append({
                    'model_name': model_config['name'],
                    'run_id': run_id,
                    **metrics
                })
                
                print(f"\n[OK] {model_config['name']} completed successfully")
                
            except Exception as e:
                print(f"\n[ERROR] Failed to train {model_config['name']}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Compare results
        print("\n" + "="*70)
        print("COMPARING ALL MODELS")
        print("="*70)
        
        comparison_df = compare_models(EXPERIMENT_NAME, metric="test_mcc", top_n=10)
        
        print("\nTop Models by Test MCC:")
        print(comparison_df[['run_id', 'metrics.test_mcc', 'metrics.test_accuracy', 
                            'metrics.test_auc', 'tags.model_type']].to_string())
        
        # Summary
        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print("="*70)
        print(f"""
Summary:
  - Experiment: {EXPERIMENT_NAME}
  - Models trained: {len(results)}/{len(models)}
  - View results: http://localhost:5001
  
Next Steps:
  1. Review model performance in MLFlow UI
  2. Select champion model based on MCC score
  3. Proceed with Champion/Challenger implementation
""")
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
