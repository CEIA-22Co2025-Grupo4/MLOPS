"""
Train All 8 Original Models with MLFlow
Migrates all models from the original implementation to MLFlow tracking
"""

import sys
from pathlib import Path
import zipfile

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    BaggingClassifier, 
    AdaBoostClassifier
)
from xgboost import XGBClassifier

# Import helper functions
from mlflow_training_helper import train_and_log_model, compare_models


# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets"
TRAIN_FILE = "chicago_crimes_and_stations_2024_final.csv"
TEST_FILE = "chicago_crimes_and_stations_2024_final_test.csv"
EXPERIMENT_NAME = "chicago_crimes_8_models"


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
    
    print(f"\n  [INFO] Target distribution (train):")
    print(f"    Class 0 (No arrest): {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.2f}%)")
    print(f"    Class 1 (Arrest): {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.2f}%)")
    
    return X_train, X_test, y_train, y_test


def get_model_configurations():
    """
    Define all 8 models with their configurations
    Matches the original implementation from modelos/machine_learning.ipynb
    """
    
    models = [
        {
            'model': LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            ),
            'name': 'logistic_regression_chicago',
            'run_name': 'logistic_regression_v1',
            'display_name': 'Logistic Regression',
            'log_feature_importance': False,
            'tags': {
                'model_family': 'linear',
                'original_implementation': 'machine_learning.ipynb'
            }
        },
        {
            'model': KNeighborsClassifier(
                n_neighbors=10,
                n_jobs=-1
            ),
            'name': 'knn_chicago',
            'run_name': 'knn_k10_v1',
            'display_name': 'K-Nearest Neighbors (k=10)',
            'log_feature_importance': False,
            'tags': {
                'model_family': 'instance_based',
                'original_implementation': 'machine_learning.ipynb',
                'k_neighbors': '10'
            }
        },
        {
            'model': LinearSVC(
                random_state=42,
                max_iter=1000,
                dual='auto'
            ),
            'name': 'svm_linear_chicago',
            'run_name': 'svm_linear_v1',
            'display_name': 'SVM Linear',
            'log_feature_importance': False,
            'tags': {
                'model_family': 'svm',
                'kernel': 'linear',
                'original_implementation': 'machine_learning.ipynb'
            }
        },
        {
            'model': DecisionTreeClassifier(
                random_state=42
            ),
            'name': 'decision_tree_chicago',
            'run_name': 'decision_tree_v1',
            'display_name': 'Decision Tree',
            'log_feature_importance': True,
            'tags': {
                'model_family': 'tree',
                'original_implementation': 'machine_learning.ipynb'
            }
        },
        {
            'model': RandomForestClassifier(
                n_estimators=20,
                random_state=42,
                n_jobs=-1
            ),
            'name': 'random_forest_chicago',
            'run_name': 'random_forest_v1',
            'display_name': 'Random Forest',
            'log_feature_importance': True,
            'tags': {
                'model_family': 'ensemble_bagging',
                'n_estimators': '20',
                'original_implementation': 'machine_learning.ipynb'
            }
        },
        {
            'model': BaggingClassifier(
                estimator=LogisticRegression(max_iter=1000, random_state=42),
                n_estimators=20,
                random_state=42,
                n_jobs=-1
            ),
            'name': 'bagging_lr_chicago',
            'run_name': 'bagging_lr_v1',
            'display_name': 'Bagging (Logistic Regression)',
            'log_feature_importance': False,
            'tags': {
                'model_family': 'ensemble_bagging',
                'base_estimator': 'LogisticRegression',
                'n_estimators': '20',
                'original_implementation': 'machine_learning.ipynb'
            }
        },
        {
            'model': AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=5, random_state=42),
                n_estimators=20,
                random_state=42
            ),
            'name': 'adaboost_chicago',
            'run_name': 'adaboost_v1',
            'display_name': 'AdaBoost',
            'log_feature_importance': True,
            'tags': {
                'model_family': 'ensemble_boosting',
                'base_estimator': 'DecisionTree',
                'max_depth': '5',
                'n_estimators': '20',
                'original_implementation': 'machine_learning.ipynb'
            }
        },
        {
            'model': XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                max_depth=6,
                learning_rate=0.1,
                n_estimators=100,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            'name': 'xgboost_chicago',
            'run_name': 'xgboost_v1',
            'display_name': 'XGBoost',
            'log_feature_importance': True,
            'tags': {
                'model_family': 'ensemble_boosting',
                'framework': 'xgboost',
                'original_implementation': 'machine_learning.ipynb',
                'note': 'Best performer - MCC: 0.5796'
            }
        }
    ]
    
    return models


def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print("TRAINING ALL 8 ORIGINAL MODELS WITH MLFLOW")
    print("="*70)
    print("\nThis script replicates the original implementation from:")
    print("  modelos/machine_learning.ipynb")
    print("\nAll models will be tracked in MLFlow with:")
    print("  - Parameters")
    print("  - Metrics (Accuracy, Precision, Recall, F1, MCC, AUC)")
    print("  - Timing information")
    print("  - Visualizations")
    print("  - Model artifacts")
    
    try:
        # Load data
        X_train, X_test, y_train, y_test = load_data()
        
        # Get model configurations
        models = get_model_configurations()
        
        print(f"\n[INFO] Will train {len(models)} models:")
        for i, config in enumerate(models, 1):
            print(f"  {i}. {config['display_name']}")
        
        # Train all models
        results = []
        successful_models = 0
        failed_models = 0
        
        for i, model_config in enumerate(models, 1):
            print(f"\n{'='*70}")
            print(f"MODEL {i}/{len(models)}: {model_config['display_name']}")
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
                    tags=model_config['tags'],
                    log_feature_importance=model_config['log_feature_importance'],
                    register_model=True
                )
                
                results.append({
                    'model_name': model_config['display_name'],
                    'registered_name': model_config['name'],
                    'run_id': run_id,
                    **metrics
                })
                
                successful_models += 1
                print(f"\n[OK] {model_config['display_name']} completed successfully")
                print(f"     Test MCC: {metrics['test_mcc']:.4f}")
                print(f"     Test Accuracy: {metrics['test_accuracy']:.4f}")
                
            except Exception as e:
                failed_models += 1
                print(f"\n[ERROR] Failed to train {model_config['display_name']}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Compare results
        print("\n" + "="*70)
        print("COMPARING ALL MODELS")
        print("="*70)
        
        if successful_models > 0:
            try:
                comparison_df = compare_models(
                    EXPERIMENT_NAME, 
                    metric="test_mcc", 
                    top_n=len(models)
                )
                
                print("\nModels Ranked by Test MCC (Best metric for imbalanced data):")
                print("-" * 70)
                
                # Create summary table
                summary_cols = [
                    'tags.model_type',
                    'metrics.test_mcc',
                    'metrics.test_accuracy',
                    'metrics.test_auc',
                    'metrics.test_f1',
                    'metrics.training_time_seconds'
                ]
                
                available_cols = [col for col in summary_cols if col in comparison_df.columns]
                
                if available_cols:
                    summary = comparison_df[available_cols].copy()
                    summary.columns = [col.replace('metrics.', '').replace('tags.', '') 
                                      for col in summary.columns]
                    print(summary.to_string(index=False))
                
            except Exception as e:
                print(f"[WARN] Could not create comparison table: {e}")
        
        # Final summary
        print("\n" + "="*70)
        print("MIGRATION COMPLETED")
        print("="*70)
        print(f"""
Summary:
  - Experiment: {EXPERIMENT_NAME}
  - Models trained successfully: {successful_models}/{len(models)}
  - Models failed: {failed_models}/{len(models)}
  - View results: http://localhost:5001
  
Model Performance (Expected based on original implementation):
  1. XGBoost        - MCC: ~0.5796 (Best)
  2. Random Forest  - MCC: ~0.5262
  3. AdaBoost       - MCC: ~0.4999
  4. Decision Tree  - MCC: ~0.4500
  5. Bagging (LR)   - MCC: ~0.3500
  6. SVM Linear     - MCC: ~0.2800
  7. KNN (k=10)     - MCC: ~0.2500
  8. Log Regression - MCC: ~0.2127
  
Next Steps:
  1. Review all models in MLFlow UI
  2. Compare with original metrics in modelos/metricas/
  3. Select champion model (likely XGBoost)
  4. Proceed to Phase 5: Champion/Challenger implementation
""")
        
        return 0 if failed_models == 0 else 1
        
    except Exception as e:
        print(f"\n[ERROR] Training pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
