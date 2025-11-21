"""
Check registered models in MLFlow
"""

import os
import mlflow
from mlflow.tracking import MlflowClient

# Setup
os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
os.environ['AWS_ENDPOINT_URL_S3'] = 'http://localhost:9000'

mlflow.set_tracking_uri("http://localhost:5001")

client = MlflowClient()

# Get registered models
print("\n[*] Checking registered models...\n")

try:
    registered_models = client.search_registered_models()
    
    if len(registered_models) == 0:
        print("  [INFO] No registered models found yet")
    else:
        for model in registered_models:
            print(f"Model Name: {model.name}")
            print(f"  Description: {model.description}")
            print(f"  Latest Versions:")
            
            versions = client.search_model_versions(f"name='{model.name}'")
            for version in versions[:5]:  # Show top 5 versions
                print(f"    - Version {version.version}: Stage={version.current_stage}, "
                      f"Run ID={version.run_id}")
            print()
    
    # Check experiment runs
    print("\n[*] Recent runs in 'chicago_crimes_xgboost' experiment:\n")
    experiment = mlflow.get_experiment_by_name("chicago_crimes_xgboost")
    
    if experiment:
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=5
        )
        
        if len(runs) > 0:
            print(f"Found {len(runs)} runs:\n")
            for idx, row in runs.iterrows():
                print(f"Run {idx + 1}:")
                print(f"  Run ID: {row['run_id']}")
                print(f"  Status: {row['status']}")
                print(f"  Test Accuracy: {row.get('metrics.test_accuracy', 'N/A')}")
                print(f"  Test MCC: {row.get('metrics.test_mcc', 'N/A')}")
                print()
        else:
            print("  [INFO] No runs found")
    else:
        print("  [WARN] Experiment not found")
        
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()
