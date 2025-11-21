"""
MLFlow Configuration Verification Script
Verifies connection to MLFlow server and MinIO
"""

import os
import sys

def verify_mlflow_connection():
    """Verifies MLFlow connection and configuration"""
    
    print("=" * 70)
    print("MLFLOW CONFIGURATION VERIFICATION")
    print("=" * 70)
    
    # 1. Verify Python version
    print(f"\n[OK] Python version: {sys.version}")
    
    # 2. Import and verify required libraries
    print("\n[*] Checking installed libraries...")
    try:
        import mlflow
        print(f"  [OK] MLFlow version: {mlflow.__version__}")
    except ImportError as e:
        print(f"  [ERROR] MLFlow is not installed - {e}")
        return False
    
    try:
        import pandas as pd
        print(f"  [OK] Pandas version: {pd.__version__}")
    except ImportError:
        print("  [WARN] Pandas is not installed")
    
    try:
        import sklearn
        print(f"  [OK] Scikit-learn version: {sklearn.__version__}")
    except ImportError:
        print("  [WARN] Scikit-learn is not installed")
    
    try:
        import xgboost
        print(f"  [OK] XGBoost version: {xgboost.__version__}")
    except ImportError:
        print("  [WARN] XGBoost is not installed")
    
    # 3. Configure environment variables for MinIO
    print("\n[*] Configuring environment variables...")
    os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
    os.environ['AWS_ENDPOINT_URL_S3'] = 'http://localhost:9000'
    print("  [OK] MinIO variables configured")
    
    # 4. Configure tracking URI
    mlflow_uri = 'http://localhost:5001'
    mlflow.set_tracking_uri(mlflow_uri)
    print(f"  [OK] Tracking URI configured: {mlflow_uri}")
    
    # 5. Verify server connection
    print("\n[*] Verifying MLFlow server connection...")
    try:
        actual_uri = mlflow.get_tracking_uri()
        print(f"  [OK] Connected to: {actual_uri}")
    except Exception as e:
        print(f"  [ERROR] Failed to get tracking URI: {e}")
        return False
    
    # 6. Try to list experiments
    print("\n[*] Listing existing experiments...")
    try:
        experiments = mlflow.search_experiments()
        print(f"  [OK] Found {len(experiments)} experiments")
        
        if len(experiments) > 0:
            print("\n  Available experiments:")
            for exp in experiments:
                print(f"    - ID: {exp.experiment_id}, Name: {exp.name}")
        else:
            print("  [INFO] No experiments created yet (this is normal for a new installation)")
            
    except Exception as e:
        print(f"  [ERROR] Failed to list experiments: {e}")
        return False
    
    # 7. Create test experiment
    print("\n[*] Creating test experiment...")
    try:
        test_exp_name = "test_verification"
        
        # Check if already exists
        existing = mlflow.get_experiment_by_name(test_exp_name)
        if existing:
            print(f"  [INFO] Experiment '{test_exp_name}' already exists (ID: {existing.experiment_id})")
            exp_id = existing.experiment_id
        else:
            exp_id = mlflow.create_experiment(
                test_exp_name,
                tags={"type": "verification", "purpose": "setup_test"}
            )
            print(f"  [OK] Experiment '{test_exp_name}' created successfully (ID: {exp_id})")
        
        # 8. Create a test run
        print("\n[*] Creating test run...")
        with mlflow.start_run(experiment_id=exp_id, run_name="verification_run") as run:
            # Log test parameters
            mlflow.log_param("test_param", "test_value")
            mlflow.log_param("python_version", sys.version.split()[0])
            
            # Log test metrics
            mlflow.log_metric("test_metric", 0.95)
            mlflow.log_metric("accuracy", 0.91)
            
            print(f"  [OK] Run created successfully (ID: {run.info.run_id})")
            print(f"  [INFO] View this run at: {mlflow_uri}/#/experiments/{exp_id}/runs/{run.info.run_id}")
        
    except Exception as e:
        print(f"  [ERROR] Failed to create experiment/run: {e}")
        return False
    
    # 9. Final summary
    print("\n" + "=" * 70)
    print("[OK] VERIFICATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"""
Summary:
  - MLFlow Server: [OK] Running at {mlflow_uri}
  - MinIO (S3): [OK] Configured
  - Connection: [OK] Established
  - Experiments: [OK] Working
  - Runs: [OK] Working
  
Important URLs:
  - MLFlow UI: http://localhost:5001
  - MinIO Console: http://localhost:9001 (user: minio, password: minio123)
  
[OK] System is ready. You can proceed with PHASE 2.
""")
    
    return True


if __name__ == "__main__":
    try:
        success = verify_mlflow_connection()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
