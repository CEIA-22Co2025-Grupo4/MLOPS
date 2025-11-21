"""
Complete Champion/Challenger Workflow Demonstration
Shows the full lifecycle of model promotion and management
"""

import sys
from pathlib import Path
import zipfile

import pandas as pd

from champion_challenger import ChampionChallengerManager


# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets"
TEST_FILE = "chicago_crimes_and_stations_2024_final_test.csv"


def extract_dataset(zip_path):
    """Extract CSV from zip file if needed"""
    zip_path = Path(zip_path)
    csv_path = zip_path.with_suffix('')
    
    if csv_path.exists():
        return csv_path
    
    print(f"[*] Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(zip_path.parent)
    return csv_path


def load_test_data():
    """Load test dataset for live comparison"""
    test_zip = DATASET_DIR / f"{TEST_FILE}.zip"
    test_csv = DATASET_DIR / TEST_FILE
    
    if not test_csv.exists() and test_zip.exists():
        test_csv = extract_dataset(test_zip)
    
    df_test = pd.read_csv(test_csv)
    
    target_col = "Arrest_tag"
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]
    
    return X_test, y_test


def main():
    """
    Complete demonstration of Champion/Challenger workflow
    """
    
    print("\n" + "="*70)
    print("CHAMPION/CHALLENGER WORKFLOW DEMONSTRATION")
    print("="*70)
    print("\nThis demo shows:")
    print("  1. Initial setup of champion and challenger")
    print("  2. Model comparison (stored metrics)")
    print("  3. Live comparison on test data")
    print("  4. Promotion decision")
    print("  5. Model promotion (if applicable)")
    print("  6. Rollback capability")
    
    try:
        # Initialize manager
        manager = ChampionChallengerManager()
        
        # ===================================================================
        # STEP 1: Initial Setup
        # ===================================================================
        print("\n" + "="*70)
        print("STEP 1: INITIAL SETUP")
        print("="*70)
        
        print("\n[*] Setting up XGBoost as Champion (best MCC: 0.5624)")
        manager.set_alias("xgboost_chicago", "champion", "1")
        
        print("[*] Setting up Random Forest as Challenger (MCC: 0.5230)")
        manager.set_alias("random_forest_chicago", "challenger", "1")
        
        # Show current status
        print("\n[*] Current Champion status:")
        xgb_versions = manager.get_model_versions("xgboost_chicago")
        print(xgb_versions[['version', 'aliases', 'status']].to_string(index=False))
        
        print("\n[*] Current Challenger status:")
        rf_versions = manager.get_model_versions("random_forest_chicago")
        print(rf_versions[['version', 'aliases', 'status']].to_string(index=False))
        
        # ===================================================================
        # STEP 2: Compare Models (Stored Metrics)
        # ===================================================================
        print("\n" + "="*70)
        print("STEP 2: COMPARE MODELS (STORED METRICS)")
        print("="*70)
        
        comparison = manager.compare_models(
            champion_model_name="xgboost_chicago",
            challenger_model_name="random_forest_chicago",
            champion_alias="champion",
            challenger_alias="challenger"
        )
        
        # ===================================================================
        # STEP 3: Live Comparison on Test Data
        # ===================================================================
        print("\n" + "="*70)
        print("STEP 3: LIVE COMPARISON ON TEST DATA")
        print("="*70)
        
        print("\n[*] Loading test dataset...")
        X_test, y_test = load_test_data()
        print(f"  [OK] Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        
        live_comparison = manager.compare_models(
            champion_model_name="xgboost_chicago",
            challenger_model_name="random_forest_chicago",
            champion_alias="champion",
            challenger_alias="challenger",
            X_test=X_test,
            y_test=y_test
        )
        
        # ===================================================================
        # STEP 4: Promotion Decision
        # ===================================================================
        print("\n" + "="*70)
        print("STEP 4: PROMOTION DECISION")
        print("="*70)
        
        mcc_improvement = comparison['test_mcc']['improvement']
        champion_mcc = comparison['test_mcc']['champion']
        challenger_mcc = comparison['test_mcc']['challenger']
        
        print(f"\nChampion MCC:    {champion_mcc:.4f}")
        print(f"Challenger MCC:  {challenger_mcc:.4f}")
        print(f"Improvement:     {mcc_improvement:+.4f}")
        
        # Decision threshold
        PROMOTION_THRESHOLD = 0.01  # Require 1% improvement
        
        if mcc_improvement > PROMOTION_THRESHOLD:
            print(f"\n[DECISION] Challenger shows significant improvement (>{PROMOTION_THRESHOLD:.2%})")
            print("[DECISION] RECOMMEND: PROMOTE challenger to champion")
            
            # Ask for confirmation (in production, this would be automatic or manual approval)
            print("\n[INFO] In production, promotion would require approval")
            print("[INFO] For this demo, we'll skip the actual promotion")
            
            # Uncomment to actually promote:
            # manager.promote_challenger(
            #     model_name="xgboost_chicago",
            #     champion_alias="champion",
            #     challenger_alias="challenger"
            # )
            
        else:
            print(f"\n[DECISION] Improvement not significant enough (<{PROMOTION_THRESHOLD:.2%})")
            print("[DECISION] RECOMMEND: KEEP current champion")
            print("[INFO] Challenger can remain for further evaluation")
        
        # ===================================================================
        # STEP 5: Demonstrate Rollback Capability
        # ===================================================================
        print("\n" + "="*70)
        print("STEP 5: ROLLBACK CAPABILITY")
        print("="*70)
        
        print("\n[INFO] Demonstrating rollback capability...")
        print("[INFO] If a promoted model underperforms, we can rollback:")
        print("\n  manager.rollback_champion(")
        print("      model_name='xgboost_chicago',")
        print("      champion_alias='champion',")
        print("      backup_alias='previous_champion'")
        print("  )")
        
        # ===================================================================
        # STEP 6: Alternative Challenger Setup
        # ===================================================================
        print("\n" + "="*70)
        print("STEP 6: TESTING WITH DIFFERENT CHALLENGER")
        print("="*70)
        
        print("\n[*] Let's try AdaBoost as a new challenger...")
        manager.set_alias("adaboost_chicago", "challenger", "1")
        
        comparison_ada = manager.compare_models(
            champion_model_name="xgboost_chicago",
            challenger_model_name="adaboost_chicago",
            champion_alias="champion",
            challenger_alias="challenger"
        )
        
        mcc_improvement_ada = comparison_ada['test_mcc']['improvement']
        print(f"\nAdaBoost MCC Improvement: {mcc_improvement_ada:+.4f}")
        
        if mcc_improvement_ada > PROMOTION_THRESHOLD:
            print("[DECISION] AdaBoost shows improvement - consider for promotion")
        else:
            print("[DECISION] AdaBoost does not outperform champion")
        
        # ===================================================================
        # FINAL SUMMARY
        # ===================================================================
        print("\n" + "="*70)
        print("WORKFLOW COMPLETED")
        print("="*70)
        
        print(f"""
Summary:
  - Champion: XGBoost (MCC: {champion_mcc:.4f})
  - Best Challenger: Random Forest (MCC: {challenger_mcc:.4f})
  - Recommendation: Keep XGBoost as champion
  
Key Features Demonstrated:
  [OK] Model alias management
  [OK] Stored metrics comparison
  [OK] Live performance evaluation
  [OK] Automated promotion decision
  [OK] Rollback capability
  [OK] Multiple challenger evaluation
  
Next Steps:
  1. Monitor champion performance over time
  2. Retrain models with new data
  3. Evaluate new challengers
  4. Promote when significant improvement is found
  
View models in MLFlow UI: http://localhost:5001
""")
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
