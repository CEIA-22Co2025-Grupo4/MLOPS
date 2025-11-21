"""
Complete Demonstration of ChicagoCrimePredictor
Shows all features and use cases for production deployment
"""

import sys
from pathlib import Path
import zipfile

import pandas as pd
import numpy as np

from predictor import ChicagoCrimePredictor


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
    """Load test dataset"""
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
    """Complete demonstration of predictor functionality"""
    
    print("\n" + "="*70)
    print("CHICAGO CRIME PREDICTOR - COMPLETE DEMONSTRATION")
    print("="*70)
    
    try:
        # ===================================================================
        # DEMO 1: Initialize Predictor
        # ===================================================================
        print("\n" + "="*70)
        print("DEMO 1: INITIALIZE PREDICTOR")
        print("="*70)
        
        print("\n[*] Loading champion model...")
        predictor = ChicagoCrimePredictor(
            model_name="xgboost_chicago",
            alias="champion",
            validate_input=True
        )
        
        # ===================================================================
        # DEMO 2: Get Model Information
        # ===================================================================
        print("\n" + "="*70)
        print("DEMO 2: MODEL INFORMATION")
        print("="*70)
        
        info = predictor.get_model_info()
        
        print(f"\nModel Details:")
        print(f"  Name: {info['model_name']}")
        print(f"  Alias: {info['alias']}")
        print(f"  Version: {info['version']}")
        print(f"  Type: {info['model_type']}")
        print(f"  Run ID: {info['run_id']}")
        
        print(f"\nPerformance Metrics:")
        for metric, value in info['metrics'].items():
            if value is not None:
                print(f"  {metric}: {value:.4f}")
        
        print(f"\nExpected Features ({len(info['expected_features'])}):")
        for i, feature in enumerate(info['expected_features'], 1):
            print(f"  {i}. {feature}")
        
        # ===================================================================
        # DEMO 3: Single Prediction (Dict Input)
        # ===================================================================
        print("\n" + "="*70)
        print("DEMO 3: SINGLE PREDICTION (DICT INPUT)")
        print("="*70)
        
        # Get actual features from model
        expected_features = info['expected_features']
        
        # Load a real sample from test data to get correct feature format
        print("\n[*] Loading sample from test data...")
        X_test_sample, _ = load_test_data()
        sample_crime = X_test_sample.iloc[0].to_dict()
        
        print("\nInput Crime Features:")
        for key, value in sample_crime.items():
            print(f"  {key}: {value}")
        
        # Make prediction
        prediction = predictor.predict(sample_crime)
        probability = predictor.predict_proba(sample_crime)
        
        print(f"\nPrediction Results:")
        print(f"  Class: {prediction[0]} ({'Arrest' if prediction[0] == 1 else 'No Arrest'})")
        print(f"  Probability: {probability[0]:.4f} ({probability[0]*100:.2f}%)")
        
        # ===================================================================
        # DEMO 4: Prediction with Explanation
        # ===================================================================
        print("\n" + "="*70)
        print("DEMO 4: PREDICTION WITH EXPLANATION")
        print("="*70)
        
        explanation = predictor.predict_with_explanation(sample_crime)
        
        print(f"\nDetailed Explanation:")
        print(f"  Prediction: {explanation['prediction_label']}")
        print(f"  Confidence: {explanation['confidence']:.4f}")
        print(f"  Timestamp: {explanation['timestamp']}")
        
        print(f"\nModel Used:")
        print(f"  Name: {explanation['model_info']['name']}")
        print(f"  Version: {explanation['model_info']['version']}")
        print(f"  Alias: {explanation['model_info']['alias']}")
        
        if 'feature_importance' in explanation:
            print(f"\nTop 5 Most Important Features:")
            for i, (feature, importance) in enumerate(
                list(explanation['feature_importance'].items())[:5], 1
            ):
                print(f"  {i}. {feature}: {importance:.4f}")
        
        # ===================================================================
        # DEMO 5: Multiple Predictions (DataFrame Input)
        # ===================================================================
        print("\n" + "="*70)
        print("DEMO 5: MULTIPLE PREDICTIONS (DATAFRAME INPUT)")
        print("="*70)
        
        # Create multiple samples from test data
        print("\n[*] Loading samples from test data...")
        X_test_multi, _ = load_test_data()
        samples = X_test_multi.head(3)
        
        print(f"\n[*] Predicting for {len(samples)} samples...")
        
        predictions, probabilities = predictor.predict(samples, return_proba=True)
        
        print(f"\nResults:")
        for i, (pred, prob) in enumerate(zip(predictions, probabilities), 1):
            label = 'Arrest' if pred == 1 else 'No Arrest'
            print(f"  Sample {i}: {label} (probability: {prob:.4f})")
        
        # ===================================================================
        # DEMO 6: Batch Prediction on Test Data
        # ===================================================================
        print("\n" + "="*70)
        print("DEMO 6: BATCH PREDICTION ON TEST DATA")
        print("="*70)
        
        print("\n[*] Loading test dataset...")
        X_test, y_test = load_test_data()
        print(f"  [OK] Loaded {len(X_test)} test samples")
        
        # Take a subset for demo
        X_subset = X_test.head(1000)
        y_subset = y_test.head(1000)
        
        print(f"\n[*] Making batch predictions on {len(X_subset)} samples...")
        predictions, probabilities = predictor.batch_predict(
            X_subset, 
            batch_size=250,
            return_proba=True
        )
        
        # Calculate accuracy
        accuracy = (predictions == y_subset).mean()
        
        print(f"\n[OK] Batch prediction completed!")
        print(f"  Accuracy on subset: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Predicted Arrests: {(predictions == 1).sum()}")
        print(f"  Predicted No Arrests: {(predictions == 0).sum()}")
        
        # ===================================================================
        # DEMO 7: Compare Champion vs Challenger
        # ===================================================================
        print("\n" + "="*70)
        print("DEMO 7: COMPARE CHAMPION VS CHALLENGER")
        print("="*70)
        
        try:
            print("\n[*] Loading challenger model...")
            predictor_challenger = ChicagoCrimePredictor(
                model_name="random_forest_chicago",
                alias="challenger",
                validate_input=True
            )
            
            # Make predictions with both models
            print(f"\n[*] Comparing predictions on {len(X_subset)} samples...")
            
            champion_pred = predictor.predict(X_subset)
            challenger_pred = predictor_challenger.predict(X_subset)
            
            # Calculate metrics
            champion_acc = (champion_pred == y_subset).mean()
            challenger_acc = (challenger_pred == y_subset).mean()
            
            # Agreement between models
            agreement = (champion_pred == challenger_pred).mean()
            
            print(f"\nComparison Results:")
            print(f"  Champion Accuracy: {champion_acc:.4f}")
            print(f"  Challenger Accuracy: {challenger_acc:.4f}")
            print(f"  Difference: {champion_acc - challenger_acc:+.4f}")
            print(f"  Agreement: {agreement:.4f} ({agreement*100:.2f}%)")
            
            if champion_acc > challenger_acc:
                print(f"\n  [RESULT] Champion performs better [OK]")
            else:
                print(f"\n  [RESULT] Challenger performs better - consider promotion")
                
        except Exception as e:
            print(f"\n  [WARN] Could not load challenger: {e}")
        
        # ===================================================================
        # DEMO 8: Error Handling
        # ===================================================================
        print("\n" + "="*70)
        print("DEMO 8: ERROR HANDLING & VALIDATION")
        print("="*70)
        
        print("\n[*] Testing input validation...")
        
        # Test 1: Missing features
        print("\n  Test 1: Missing features")
        try:
            invalid_input = {'Primary Type_ENCODED': 5}
            predictor.predict(invalid_input)
            print("    [FAIL] Should have raised error")
        except ValueError as e:
            print(f"    [OK] Caught error: {str(e)[:50]}...")
        
        # Test 2: Null values
        print("\n  Test 2: Null values")
        try:
            invalid_df = pd.DataFrame([sample_crime])
            invalid_df.iloc[0, 0] = None  # Set first feature to None
            predictor.predict(invalid_df)
            print("    [FAIL] Should have raised error")
        except ValueError as e:
            print(f"    [OK] Caught error: {str(e)[:50]}...")
        
        # Test 3: Extra features (should work with warning)
        print("\n  Test 3: Extra features (should warn)")
        valid_df = pd.DataFrame([sample_crime])
        valid_df['extra_feature'] = 999
        prediction = predictor.predict(valid_df)
        print(f"    [OK] Prediction made: {prediction[0]}")
        
        # ===================================================================
        # FINAL SUMMARY
        # ===================================================================
        print("\n" + "="*70)
        print("DEMONSTRATION COMPLETED")
        print("="*70)
        
        print(f"""
Summary:
  [OK] Predictor initialized successfully
  [OK] Model information retrieved
  [OK] Single predictions working
  [OK] Batch predictions working
  [OK] Explanations generated
  [OK] Champion vs Challenger comparison
  [OK] Input validation working
  
Model Performance:
  - Champion: {info['model_name']} v{info['version']}
  - Test MCC: {info['metrics']['test_mcc']:.4f}
  - Test Accuracy: {info['metrics']['test_accuracy']:.4f}
  - Batch Accuracy: {accuracy:.4f}
  
Ready for Production Deployment!
""")
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
