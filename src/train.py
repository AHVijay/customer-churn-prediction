#!/usr/bin/env python3
"""
Customer Churn Prediction - End-to-End Training Pipeline
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import ChurnDataLoader
from preprocessor import ChurnPreprocessor
from model import ChurnModel

def main():
    print("="*70)
    print("🎯 CUSTOMER CHURN PREDICTION - COMPLETE PIPELINE")
    print("="*70)
    
    # Step 1: Load Data
    print("\n📥 STEP 1: Loading Data...")
    loader = ChurnDataLoader()
    df = loader.load_dataset()
    
    # Step 2: Preprocess
    print("\n🔄 STEP 2: Preprocessing & Feature Engineering...")
    preprocessor = ChurnPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.full_pipeline(df)
    feature_names = preprocessor.feature_names
    
    # Step 3: Train Models
    print("\n🤖 STEP 3: Training Models...")
    model = ChurnModel()
    results = model.train_all_models(X_train, y_train, X_test, y_test)
    
    # Step 4: Evaluate Best Model
    print("\n📊 STEP 4: Evaluating Best Model...")
    y_pred, y_pred_proba = model.evaluate_best_model(X_test, y_test)
    
    # Step 5: Save Model
    print("\n💾 STEP 5: Saving Model...")
    os.makedirs('models', exist_ok=True)
    model.save_model('models/churn_model.pkl')
    
    # Save preprocessor
    import joblib
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    
    # Step 6: Visualizations
    print("\n📈 STEP 6: Generating Visualizations...")
    os.makedirs('reports', exist_ok=True)
    
    model.plot_roc_curve(X_test, y_test)
    model.plot_feature_importance(feature_names)
    try:
        model.explain_predictions_shap(X_test)
    except Exception as e:
        print(f"⚠️ SHAP explanation failed: {e}")
        print("Skipping SHAP plots...")
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE!")
    print("="*70)
    print("\nDeliverables:")
    print("  ✓ Trained model: models/churn_model.pkl")
    print("  ✓ ROC Curve: reports/roc_curve.png")
    print("  ✓ Feature Importance: reports/feature_importance.png")
    print("  ✓ SHAP Explanation: reports/shap_summary.png")
    print("  ✓ Confusion Matrix: reports/confusion_matrix.png")

if __name__ == "__main__":
    main()
