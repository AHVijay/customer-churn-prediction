import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

class ChurnPredictor:
    """Make predictions on new customer data."""
    
    def __init__(self, model_path='models/churn_model.pkl',
                 preprocessor_path='models/preprocessor.pkl'):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        print("✅ Model and preprocessor loaded")
    
    def predict_single(self, customer_data_dict):
        """
        Predict churn for a single customer.
        
        Args:
            customer_data_dict: Dict with customer features
            
        Returns:
            Dict with prediction and probability
        """
        # Default values for missing features
        defaults = {
            'gender': 'Male',
            'SeniorCitizen': 0,
            'Partner': 'No',
            'Dependents': 'No',
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check'
        }
        
        # Merge provided data with defaults
        full_data = {**defaults, **customer_data_dict}
        
        # Convert to DataFrame
        df = pd.DataFrame([full_data])
        
        # Apply preprocessing
        df_clean = self.preprocessor.clean_data(df)
        df_fe = self.preprocessor.feature_engineering(df_clean)
        X = self.preprocessor.prepare_features_for_prediction(df_fe)
        X_scaled = self.preprocessor.scale_features_for_prediction(X)
        
        # Predict
        pred = self.model.predict(X_scaled)
        proba = self.model.predict_proba(X_scaled)
        
        return {
            'will_churn': bool(pred[0]),
            'churn_probability': float(proba[0][1]),
            'retention_probability': float(proba[0][0])
        }
    
    def predict_batch(self, df):
        """Predict churn for multiple customers."""
        df_clean = self.preprocessor.clean_data(df)
        df_fe = self.preprocessor.feature_engineering(df_clean)
        X, _ = self.preprocessor.prepare_features(df_fe)
        X_scaled = self.preprocessor.scale_features(X, X)
        
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        results = pd.DataFrame({
            'will_churn': predictions.astype(bool),
            'churn_probability': probabilities
        })
        
        return results


# Example usage
if __name__ == "__main__":
    predictor = ChurnPredictor()
    
    # Single prediction
    customer = {
        'tenure': 12,
        'MonthlyCharges': 65.0,
        'TotalCharges': 780.0,
        'Contract': 'Month-to-month',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'TechSupport': 'No'
        # Add all other required columns
    }
    
    result = predictor.predict_single(customer)
    print(f"Churn Probability: {result['churn_probability']:.2%}")
    print(f"Will Churn: {result['will_churn']}")
