import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class ChurnPreprocessor:
    """Data Cleaning, Feature Engineering, and Preprocessing.  """

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders ={}
        self. feature_names = None

    def clean_data(self, df):
        """ Clean adn Validate Data."""
        df_clean = df.copy()

        # Remove customerID (not needed)
        if 'customerID' in df_clean.columns:
            df_clean.drop('customerID', axis=1, inplace=True)

        # TotalCharges should be numeric
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        
        # Remove rows with missing values
        df_clean.dropna(inplace=True)
        print(f" Cleaned data shape: {df_clean.shape}")
        return df_clean
    
    def feature_engineering(self, df):
        """ Create new Features."""
        df_fe = df.copy()

        # 1. Tenure groups (RFM-like segmentation)
        df_fe['TenureGroup'] = pd.cut(df_fe['tenure'], 
                                      bins=[0, 6, 12, 24, 48, 72],
                                      labels=['0-6M', '6-12M', '1-2Y', '2-4Y', '4+Y'])

        df_fe['HighValueCustomer'] = (
            (df_fe['MonthlyCharges'] * df_fe['tenure']) > 
            df_fe['MonthlyCharges'].quantile(0.75) * 12
        ).astype(int)

        # 3. Monthly charges relative to average
        avg_monthly = df_fe['MonthlyCharges'].mean()
        df_fe['HighMonthlyCharges'] = (df_fe['MonthlyCharges'] > avg_monthly).astype(int)
        
        # 4. Customer loyalty (proxy: long tenure + low churn probability)
        df_fe['LoyalCustomer'] = ((df_fe['tenure'] > 24) & 
                                  (df_fe['Contract'] != 'Month-to-month')).astype(int)
        

        # 5. Multi-service user (stickiness indicator)
        online_security = (df_fe['OnlineSecurity'] == 'Yes').astype(int)
        tech_support = (df_fe['TechSupport'] == 'Yes').astype(int)
        backup = (df_fe['OnlineBackup'] == 'Yes').astype(int)
        device_protect = (df_fe['DeviceProtection'] == 'Yes').astype(int)
        
        df_fe['NumServices'] = online_security + tech_support + backup + device_protect
        
        print(f"✅ New features created: TenureGroup, HighValueCustomer, HighMonthlyCharges, LoyalCustomer, NumServices")
        return df_fe
    
    def prepare_features(self, df):
        """Encode categorical variables and prepare X, y."""
        X = df.drop('Churn', axis=1)
        y = df['Churn'].map({'Yes': 1, 'No': 0})
        
        # Identify categorical columns (object and category dtypes)
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # One-hot encoding with drop_first=True
        X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

        self.feature_names = X_encoded.columns.tolist()
        
        print(f"✅ Features after encoding: {X_encoded.shape} features")
        print(f"   Top 10 features: {self.feature_names[:10]}")
        
        return X_encoded, y
    
    def prepare_features_for_prediction(self, df):
        """Encode categorical variables for prediction (no Churn column)."""
        X = df.copy()
        
        # Identify categorical columns (object and category dtypes)
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # One-hot encoding with drop_first=True
        X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        
        # Ensure all expected features are present (add missing with 0)
        for feature in self.feature_names:
            if feature not in X_encoded.columns:
                X_encoded[feature] = 0
        
        # Reorder columns to match training order
        X_encoded = X_encoded[self.feature_names]
        
        return X_encoded
    

    def train_test_split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data maintaining class balance."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y  # Keep churn ratio same in train/test
        )
        
        print(f"✅ Train-test split:")
        print(f"   Training: {X_train.shape} ({y_train.mean()*100:.1f}% churn)")
        print(f"   Testing:  {X_test.shape} ({y_test.mean()*100:.1f}% churn)")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """Standardize numeric features."""
        # Scale all features (after encoding, all are numeric)
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[X_train.columns] = self.scaler.fit_transform(X_train)
        X_test_scaled[X_test.columns] = self.scaler.transform(X_test)
        
        print(f"✅ Features scaled (mean=0, std=1) for {len(X_train.columns)} features")
        
        return X_train_scaled, X_test_scaled
    
    def scale_features_for_prediction(self, X):
        """Standardize numeric features for prediction."""
        # Scale all features
        X_scaled = X.copy()
        X_scaled[X.columns] = self.scaler.transform(X)
        
        return X_scaled
    
    def full_pipeline(self, df):
        """Execute full preprocessing pipeline."""
        print("\n🔄 Starting preprocessing pipeline...\n")
        
        df_clean = self.clean_data(df)
        df_fe = self.feature_engineering(df_clean)
        X, y = self.prepare_features(df_fe)
        X_train, X_test, y_train, y_test = self.train_test_split_data(X, y)
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)

        print("\n✅ Preprocessing complete!\n")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
