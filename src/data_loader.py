import pandas as pd
import numpy as np
import os
from pathlib import Path

class ChurnDataLoader:
    """Load and Validate Telco Customer CHurn dataset."""

    def __init__(self, data_path = 'data'):
        self.data_path = data_path

    def download_dataset(self):
        """Download Teclo Churn dataset from Kaggle or web source."""
        url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

        os.makedirs(self.data_path, exist_ok = True)
        file_path = f"{self.data_path}/telco-churn.csv"

        if not os.path.exists(file_path):
            print(f"Downloading dataset from {url} to {file_path}...")
            df = pd.read_csv(url)
            df.to_csv(file_path, index = False)
            print(f"Dataset saved to {file_path}")
        else:
            print(f"Dataset already exists at {file_path}")
        
        return file_path

    def load_dataset(self):
        """Load dataset with Validation."""
        file_path = self.download_dataset()

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found at {file_path}")
        
        df = pd.read_csv(file_path)

        print(f"\n Dataset Info:")
        print(f" Shape: {df.shape} rows, {df.shape} columns")
        print(f" Missing Values: {df.isnull().sum().sum()}")
        print(f" Churn distribution:\n {df['Churn'].value_counts(normalize=True)}") 

        return df
    
    def get_data_summary(self, df):
        """Get statistical summary."""
        return {
            'shape' : df.shape,
            'missing values' : df.isnull().sum().sum(),
            'dtypes' : df.dtypes.value_counts().to_dict(),
            'churn_rate' : (df['Churn'] == 'Yes').mean()
        }
    

