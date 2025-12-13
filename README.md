# Customer Churn Prediction

Predict customer churn probability for SaaS/Telecom companies using ML.

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Training
python src/train.py

# Predictions
python src/predict.py

# Dashboard
streamlit run app.py
