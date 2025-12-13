import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.predict import ChurnPredictor

st.set_page_config(page_title="Churn Predictor", layout="wide")

st.title("🎯 Customer Churn Prediction Dashboard")

# Load predictor
predictor = ChurnPredictor()

# Sidebar
st.sidebar.header("Input Customer Data")

# Form inputs
col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0)
    
with col2:
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=780.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No"])
tech_support = st.selectbox("Tech Support", ["Yes", "No"])

# Make prediction
customer_data = {
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'Contract': contract,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'TechSupport': tech_support
}

if st.button("🔮 Predict Churn"):
    result = predictor.predict_single(customer_data)
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Churn Probability", f"{result['churn_probability']:.1%}")
    
    with col2:
        st.metric("Retention Probability", f"{result['retention_probability']:.1%}")
    
    with col3:
        status = "🔴 HIGH RISK" if result['churn_probability'] > 0.5 else "🟢 LOW RISK"
        st.metric("Risk Level", status)
    
    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=result['churn_probability'] * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Probability (%)"},
        delta={'reference': 27},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "lightgreen"},
                {'range': [33, 66], 'color': "lightyellow"},
                {'range': [66, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    st.plotly_chart(fig, use_container_width=True)
