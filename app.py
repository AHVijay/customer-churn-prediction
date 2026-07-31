import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.predict import ChurnPredictor

st.set_page_config(page_title="Churn Predictor", layout="wide", initial_sidebar_state="expanded")

# Custom styling
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .high-risk { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .low-risk { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
</style>
""", unsafe_allow_html=True)

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("# 🎯 Customer Churn Prediction Dashboard")
    st.markdown("*Predict customer churn risk and retention probability using AI*")
with col2:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)

st.divider()

# Load predictor
predictor = ChurnPredictor()

# Create two columns: inputs and info
col_input, col_info = st.columns([2, 1])

with col_info:
    st.markdown("### 📊 About This Tool")
    st.info("""
    **What does this predict?**
    - Whether a customer is likely to leave (churn) or stay
    
    **What do you need?**
    - Customer billing and service details
    
    **Accuracy:**
    - Model trained on 7,000+ customer records
    - XGBoost algorithm for high precision
    """)

with col_input:
    st.markdown("### 📋 Customer Information")
    
    # Create form
    with st.form("customer_form", border=True):
        st.markdown("**Billing Details**")
        col1, col2 = st.columns(2)
        
        with col1:
            tenure = st.number_input(
                "Tenure (months)",
                min_value=0, max_value=100, value=12,
                help="How long the customer has been with the company"
            )
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0, max_value=200.0, value=65.0,
                help="Monthly billing amount"
            )
        
        with col2:
            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0, max_value=10000.0, value=780.0,
                help="Total amount paid to date"
            )
            contract = st.selectbox(
                "Contract Type",
                ["Month-to-month", "One year", "Two year"],
                help="Length of customer contract"
            )
        
        st.markdown("**Services**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"],
                help="Type of internet service"
            )
        
        with col2:
            online_security = st.selectbox(
                "Online Security",
                ["Yes", "No"],
                help="Has online security service"
            )
        
        with col3:
            tech_support = st.selectbox(
                "Tech Support",
                ["Yes", "No"],
                help="Has technical support service"
            )
        
        st.divider()
        
        # Submit button
        submitted = st.form_submit_button(
            "🔮 Predict Churn", use_container_width=True,
            type="primary"
        )

# Display results
if submitted:
    customer_data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'TechSupport': tech_support
    }
    
    with st.spinner("🤖 Analyzing customer data..."):
        result = predictor.predict_single(customer_data)
    
    st.divider()
    st.markdown("### 📈 Prediction Results")
    
    # Results metrics
    col1, col2, col3 = st.columns(3)
    
    churn_prob = result['churn_probability']
    retention_prob = result['retention_probability']
    
    with col1:
        st.metric(
            "Churn Risk",
            f"{churn_prob:.1%}",
            delta=f"{(churn_prob - 0.27) * 100:+.1f}% vs avg",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            "Retention Probability",
            f"{retention_prob:.1%}",
            delta=f"{(retention_prob - 0.73) * 100:+.1f}% vs avg"
        )
    
    with col3:
        risk_level = "HIGH RISK" if churn_prob > 0.5 else "MEDIUM RISK" if churn_prob > 0.3 else "LOW RISK"
        risk_color = "🔴" if churn_prob > 0.5 else "🟡" if churn_prob > 0.3 else "🟢"
        st.metric("Risk Classification", f"{risk_color} {risk_level}")
    
    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=churn_prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Probability (%)"},
        delta={'reference': 27, 'suffix': "% vs industry avg"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "rgba(79, 195, 247, 0.3)"},
                {'range': [33, 66], 'color': "rgba(255, 193, 7, 0.3)"},
                {'range': [66, 100], 'color': "rgba(244, 67, 54, 0.3)"}
            ],
            'threshold': {
                'line': {'color': "darkred", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(height=400, font={"size": 12})
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommendation
    st.divider()
    
    if churn_prob > 0.5:
        st.warning(
            f"⚠️ **High Churn Risk** ({churn_prob:.0%})\n\n"
            f"**Recommended Actions:**\n"
            f"- Review customer satisfaction\n"
            f"- Offer retention incentives\n"
            f"- Increase support engagement"
        )
    elif churn_prob > 0.3:
        st.info(
            f"ℹ️ **Moderate Churn Risk** ({churn_prob:.0%})\n\n"
            f"**Recommended Actions:**\n"
            f"- Monitor customer activity\n"
            f"- Proactive customer outreach\n"
            f"- Offer loyalty rewards"
        )
    else:
        st.success(
            f"✅ **Low Churn Risk** ({churn_prob:.0%})\n\n"
            f"**Recommended Actions:**\n"
            f"- Maintain current service level\n"
            f"- Continue engagement programs\n"
            f"- Explore upsell opportunities"
        )
