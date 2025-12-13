# 📊 Customer Churn Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Predict customer churn probability for SaaS/Telecom companies using Machine Learning. This project includes a complete ML pipeline with data preprocessing, model training, evaluation, and a web dashboard for real-time predictions.

## 🚀 Features

- **Complete ML Pipeline**: Data loading, preprocessing, feature engineering, model training
- **Multiple Algorithms**: Logistic Regression, Random Forest, XGBoost
- **Web Dashboard**: Interactive Streamlit app for real-time predictions
- **Model Evaluation**: ROC curves, confusion matrices, feature importance
- **Data Visualization**: SHAP explanations, performance metrics

## 📈 Model Performance

- **Accuracy**: 80%
- **AUC Score**: 0.838
- **F1 Score**: 0.584

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd customer-churn-prediction

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start (Step-by-Step)

Follow these steps to get the project running:

### Step 1: Clone and Setup
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd customer-churn-prediction

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download Dataset
The dataset is not included in the repository. Download it from Kaggle:
1. Go to [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
2. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv`
3. Place the file in the `data/` directory as `telco-churn.csv`

### Step 4: Train the Model
```bash
python src/train.py
```
This will:
- Load and preprocess the data
- Train multiple ML models (Logistic Regression, Random Forest, XGBoost)
- Save the best model and preprocessor
- Generate evaluation reports and visualizations

### Step 5: Run the Web Dashboard
```bash
streamlit run app.py
```
- Open your browser to `http://localhost:8501`
- Enter customer details in the sidebar
- Click "Predict Churn" to get real-time predictions

### Step 6: Make API Predictions (Optional)
```python
from src.predict import ChurnPredictor

# Initialize predictor
predictor = ChurnPredictor()

# Make prediction
customer_data = {
    'tenure': 12,
    'MonthlyCharges': 65.0,
    'TotalCharges': 780.0,
    'Contract': 'Month-to-month',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'TechSupport': 'No'
}

result = predictor.predict_single(customer_data)
print(f"Churn Probability: {result['churn_probability']:.1%}")
```

## 🎯 Usage

### Training the Model
```bash
python src/train.py
```

### Running the Web Dashboard
```bash
streamlit run app.py
```
Access the dashboard at `http://localhost:8501`

### Making Predictions (API)
```python
from src.predict import ChurnPredictor

predictor = ChurnPredictor()
result = predictor.predict_single({
    'tenure': 12,
    'MonthlyCharges': 65.0,
    'TotalCharges': 780.0,
    'Contract': 'Month-to-month',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'TechSupport': 'No'
})

print(f"Churn Probability: {result['churn_probability']:.1%}")
```

## 🔧 Troubleshooting

### Common Issues:

**1. Module Import Errors**
```bash
# If you get import errors, ensure you're in the correct directory
cd customer-churn-prediction

# And virtual environment is activated
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
```

**2. Dataset Not Found**
- Ensure `data/telco-churn.csv` exists
- Download from Kaggle if missing
- Check file permissions

**3. Model Files Missing**
- Run `python src/train.py` to generate models
- Check `models/` directory for `.pkl` files

**4. Port Already in Use**
```bash
# Kill process using port 8501
# Windows:
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# Or run on different port:
streamlit run app.py --server.port 8502
```

**5. Memory Issues**
- XGBoost training might require more RAM
- Reduce data size or use simpler models if needed

**6. Permission Errors**
```bash
# On Windows, run terminal as Administrator
# Or check write permissions for models/ and reports/ directories
```

### Getting Help:
- Check the terminal output for specific error messages
- Ensure all dependencies are installed: `pip list`
- Verify Python version: `python --version`

## 📁 Project Structure

```
customer-churn-prediction/
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Data acquisition and loading
│   ├── preprocessor.py     # Data cleaning and feature engineering
│   ├── model.py           # Model training and evaluation
│   ├── predict.py         # Prediction interface
│   └── train.py           # Main training pipeline
├── app.py                 # Streamlit web dashboard
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .gitignore           # Git ignore rules
├── data/                # Dataset storage (excluded from git)
├── models/              # Trained models (excluded from git)
├── reports/             # Generated reports and plots (excluded from git)
└── notebooks/           # Jupyter notebooks for EDA
```

## 🔧 Dependencies

- pandas
- numpy
- scikit-learn
- xgboost
- streamlit
- plotly
- joblib
- shap

## 📊 Dataset

The project uses the Telco Customer Churn dataset from Kaggle, which contains customer information and churn status for a telecommunications company.

**Features:**
- Customer demographics (gender, age, partner, dependents)
- Service information (phone, internet, security, support)
- Account information (tenure, contract type, billing)
- Churn status (target variable)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Telco Customer Churn dataset from [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)
- Inspired by various customer analytics projects
