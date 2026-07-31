# 🏗️ Project Architecture & Flow

## 📌 Quick Start

### Two Entry Points

#### 1. **Training Pipeline** (One-time setup)
```bash
python src/train.py
```
- Loads raw data → Trains models → Saves artifacts → Generates reports

#### 2. **Streamlit Dashboard** (Daily use)
```bash
streamlit run app.py
```
- Web UI for making predictions on new customers

---

## 🚀 Execution Flow

### **Flow 1: Training Pipeline** (`python src/train.py`)

```
src/train.py (main orchestrator)
    │
    ├─→ ChurnDataLoader (src/data_loader.py)
    │   └─→ downloads & loads: data/telco-churn.csv
    │
    ├─→ ChurnPreprocessor (src/preprocessor.py)
    │   ├─→ clean_data() - handle missing values, type conversions
    │   ├─→ feature_engineering() - create new features
    │   ├─→ encode_features() - categorical encoding
    │   ├─→ scale_features() - normalize numerical data
    │   └─→ train_test_split() - split 80/20
    │
    ├─→ ChurnModel (src/model.py)
    │   ├─→ train_all_models() - LR, RF, XGB
    │   ├─→ evaluate_best_model() - metrics & confusion matrix
    │   ├─→ plot_roc_curve() - ROC/AUC visualization
    │   ├─→ plot_feature_importance() - top 15 features
    │   └─→ explain_predictions_shap() - SHAP summaries
    │
    └─→ Save Artifacts
        ├─→ models/churn_model.pkl (trained best model)
        ├─→ models/preprocessor.pkl (scaler, encoders)
        ├─→ models/feature_names.pkl (feature list)
        └─→ reports/ (visualizations & metrics)
```

**Output Files Created:**
- `models/churn_model.pkl` - Best trained model (XGB/RF/LR)
- `models/preprocessor.pkl` - Data transformation pipeline
- `reports/confusion_matrix.png` - Prediction accuracy matrix
- `reports/roc_curve.png` - ROC-AUC visualization
- `reports/feature_importance.png` - Top 15 important features
- `reports/shap_summary.png` - SHAP value explanations

---

### **Flow 2: Streamlit Dashboard** (`streamlit run app.py`)

```
app.py (Streamlit web interface)
    │
    ├─→ Load ChurnPredictor (src/predict.py)
    │   │
    │   ├─→ Load models/churn_model.pkl (trained model)
    │   ├─→ Load models/preprocessor.pkl (transformations)
    │   └─→ ChurnPredictor class initialized
    │
    ├─→ Display UI
    │   ├─→ Input form (tenure, charges, contract, services)
    │   └─→ Help text & info panels
    │
    └─→ On "Predict Churn" button click:
        │
        ├─→ Collect customer data from form
        │
        ├─→ ChurnPredictor.predict_single() (src/predict.py)
        │   │
        │   ├─→ preprocessor.clean_data()
        │   ├─→ preprocessor.feature_engineering()
        │   ├─→ preprocessor.prepare_features_for_prediction()
        │   ├─→ preprocessor.scale_features_for_prediction()
        │   ├─→ model.predict() & model.predict_proba()
        │   │
        │   └─→ Return: churn_probability, retention_probability
        │
        └─→ Display Results
            ├─→ Metrics cards (risk levels)
            ├─→ Gauge chart (churn probability)
            └─→ Actionable recommendations
```

**UI Result:**
- Churn probability percentage
- Risk classification (HIGH/MEDIUM/LOW)
- Gauge chart visualization
- Specific business recommendations

---

## 📦 Module Breakdown

### **1. src/data_loader.py** - Data Ingestion
```python
ChurnDataLoader
├── __init__(data_path)
├── download_dataset()        # Download from web if not exists
├── load_dataset()            # Load CSV & print stats
└── get_data_summary()        # Return metadata dict
```
**Input:** None (downloads automatically)  
**Output:** Pandas DataFrame with 7,000+ customer records

---

### **2. src/preprocessor.py** - Data Transformation
```python
ChurnPreprocessor
├── clean_data(df)                              # Handle nulls, type conversions
├── feature_engineering(df)                     # Create new features
├── encode_features(df, is_training=True)       # Categorical encoding
├── scale_features(X_train, X_test)             # StandardScaler normalization
├── prepare_features(df)                        # Clean → FE → Encode → Scale
├── prepare_features_for_prediction(df)         # For single customer
├── train_test_split(df)                        # 80/20 split
└── full_pipeline(df)                           # Complete end-to-end
```
**Input:** Raw DataFrame  
**Output:** X_train, X_test, y_train, y_test (preprocessed & scaled)

---

### **3. src/model.py** - Model Training & Evaluation
```python
ChurnModel
├── train_all_models()          # Train LR, RF, XGB
├── evaluate_best_model()       # Metrics, confusion matrix
├── plot_roc_curve()            # ROC-AUC curve
├── plot_feature_importance()   # Top 15 features
├── explain_predictions_shap()  # SHAP values visualization
├── save_model(filepath)        # Save best model
└── load_model(filepath)        # Load saved model
```
**Input:** X_train, y_train, X_test, y_test  
**Output:** Trained model + visualizations (PNG files)

---

### **4. src/predict.py** - Inference & Predictions
```python
ChurnPredictor
├── __init__()                           # Load model & preprocessor
├── predict_single(customer_data_dict)   # Single customer prediction
└── predict_batch(df)                    # Multiple customers
```
**Input:** Customer feature dictionary  
**Output:** `{churn_probability, retention_probability, will_churn}`

---

### **5. app.py** - Streamlit Web Dashboard
```
UI Components:
├── Header & branding
├── Input form (7 customer fields)
├── Info panel (model details)
├── Predict button
├── Results display
│   ├── Metric cards
│   ├── Gauge chart
│   └── Recommendations
└── Business actions (based on risk level)
```

---

## 🔄 Data Pipeline Transformations

```
Raw Data (Telco-Churn.csv)
    ↓ load_dataset()
Loaded DF (7043 rows)
    ↓ clean_data()
Cleaned (nulls removed, types fixed)
    ↓ feature_engineering()
Features Added (dummies, interactions)
    ↓ encode_features()
Encoded (categorical → numeric)
    ↓ scale_features()
Scaled (mean=0, std=1)
    ↓ train_test_split()
X_train (5634 rows) | X_test (1409 rows)
y_train (5634 rows) | y_test (1409 rows)
    ↓ train_all_models()
3 Models Trained (LR, RF, XGB)
    ↓ best model = max(AUC)
Best Model Selected (usually XGB)
    ↓ save & evaluate
Artifacts Ready for Prediction
```

---

## 📊 File Dependencies

```
app.py
  ├─ imports: ChurnPredictor (src/predict.py)
  │   └─ imports: joblib, pandas
  │
  └─ ChurnPredictor loads:
      ├─ models/churn_model.pkl
      └─ models/preprocessor.pkl


src/train.py
  ├─ imports: ChurnDataLoader (src/data_loader.py)
  ├─ imports: ChurnPreprocessor (src/preprocessor.py)
  ├─ imports: ChurnModel (src/model.py)
  │
  └─ Saves:
      ├─ models/churn_model.pkl
      ├─ models/preprocessor.pkl
      ├─ models/feature_names.pkl
      └─ reports/*.png


src/predict.py
  ├─ imports: ChurnPreprocessor
  └─ loads: trained model & preprocessor


src/preprocessor.py
  └─ standalone module (no internal dependencies)


src/model.py
  └─ imports: scikit-learn, xgboost, shap, matplotlib
```

---

## ⚙️ Key Configuration

| Component | Configuration | Location |
|-----------|---------------|----------|
| **Data Source** | Telco Customer Churn (IBM) | `data/telco-churn.csv` |
| **Models Trained** | LR, RF, XGB | Selected by AUC score |
| **Train/Test Split** | 80/20 | `preprocessor.py` |
| **Scaler** | StandardScaler | `preprocessor.py` |
| **Features Used** | ~30 after encoding | Auto-generated |
| **Model Selection** | Best AUC on test set | `model.py` |

---

## 🎯 Prediction Flow (Visual)

```
User Input (form)
    ↓
customer_data_dict = {
    'tenure': 12,
    'MonthlyCharges': 65.0,
    'TotalCharges': 780.0,
    'Contract': 'Month-to-month',
    'InternetService': 'DSL',
    'OnlineSecurity': 'Yes',
    'TechSupport': 'Yes'
}
    ↓ (ChurnPredictor.predict_single)
    ├─ Add default values for missing features
    ├─ Convert to DataFrame
    ├─ Apply cleaning
    ├─ Apply feature engineering
    ├─ Apply encoding
    ├─ Apply scaling
    └─ model.predict_proba()
    ↓
Result Dict:
{
    'will_churn': False,
    'churn_probability': 0.23,
    'retention_probability': 0.77
}
    ↓
Display:
├─ Churn Probability: 23%
├─ Retention: 77%
├─ Risk Level: LOW RISK 🟢
├─ Gauge chart
└─ Recommendations
```

---

## 🔄 Workflow Summary

### **First Time Setup**
1. ✅ Run `python src/train.py`
   - Downloads data automatically
   - Trains 3 models
   - Saves best model to `models/`
   - Generates reports to `reports/`
   - Takes ~2-5 minutes

### **Daily Usage**
2. ✅ Run `streamlit run app.py`
   - Loads pre-trained model
   - Shows web dashboard
   - Makes predictions instantly
   - No retraining needed

### **To Retrain** (if you update training data)
3. ✅ Run `python src/train.py` again
   - Overwrites previous model
   - Updates reports
   - Streamlit will use new model automatically

---

## 🧠 Model Selection Logic

All three models are trained:
- **Logistic Regression (LR)** - Simple, interpretable baseline
- **Random Forest (RF)** - Medium complexity, robust
- **XGBoost (XGB)** - Most complex, highest accuracy

**Winner:** Model with highest **AUC score** on test set

Typical Results:
- LR: AUC ≈ 0.82
- RF: AUC ≈ 0.85
- XGB: AUC ≈ 0.87 ✅ (selected)

---

## 🚨 Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| `FileNotFoundError: churn_model.pkl` | Train data not found | Run `python src/train.py` |
| `No module named 'src'` | Wrong directory | `cd customer-churn-prediction` first |
| `SHAP explanation failed` | SHAP visualization issue | OK to skip, non-critical |
| `Streamlit error: port 8501` | Port in use | Use `streamlit run app.py --server.port 8502` |

---

## 📈 Performance Metrics

**Model Evaluation (on test set of 1,409 customers):**
- Accuracy: ~79%
- Precision: ~66% (when predicting churn, correct 66% of time)
- Recall: ~52% (catches 52% of actual churners)
- AUC-ROC: ~0.87

**Why not 100%?** Real customer behavior is complex; some patterns aren't captured by available features.

---

## 🎓 How Files Talk to Each Other

```python
# train.py orchestrates everything:
loader = ChurnDataLoader()
df = loader.load_dataset()  # Returns DataFrame

preprocessor = ChurnPreprocessor()
X_train, X_test, y_train, y_test = preprocessor.full_pipeline(df)

model = ChurnModel()
model.train_all_models(X_train, y_train, X_test, y_test)
model.save_model('models/churn_model.pkl')

# Save preprocessor for later use
joblib.dump(preprocessor, 'models/preprocessor.pkl')
```

```python
# app.py uses the saved artifacts:
from src.predict import ChurnPredictor

predictor = ChurnPredictor()  # Loads saved model & preprocessor
result = predictor.predict_single(customer_data)
# Returns: {'will_churn': bool, 'churn_probability': float, ...}
```

---

## 🔗 Dependency Chain

```
Requirements: pandas, numpy, scikit-learn, xgboost, shap, matplotlib, streamlit, plotly, joblib

app.py
  → src/predict.py
    → models/churn_model.pkl (joblib loaded)
    → models/preprocessor.pkl (joblib loaded)
      → sklearn.preprocessing.StandardScaler
      → sklearn.preprocessing.LabelEncoder/OneHotEncoder
      → data transformations

src/train.py
  → src/data_loader.py → data/telco-churn.csv
  → src/preprocessor.py → sklearn transformers
  → src/model.py → sklearn, xgboost, shap
    → models/churn_model.pkl (saved)
    → models/preprocessor.pkl (saved)
    → reports/*.png (matplotlib)
```

---

## ✅ Checklist for Understanding

- [ ] I know `python src/train.py` trains models
- [ ] I know `streamlit run app.py` runs the web UI
- [ ] I understand data flows: load → clean → encode → scale → train
- [ ] I know the model artifacts are saved to `models/` folder
- [ ] I understand ChurnPredictor loads those artifacts
- [ ] I know how the 5 main modules connect together
- [ ] I can explain the prediction flow from form input to gauge chart output

---

**Last Updated:** 2026-08-01  
**Project:** Customer Churn Prediction ML Pipeline
