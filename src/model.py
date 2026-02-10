import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, f1_score, precision_score, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

class ChurnModel:
    """Train, evaluate, and explain churn prediction models."""
    
    def __init__(self):
        self.models = {
            'lr': LogisticRegression(random_state=42, max_iter=1000),
            'rf': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'xgb': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', n_jobs=-1)
        }
        self.trained_models = {}
        self.best_model = None
        self.best_model_name = None
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """Train all models and compare."""
        print("\n🚀 Training models...\n")
        
        results = {}
        
        for name, model in self.models.items():
            print(f"  Training {name.upper()}...", end=' ')
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            auc = roc_auc_score(y_test, y_pred_proba)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            results[name] = {
                'auc': auc,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
            
            print(f"✅ AUC={auc:.3f}, F1={f1:.3f}")
        
        # Find best model (by AUC)
        self.best_model_name = max(results, key=lambda x: results[x]['auc'])
        self.best_model = self.trained_models[self.best_model_name]
        
        print(f"\n🏆 Best Model: {self.best_model_name.upper()} (AUC={results[self.best_model_name]['auc']:.3f})")
        
        return results
    
    def evaluate_best_model(self, X_test, y_test):
        """Detailed evaluation of best model."""
        print(f"\n📊 Detailed Evaluation - {self.best_model_name.upper()}\n")
        
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
        
        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        plt.title(f'Confusion Matrix - {self.best_model_name.upper()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        os.makedirs('reports', exist_ok=True)
        plt.savefig('reports/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return y_pred, y_pred_proba
    
    def plot_roc_curve(self, X_test, y_test):
        """Plot ROC curve."""
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC={auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.best_model_name.upper()}')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        os.makedirs('reports', exist_ok=True)
        plt.savefig('reports/roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, feature_names):
        """Plot feature importance."""
        if self.best_model_name == 'lr':
            # For logistic regression, use coefficients
            importance = np.abs(self.best_model.coef_.flatten())
            method = "Coefficients"
        else:
            # For RF and XGB, use feature_importances_
            importance = self.best_model.feature_importances_
            method = "Importance"
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(15)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
        plt.title(f'Top 15 Features - {self.best_model_name.upper()} ({method})')
        plt.xlabel(method)
        plt.tight_layout()
        os.makedirs('reports', exist_ok=True)
        plt.savefig('reports/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n🎯 Top 5 Important Features:")
        for idx, row in importance_df.head(5).iterrows():
            print(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}")
    
    def explain_predictions_shap(self, X_test, num_samples=100):
        """Generate SHAP explanations."""
        print("\n📊 Generating SHAP explanations...")
        
        # Use sample for faster computation
        X_sample = X_test.sample(n=min(num_samples, len(X_test)), random_state=42)
        
        explainer = shap.Explainer(self.best_model, X_sample)
        shap_values = explainer.shap_values(X_sample)
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        os.makedirs('reports', exist_ok=True)
        plt.savefig('reports/shap_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath='models/churn_model.pkl'):
        """Save trained model."""
        joblib.dump(self.best_model, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath='models/churn_model.pkl'):
        """Load trained model."""
        self.best_model = joblib.load(filepath)
        print(f"✅ Model loaded from {filepath}")
