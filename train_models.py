import os
import json
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

def train_and_evaluate():
    print("Loading synthetic clinical data...")
    csv_path = os.path.join("data", "synthetic_clinical_data.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please run synthetic_data_generator.py first.")
        
    df = pd.read_csv(csv_path)
    
    # Feature engineering / encoding
    # Encode Gender: Male -> 1, Female -> 0
    df['Gender_Encoded'] = df['Gender'].map({'Male': 1, 'Female': 0})
    
    # Features (X) and Target (y)
    feature_cols = [
        "Age", "Gender_Encoded", "Hemoglobin", "WBC", "Platelets", 
        "Fasting_Blood_Sugar", "Creatinine", "TSH", "Cholesterol"
    ]
    X = df[feature_cols]
    y = df["Diagnosis"]
    
    # Encode Diagnosis labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Save the label encoder classes so we can reconstruct labels later
    os.makedirs("models", exist_ok=True)
    joblib.dump(le, os.path.join("models", "label_encoder.joblib"))
    print("Label encoder saved.")
    
    # Split into train and test sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric="mlogloss")
    }
    
    comparison_metrics = {}
    best_accuracy = 0.0
    best_model_name = ""
    best_model_obj = None
    
    for name, model in models.items():
        print(f"Training {name}...")
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Predict & measure latency
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = (time.time() - start_time) / len(X_test) * 1000 # in milliseconds per sample
        
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        
        report_dict = classification_report(
            y_test, y_pred, target_names=le.classes_, output_dict=True
        )
        
        comparison_metrics[name] = {
            "accuracy": round(float(accuracy), 4),
            "f1_macro": round(float(f1_macro), 4),
            "training_time_sec": round(training_time, 4),
            "inference_latency_ms": round(inference_time, 6),
            "classification_report": report_dict
        }
        
        print(f"{name} Results - Accuracy: {accuracy:.4f}, F1-Macro: {f1_macro:.4f}, Latency: {inference_time:.4f} ms")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            best_model_obj = model
            
    print(f"\nBest Model: {best_model_name} with Accuracy {best_accuracy:.4f}")
    
    # Save the best model
    model_save_path = os.path.join("models", "clinical_classifier.joblib")
    joblib.dump(best_model_obj, model_save_path)
    print(f"Saved best model to {model_save_path}")
    
    # Save model features list for consistency
    joblib.dump(feature_cols, os.path.join("models", "feature_columns.joblib"))
    
    # Save metrics comparison JSON for the UI dashboard
    comparison_save_path = os.path.join("data", "model_comparison.json")
    with open(comparison_save_path, "w", encoding="utf-8") as f:
        json.dump(comparison_metrics, f, indent=4)
    print(f"Saved evaluation metrics to {comparison_save_path}")

if __name__ == "__main__":
    train_and_evaluate()
