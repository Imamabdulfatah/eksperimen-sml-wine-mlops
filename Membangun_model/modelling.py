"""
Modelling - Wine Quality Classification (Basic)
================================================
MLflow Tracking dengan autolog (lokal).
Melatih model Random Forest tanpa hyperparameter tuning.

Imam Abdul Fatah

Usage:
    python modelling.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import os


def main():
    # ===== 1. Load Preprocessed Dataset =====
    print("Loading preprocessed dataset...")
    train_path = 'winequality_preprocessing/train_data.csv'
    test_path = 'winequality_preprocessing/test_data.csv'
    if not os.path.exists(train_path):
        train_path = 'winequality_preprocessing/winequality_preprocessing/train_data.csv'
        test_path = 'winequality_preprocessing/winequality_preprocessing/test_data.csv'

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    X_train = train_data.drop(columns=['quality_label'])
    y_train = train_data['quality_label']
    X_test = test_data.drop(columns=['quality_label'])
    y_test = test_data['quality_label']

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # ===== 2. MLflow Setup (Lokal) =====
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("wine-quality-basic")

    # ===== 3. Train Model dengan Autolog =====
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="RandomForest_Autolog"):
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Low', 'High']))

        print("\n[INFO] Model dan metrik tersimpan di MLflow (lokal)")
        print("[INFO] Jalankan 'mlflow ui' untuk melihat dashboard")


if __name__ == "__main__":
    main()
