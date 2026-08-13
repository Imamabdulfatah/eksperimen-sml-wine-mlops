"""
Modelling for MLflow Project CI Pipeline
==========================================
Adapted from modelling_tuning.py for use within MLflow Project context.
Trains a Random Forest model with hyperparameter tuning and logs to MLflow.

Imam Abdul Fatah
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import json
import os
import warnings

warnings.filterwarnings('ignore')


def main():
    print("=" * 60)
    print("  MLflow Project - Wine Quality Model Training")
    print("  Imam Abdul Fatah")
    print("=" * 60)

    # ===== 1. Load Data =====
    print("\n[STEP 1] Loading preprocessed dataset...")
    train_data = pd.read_csv('winequality_preprocessing/train_data.csv')
    test_data = pd.read_csv('winequality_preprocessing/test_data.csv')

    X_train = train_data.drop(columns=['quality_label'])
    y_train = train_data['quality_label']
    X_test = test_data.drop(columns=['quality_label'])
    y_test = test_data['quality_label']

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # ===== 2. MLflow Setup =====
    mlflow.set_experiment("wine-quality-ci")

    # ===== 3. Hyperparameter Tuning =====
    print("\n[STEP 2] Hyperparameter Tuning...")

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    }

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"  Best Params: {grid_search.best_params_}")

    # ===== 4. Evaluate & Log =====
    print("\n[STEP 3] Evaluasi dan Logging...")

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    with mlflow.start_run(run_name="CI_RandomForest_Tuned"):
        # Log params
        mlflow.log_params(grid_search.best_params_)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # Log model
        mlflow.sklearn.log_model(best_model, "model")

        # Artifacts
        artifact_dir = "artifacts"
        os.makedirs(artifact_dir, exist_ok=True)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        cm_path = os.path.join(artifact_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(cm_path)

        # Classification Report
        report = classification_report(y_test, y_pred,
                                       target_names=['Low', 'High'],
                                       output_dict=True)
        report_path = os.path.join(artifact_dir, "classification_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(report_path)

        # Tags
        mlflow.set_tag("student", "Imam Abdul Fatah")
        mlflow.set_tag("pipeline", "CI")

        run_id = mlflow.active_run().info.run_id
        print(f"\n  Run ID: {run_id}")
        print(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

    print("\n" + "=" * 60)
    print("  MODEL TRAINING SELESAI!")
    print("=" * 60)


if __name__ == "__main__":
    main()
