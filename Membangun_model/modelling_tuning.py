"""
Modelling Tuning - Wine Quality Classification (Advanced)
==========================================================
MLflow Tracking dengan manual logging + DagsHub (online).
Melatih model dengan hyperparameter tuning menggunakan GridSearchCV.
Menyimpan artefak tambahan: Confusion Matrix, ROC Curve, Feature Importance,
Classification Report JSON.

Imam Abdul Fatah

Usage:
    python modelling_tuning.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
import dagshub
import json
import os
import warnings

warnings.filterwarnings('ignore')


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot dan simpan confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low', 'High'],
                yticklabels=['Low', 'High'])
    plt.title('Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] Confusion Matrix -> {save_path}")


def plot_roc_curve(y_true, y_prob, save_path):
    """Plot dan simpan ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#2196F3', lw=2, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] ROC Curve -> {save_path}")


def plot_feature_importance(model, feature_names, save_path):
    """Plot dan simpan feature importance."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importances)),
             importances[indices],
             color='#4CAF50', alpha=0.8)
    plt.yticks(range(len(importances)),
               [feature_names[i] for i in indices])
    plt.xlabel('Importance', fontsize=12)
    plt.title('Feature Importance', fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] Feature Importance -> {save_path}")


def main():
    # ===== 1. Load Preprocessed Dataset =====
    print("=" * 60)
    print("  MODELLING TUNING - Wine Quality (Advanced)")
    print("  Imam Abdul Fatah")
    print("=" * 60)

    print("\n[STEP 1] Loading preprocessed dataset...")
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

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # ===== 2. DagsHub + MLflow Setup =====
    print("\n[STEP 2] Setting up DagsHub + MLflow...")

    # --- KONFIGURASI DAGSHUB ---
    DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME", "imamabdul8875")
    DAGSHUB_REPO = os.getenv("DAGSHUB_REPO", "wine-quality-mlops")
    DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN", "YOUR_DAGSHUB_TOKEN")

    os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
    if DAGSHUB_TOKEN != "YOUR_DAGSHUB_TOKEN":
        os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN

    mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")
    mlflow.set_experiment("wine-quality-tuning")

    print(f"  MLflow Tracking URI: https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")

    # ===== 3. Hyperparameter Tuning =====
    print("\n[STEP 3] Hyperparameter Tuning (GridSearchCV)...")

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"\n  Best Parameters: {best_params}")
    print(f"  Best CV F1 Score: {grid_search.best_score_:.4f}")

    # ===== 4. Evaluasi dan Manual Logging =====
    print("\n[STEP 4] Evaluasi dan Manual Logging ke MLflow...")

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # Hitung metrik (sama dengan autolog)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"\n  Accuracy : {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC AUC  : {roc_auc:.4f}")

    # Buat direktori temp untuk artefak
    artifact_dir = "artifacts_temp"
    os.makedirs(artifact_dir, exist_ok=True)

    with mlflow.start_run(run_name="RandomForest_Tuned_Manual"):
        # --- Manual Logging: Parameters ---
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("test_size", 0.2)

        # --- Manual Logging: Metrics (sama dengan autolog) ---
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("best_cv_f1", grid_search.best_score_)
        mlflow.log_metric("training_score", best_model.score(X_train, y_train))

        # --- Manual Logging: Model ---
        mlflow.sklearn.log_model(best_model, "model")

        # --- Artefak Tambahan 1: Confusion Matrix ---
        print("\n[STEP 5] Membuat artefak tambahan...")
        cm_path = os.path.join(artifact_dir, "confusion_matrix.png")
        plot_confusion_matrix(y_test, y_pred, cm_path)
        mlflow.log_artifact(cm_path)

        # --- Artefak Tambahan 2: ROC Curve ---
        roc_path = os.path.join(artifact_dir, "roc_curve.png")
        plot_roc_curve(y_test, y_prob, roc_path)
        mlflow.log_artifact(roc_path)

        # --- Artefak Tambahan 3: Feature Importance ---
        fi_path = os.path.join(artifact_dir, "feature_importance.png")
        plot_feature_importance(best_model, list(X_train.columns), fi_path)
        mlflow.log_artifact(fi_path)

        # --- Artefak Tambahan 4: Classification Report JSON ---
        report = classification_report(y_test, y_pred, target_names=['Low', 'High'], output_dict=True)
        report_path = os.path.join(artifact_dir, "classification_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(report_path)
        print(f"  [SAVED] Classification Report -> {report_path}")

        # --- Log tags ---
        mlflow.set_tag("student", "Imam Abdul Fatah")
        mlflow.set_tag("dataset", "Wine Quality")
        mlflow.set_tag("task", "Binary Classification")

        run_id = mlflow.active_run().info.run_id
        print(f"\n  MLflow Run ID: {run_id}")

    print("\n" + "=" * 60)
    print("  MODELLING SELESAI!")
    print(f"  Dashboard: https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")
    print("=" * 60)


if __name__ == "__main__":
    main()
