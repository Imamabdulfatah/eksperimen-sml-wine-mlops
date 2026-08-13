"""
Inference API & Model Serving - Imam Abdul Fatah
===============================================
REST API menggunakan Flask untuk model serving Wine Quality.
Terintegrasi dengan Prometheus Metrics Exporter.

Endpoints:
  - GET  /         : Info API & Status
  - GET  /health   : Health check
  - POST /predict  : Endpoint Prediksi Wine Quality
  - GET  /metrics  : Endpoint Scrape Prometheus
"""

from flask import Flask, request, jsonify, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import time
import os
import json
import importlib.util

# Import prometheus_exporter module
import importlib.util
_exporter_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "3.prometheus_exporter.py")
_spec = importlib.util.spec_from_file_location("prometheus_exporter", _exporter_path)
exporter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exporter)

app = Flask(__name__)

# Global model & scaler
MODEL = None
SCALER = None
FEATURE_NAMES = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol', 'wine_type'
]

def load_or_train_model():
    """Load model trained sebelumnya atau train model jika belum ada."""
    global MODEL, SCALER
    start_time = time.time()
    
    model_path = os.path.join(os.path.dirname(__file__), 'model.joblib')
    scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.joblib')
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        MODEL = joblib.load(model_path)
        SCALER = joblib.load(scaler_path)
        print("[INFO] Model & Scaler loaded from joblib disk cache.")
    else:
        print("[INFO] Model belum ada. Melatih model baru pada dataset preprocessed...")
        train_csv = os.path.join(
            os.path.dirname(__file__),
            '..', 'Membangun_model', 'winequality_preprocessing', 'train_data.csv'
        )
        if not os.path.exists(train_csv):
            train_csv = os.path.join(
                os.path.dirname(__file__),
                '..', 'Eksperimen_SML_Imam-Abdul-Fatah', 'preprocessing', 'winequality_preprocessing', 'train_data.csv'
            )
            
        train_df = pd.read_csv(train_csv)
        X_train = train_df.drop(columns=['quality_label'])
        y_train = train_df['quality_label']
        
        SCALER = StandardScaler()
        X_scaled = SCALER.fit_transform(X_train)
        
        MODEL = RandomForestClassifier(n_estimators=100, random_state=42)
        MODEL.fit(X_scaled, y_train)
        
        joblib.dump(MODEL, model_path)
        joblib.dump(SCALER, scaler_path)
        print("[INFO] Model baru berhasil dilatih dan disimpan.")

    elapsed = time.time() - start_time
    exporter.MODEL_LOAD_TIME_SECONDS.set(elapsed)

# Load model saat startup
load_or_train_model()


@app.route('/', methods=['GET'])
def index():
    exporter.update_system_metrics()
    return jsonify({
        "status": "online",
        "service": "Wine Quality Inference API",
        "student": "Imam Abdul Fatah",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "metrics": "/metrics"
        }
    })


@app.route('/health', methods=['GET'])
def health():
    exporter.update_system_metrics()
    return jsonify({"status": "healthy", "model_loaded": MODEL is not None})


@app.route('/metrics', methods=['GET'])
def metrics():
    """Expose metrik untuk di-scrape oleh Prometheus."""
    exporter.update_system_metrics()
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route('/predict', methods=['POST'])
def predict():
    exporter.ACTIVE_REQUESTS.inc()
    start_time = time.time()
    
    try:
        raw_text = request.get_data(as_text=True).strip()
        print(f"[DEBUG] Received raw body: {raw_text[:200] if raw_text else '<EMPTY>'}")
        
        data = request.get_json(force=True, silent=True)
        if data is None and raw_text:
            try:
                data = json.loads(raw_text)
            except Exception:
                data = None
                
        if data is None or (isinstance(data, (dict, list)) and len(data) == 0):
            exporter.PREDICTION_ERRORS_TOTAL.labels(error_type='empty_payload').inc()
            exporter.PREDICTION_REQUESTS_TOTAL.labels(endpoint='/predict', status='400').inc()
            return jsonify({
                "status": "error",
                "message": "Body JSON tidak boleh kosong atau format tidak valid. Pastikan memilih Body -> raw -> JSON di Postman dan isi data.",
                "example_payload": {
                    "fixed acidity": 7.4,
                    "volatile acidity": 0.70,
                    "citric acid": 0.00,
                    "residual sugar": 1.9,
                    "chlorides": 0.076,
                    "free sulfur dioxide": 11.0,
                    "total sulfur dioxide": 34.0,
                    "density": 0.9978,
                    "pH": 3.51,
                    "sulphates": 0.56,
                    "alcohol": 9.4,
                    "wine_type": "red"
                }
            }), 400
            
        payload_bytes = len(json.dumps(data).encode('utf-8'))
        exporter.REQUEST_PAYLOAD_SIZE_BYTES.observe(payload_bytes)
        
        # 1. Format: Dictionary
        if isinstance(data, dict):
            if 'features' in data:
                features = data['features']
                if isinstance(features, list):
                    if len(features) > 0 and isinstance(features[0], list):
                        df_input = pd.DataFrame(features, columns=FEATURE_NAMES[:len(features[0])])
                    else:
                        df_input = pd.DataFrame([features], columns=FEATURE_NAMES[:len(features)])
                else:
                    return jsonify({"error": "Format 'features' harus berupa array"}), 400
            else:
                # Direct dictionary with feature names
                row = data.copy()
                if 'wine_type' in row:
                    if str(row['wine_type']).lower() in ['red', '0']:
                        row['wine_type'] = 0
                    else:
                        row['wine_type'] = 1
                df_input = pd.DataFrame([row])
                for col in FEATURE_NAMES:
                    if col not in df_input.columns:
                        df_input[col] = 0.0
                df_input = df_input[FEATURE_NAMES]
                
        # 2. Format: Direct array/list
        elif isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], list):
                df_input = pd.DataFrame(data, columns=FEATURE_NAMES[:len(data[0])])
            else:
                df_input = pd.DataFrame([data], columns=FEATURE_NAMES[:len(data)])
        else:
            return jsonify({"error": "Format JSON tidak dikenali"}), 400

        # Predict
        X_scaled = SCALER.transform(df_input)
        preds = MODEL.predict(X_scaled)
        probs = MODEL.predict_proba(X_scaled)[:, 1]
        
        results = []
        for pred, prob in zip(preds, probs):
            pred_int = int(pred)
            prob_float = float(prob)
            
            # Record metrik per-prediksi
            exporter.MODEL_PREDICTION_CLASS_TOTAL.labels(predicted_class=str(pred_int)).inc()
            exporter.PREDICTION_CONFIDENCE_SCORE.observe(prob_float)
            
            if prob_float > 0.85 or (1 - prob_float) > 0.85:
                exporter.HIGH_CONFIDENCE_PREDICTIONS_TOTAL.inc()
            
            results.append({
                "prediction": pred_int,
                "label": "High Quality" if pred_int == 1 else "Low Quality",
                "confidence": round(prob_float, 4)
            })
        
        latency = time.time() - start_time
        exporter.PREDICTION_LATENCY_SECONDS.labels(endpoint='/predict').observe(latency)
        exporter.PREDICTION_REQUESTS_TOTAL.labels(endpoint='/predict', status='200').inc()
        
        return jsonify({
            "status": "success",
            "count": len(results),
            "results": results,
            "latency_ms": round(latency * 1000, 2)
        })

    except Exception as e:
        exporter.PREDICTION_ERRORS_TOTAL.labels(error_type=type(e).__name__).inc()
        exporter.PREDICTION_REQUESTS_TOTAL.labels(endpoint='/predict', status='500').inc()
        return jsonify({"status": "error", "message": str(e)}), 500
        
    finally:
        exporter.ACTIVE_REQUESTS.dec()


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"[INFO] Server Inference berjalan di http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
