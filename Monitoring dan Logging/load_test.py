"""
Load Generator Script - Imam Abdul Fatah
========================================
Simulasi traffic prediksi ke Inference API (http://localhost:5000/predict).
Menghasilkan data untuk memicu seluruh 12 metrik Prometheus dan 3 Alert Rules di Grafana.

Usage:
    python load_test.py
"""

import requests
import random
import time
import json

URL = "http://localhost:5000/predict"

# Sampel data fitur Wine Quality (12 fitur)
# [fixed acidity, volatile acidity, citric acid, residual sugar, chlorides,
#  free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol, wine_type]

SAMPLE_GOOD_FEATURES = [
    [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 0],
    [7.8, 0.88, 0.00, 2.6, 0.098, 25.0, 67.0, 0.9968, 3.20, 0.68, 9.8, 0],
    [6.3, 0.30, 0.34, 1.6, 0.049, 14.0, 132.0, 0.9940, 3.30, 0.49, 9.5, 1],
    [8.1, 0.28, 0.40, 6.9, 0.050, 30.0, 97.0, 0.9951, 3.26, 0.44, 10.1, 1],
    [7.0, 0.27, 0.36, 2.07, 0.045, 45.0, 170.0, 0.9910, 3.30, 0.45, 11.8, 1]
]

def generate_traffic(num_requests=200):
    print("=" * 60)
    print("  SIMULASI TRAFFIC & MONITORING LOAD TEST")
    print("  Target: http://localhost:5000/predict")
    print("  Imam Abdul Fatah")
    print("=" * 60)
    
    success_count = 0
    error_count = 0
    
    for i in range(1, num_requests + 1):
        # acak tipe request: 90% normal, 10% invalid payload untuk simulate errors
        rand_val = random.random()
        
        if rand_val < 0.85:
            # Request normal
            sample = random.choice(SAMPLE_GOOD_FEATURES)
            payload = {"features": sample}
        elif rand_val < 0.95:
            # Batch request (multiple rows)
            batch = random.choices(SAMPLE_GOOD_FEATURES, k=random.randint(2, 5))
            payload = {"features": batch}
        else:
            # Invalid request (trigger error metrics & alert)
            payload = {"invalid_key": [1, 2, 3]}
            
        try:
            start = time.time()
            resp = requests.post(URL, json=payload, timeout=5)
            latency = (time.time() - start) * 1000
            
            if resp.status_code == 200:
                success_count += 1
                res = resp.json()
                pred_label = res['results'][0]['label']
                conf = res['results'][0]['confidence']
                print(f"[{i}/{num_requests}] Status: 200 OK | Prediksi: {pred_label} (conf: {conf}) | Latensi: {latency:.1f}ms")
            else:
                error_count += 1
                print(f"[{i}/{num_requests}] Status: {resp.status_code} Error | Msg: {resp.text[:50]}")
                
        except Exception as e:
            error_count += 1
            print(f"[{i}/{num_requests}] Connection Error: {e}")
            
        time.sleep(random.uniform(0.05, 0.3))
        
    print("\n" + "=" * 60)
    print(f"  SELESAI! Total Request: {num_requests} | Berhasil: {success_count} | Error: {error_count}")
    print("=" * 60)

if __name__ == '__main__':
    generate_traffic(100)
