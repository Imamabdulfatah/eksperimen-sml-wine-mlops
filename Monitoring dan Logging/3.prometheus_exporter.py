"""
Prometheus Exporter for Wine Quality Model - Imam Abdul Fatah
=============================================================
Definisi dan pengelolaan minimal 10 metrik kustom Prometheus
untuk monitoring performa model, sistem, dan traffic inference.
"""

from prometheus_client import Counter, Histogram, Gauge, Summary
import psutil
import time

# ==============================================================================
# DEFINISI 12 METRIK PROMETHEUS (Exceeding 10 metrics requirement)
# ==============================================================================

# 1. Counter: Total request prediksi
PREDICTION_REQUESTS_TOTAL = Counter(
    'prediction_requests_total',
    'Total jumlah request prediksi yang diterima',
    ['endpoint', 'status']
)

# 2. Histogram: Latensi prediksi dalam detik
PREDICTION_LATENCY_SECONDS = Histogram(
    'prediction_latency_seconds',
    'Waktu yang dibutuhkan untuk memproses prediksi',
    ['endpoint'],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5]
)

# 3. Counter: Total error prediksi
PREDICTION_ERRORS_TOTAL = Counter(
    'prediction_errors_total',
    'Total jumlah error saat proses inference',
    ['error_type']
)

# 4. Gauge: Jumlah request yang sedang aktif diproses
ACTIVE_REQUESTS = Gauge(
    'active_requests',
    'Jumlah request yang sedang diproses secara bersamaan'
)

# 5. Counter: Distribusi kelas hasil prediksi (0=Low, 1=High)
MODEL_PREDICTION_CLASS_TOTAL = Counter(
    'model_prediction_class_total',
    'Jumlah hasil prediksi per kelas',
    ['predicted_class']
)

# 6. Histogram: Skor kepercayaan / probabilitas prediksi
PREDICTION_CONFIDENCE_SCORE = Histogram(
    'prediction_confidence_score',
    'Distribusi probabilitas/confidence score dari model',
    buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
)

# 7. Gauge: Penggunaan CPU server (%)
CPU_USAGE_PERCENT = Gauge(
    'system_cpu_usage_percent',
    'Persentase penggunaan CPU server'
)

# 8. Gauge: Penggunaan Memori server (Bytes)
MEMORY_USAGE_BYTES = Gauge(
    'system_memory_usage_bytes',
    'Penggunaan RAM server dalam Bytes'
)

# 9. Gauge: Waktu yang dibutuhkan saat meload artefak model (detik)
MODEL_LOAD_TIME_SECONDS = Gauge(
    'model_load_time_seconds',
    'Waktu yang dibutuhkan untuk me-load artefak model ke memori'
)

# 10. Histogram: Ukuran payload/fitur input request (Bytes)
REQUEST_PAYLOAD_SIZE_BYTES = Histogram(
    'request_payload_size_bytes',
    'Ukuran payload request yang dikirim pengguna',
    buckets=[100, 250, 500, 1000, 2500, 5000]
)

# 11. Gauge: Uptime server inference (detik)
SYSTEM_UPTIME_SECONDS = Gauge(
    'system_uptime_seconds',
    'Waktu aktif server dalam detik'
)

# 12. Counter: Jumlah prediksi berkepercayaan tinggi (>0.85)
HIGH_CONFIDENCE_PREDICTIONS_TOTAL = Counter(
    'high_confidence_predictions_total',
    'Jumlah prediksi dengan skor probabilitas sangat tinggi (>0.85)'
)

# Waktu mulai server untuk menghitung uptime
SERVER_START_TIME = time.time()


def update_system_metrics():
    """Meng-update metrik sistem (CPU, Memory, Uptime) secara dinamis."""
    CPU_USAGE_PERCENT.set(psutil.cpu_percent(interval=None))
    MEMORY_USAGE_BYTES.set(psutil.virtual_memory().used)
    SYSTEM_UPTIME_SECONDS.set(time.time() - SERVER_START_TIME)
