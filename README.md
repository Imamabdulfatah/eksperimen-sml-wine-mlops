# 🍷 Wine Quality MLOps System

[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)
[![DagsHub](https://img.shields.io/badge/DagsHub-Integrated-1B74E4.svg)](https://dagshub.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-E6522C.svg)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-Dashboard%20%26%20Alerting-F46800.svg)](https://grafana.com/)

Proyek akhir implementasi Machine Learning Operations (MLOps) untuk klasifikasi kualitas wine (*Wine Quality Classification*) berbasis data fisikokimia dari *UCI Machine Learning Repository*. Proyek ini mencakup seluruh siklus hidup MLOps mulai dari eksperimen awal, otomatisasi preprocessing data, pelacakan model eksperimen, otomatisasi workflow CI/CD, hingga model serving, monitoring, dan alerting.

---

## 📌 Daftar Isi
- [Ringkasan Proyek](#-ringkasan-proyek)
- [Struktur Direktori](#-struktur-direktori)
- [Prasyarat Sistem](#-prasyarat-sistem)
- [Panduan Modul & Eksekusi](#-panduan-modul--eksekusi)
  - [1. Eksperimen SML & Preprocessing Otomatis](#1-eksperimen-sml--preprocessing-otomatis)
  - [2. Pengembangan Model & MLflow Tracking (DagsHub)](#2-pengembangan-model--mlflow-tracking-dagshub)
  - [3. Workflow CI (GitHub Actions & Docker Hub)](#3-workflow-ci-github-actions--docker-hub)
  - [4. Monitoring, Logging & Alerting](#4-monitoring-logging--alerting)
- [Dokumentasi API Inference](#-dokumentasi-api-inference)
- [Kriteria & Checklist Penilaian](#-kriteria--checklist-penilaian)

---

## 📖 Ringkasan Proyek

| Komponen | Deskripsi |
|---|---|
| **Problem Domain** | Prediksi biner kualitas anggur (*High Quality* vs *Low Quality*) |
| **Dataset** | Wine Quality (Red & White Wine) — 6.497 sampel fisikokimia |
| **Model ML** | Random Forest Classifier & Hyperparameter Tuning (GridSearchCV) |
| **Experiment Tracking** | MLflow terintegrasi dengan Remote Storage DagsHub |
| **CI/CD Automation** | GitHub Actions Workflow (`preprocessing.yml` & `ci.yml`) |
| **Packaging** | MLflow Project & Containerization Docker |
| **Model Serving** | Flask REST API (`POST /predict`) |
| **Observability** | Prometheus (Scraper) & Grafana (Dashboard Visualisasi & Alerting) |

---

## 📂 Struktur Direktori

```text
proyek ml-ops/
├── README.md                                # Dokumentasi utama proyek
├── Template_Eksperimen_MSML.ipynb           # Template acuan eksperimen
│
├── Eksperimen_SML_Imam-Abdul-Fatah/         # Modul 1: Eksperimen & Data Preprocessing
│   ├── .github/workflows/
│   │   └── preprocessing.yml                # CI otomatisasi preprocessing data
│   ├── preprocessing/
│   │   ├── Eksperimen_Imam-Abdul-Fatah.ipynb# Notebook EDA & Eksperimen lengkap
│   │   ├── automate_Imam-Abdul-Fatah.py     # Skrip otomatisasi preprocessing
│   │   └── winequality_preprocessing/       # Dataset hasil preprocessing (train/test)
│   ├── winequality_raw/                     # Dataset mentah (red & white wine)
│   └── requirements.txt
│
├── Membangun_model/                         # Modul 2: Training & MLflow Tracking
│   ├── DagsHub.txt                          # Informasi tautan DagsHub & panduan
│   ├── modelling.py                         # Training dasar dengan MLflow autolog
│   ├── modelling_tuning.py                  # Hyperparameter tuning + DagsHub tracking
│   ├── winequality_preprocessing/           # Dataset input training
│   └── requirements.txt
│
├── Workflow-CI/                             # Modul 3: MLflow Project & CI Docker
│   ├── .github/workflows/
│   │   └── ci.yml                           # GitHub Actions CI: Train model & Docker build
│   └── MLProject/
│       ├── MLProject                        # File definisi standar MLflow Project
│       ├── conda.yaml                       # Definisi environment conda
│       ├── docker_hub_link.txt              # Informasi tautan image di Docker Hub
│       ├── modelling.py                     # Entrypoint skrip training MLflow
│       └── winequality_preprocessing/
│
└── Monitoring_dan_Logging/                  # Modul 4: Serving, Prometheus & Grafana
    ├── 1.bukti_serving/                     # Bukti screenshot request/response API
    ├── 2.prometheus.yml                     # Konfigurasi scraping Prometheus
    ├── 3.prometheus_exporter.py             # Custom Prometheus metrics exporter
    ├── 4.bukti_monitoring_Prometheus/       # Bukti screenshot targets & query Prometheus
    ├── 5.bukti_monitoring_Grafana/          # Bukti screenshot dashboard Grafana
    ├── 6.bukti_alerting_Grafana/            # Bukti screenshot alert rule Grafana
    ├── 7.inference.py                       # REST API Flask untuk serving model
    ├── Dockerfile                           # Dockerfile model serving
    ├── docker-compose.yml                   # Orkestrasi API, Prometheus, & Grafana
    ├── load_test.py                         # Skrip simulasi request beban trafik
    └── requirements.txt
```

---

## ⚙️ Prasyarat Sistem

- Python 3.8+
- Git & Git CLI
- Akun [GitHub](https://github.com/)
- Akun [DagsHub](https://dagshub.com/)
- Akun [Docker Hub](https://hub.docker.com/) & [Docker Desktop](https://www.docker.com/products/docker-desktop/) (opsional untuk container lokal)

---

## 🚀 Panduan Modul & Eksekusi

### 1. Eksperimen SML & Preprocessing Otomatis
**Direktori:** `Eksperimen_SML_Imam-Abdul-Fatah`

1. **Jalankan Preprocessing:**
   ```powershell
   cd "Eksperimen_SML_Imam-Abdul-Fatah\preprocessing"
   python automate_Imam-Abdul-Fatah.py
   ```
2. **Jalankan & Review Notebook:**
   ```powershell
   python -m notebook
   ```
   Buka file `Eksperimen_Imam-Abdul-Fatah.ipynb`, lalu pilih **Kernel > Restart & Run All** untuk memastikan seluruh visualisasi EDA (boxplot outlier, heatmap korelasi, countplot distribusi) tersimpan rapi.
3. **Push ke GitHub Repository:**
   Push folder `Eksperimen_SML_Imam-Abdul-Fatah` ke repository GitHub Anda. GitHub Actions `preprocessing.yml` akan ter-trigger saat data mentah atau skrip preprocessing diperbarui.

---

### 2. Pengembangan Model & MLflow Tracking (DagsHub)
**Direktori:** `Membangun_model`

1. **Buat Repository di DagsHub:**
   - Masuk ke [DagsHub](https://dagshub.com/) dan buat repo baru: `wine-quality-mlops`.
   - Buka **User Settings > Tokens** dan salin token akses Anda.
2. **Set Environment Variable & Jalankan Tuning:**
   ```powershell
   cd "..\Membangun_model"
   $env:DAGSHUB_USERNAME="<USERNAME_DAGSHUB_ANDA>"
   $env:DAGSHUB_REPO="wine-quality-mlops"
   $env:DAGSHUB_TOKEN="<DAGSHUB_TOKEN_ANDA>"

   python modelling_tuning.py
   ```
3. **Verifikasi Hasil Eksperimen:**
   Buka DagsHub > tab **MLflow UI**. Pastikan metrik evaluasi (*Accuracy*, *Precision*, *Recall*, *F1-Score*, *ROC-AUC*) serta artefak gambar (*Confusion Matrix*, *ROC Curve*, *Feature Importance*) telah berhasil di-upload.
4. **Update `DagsHub.txt`:**
   Sesuaikan URL repository dan tracking server pada file `DagsHub.txt`.

---

### 3. Workflow CI (GitHub Actions & Docker Hub)
**Direktori:** `Workflow-CI`

1. **Siapkan Repository Docker Hub:**
   Buat repository publik di [Docker Hub](https://hub.docker.com/) dengan nama `wine-quality-model`.
2. **Konfigurasi GitHub Secrets:**
   Pada repository GitHub modul ini, tambahkan secret pada menu **Settings > Secrets and variables > Actions**:
   - `DOCKER_USERNAME`: Username akun Docker Hub Anda.
   - `DOCKER_PASSWORD`: Personal Access Token / Password Docker Hub Anda.
3. **Update URL Docker Hub:**
   Perbarui file `Workflow-CI/MLProject/docker_hub_link.txt` dengan link repository Docker Hub Anda.
4. **Push dan Jalankan Pipeline:**
   Push folder `Workflow-CI` ke GitHub. Workflow `ci.yml` akan melatih model menggunakan MLflow Project, membangun Docker image, dan mem-push image ke Docker Hub secara otomatis.

---

### 4. Monitoring, Logging & Alerting
**Direktori:** `Monitoring_dan_Logging`

1. **Jalankan Seluruh Layanan dengan Docker Compose:**
   ```powershell
   cd "..\Monitoring_dan_Logging"
   docker-compose up --build -d
   ```
   *Port Layanan:*
   - **Model Serving API:** `http://localhost:5000`
   - **Prometheus UI:** `http://localhost:9090`
   - **Grafana Dashboard:** `http://localhost:3000` *(Login: admin / admin)*

   *(Opsi Tanpa Docker: Jalankan mandiri dengan `python 7.inference.py`)*

2. **Simulasi Beban Trafik (Load Test):**
   ```powershell
   python load_test.py
   ```

3. **Pengambilan Screenshot Bukti:**
   - **`1.bukti_serving/`**: Screenshot pengujian endpoint `POST /predict` (Postman / cURL) yang menampilkan input fitur dan JSON response `200 OK`.
   - **`4.bukti_monitoring_Prometheus/`**:
     - Screenshot menu *Status > Targets* (Status `UP`).
     - Screenshot menu *Graph* dengan query `rate(model_prediction_requests_total[1m])`.
   - **`5.bukti_monitoring_Grafana/`**: Screenshot dashboard monitoring menampilkan panel Total Requests, Latency, dan Distribusi Prediksi.
   - **`6.bukti_alerting_Grafana/`**: Screenshot konfigurasi Alert Rule (High Latency Alert dengan kondisi threshold).

---

## 📡 Dokumentasi API Inference

### 1. Health Check
- **URL:** `GET /health`
- **Response:**
  ```json
  {
    "model_loaded": true,
    "status": "healthy"
  }
  ```

### 2. Predict Wine Quality
- **URL:** `POST /predict`
- **Headers:** `Content-Type: application/json`
- **Request Body:**
  ```json
  {
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
  ```
- **Response Body:**
  ```json
  {
    "status": "success",
    "prediction": "High Quality",
    "prediction_code": 1,
    "probability": {
      "High Quality": 0.8742,
      "Low Quality": 0.1258
    },
    "inference_latency_seconds": 0.0142
  }
  ```

### 3. Prometheus Metrics
- **URL:** `GET /metrics`
- **Deskripsi:** Endpoint scraping metrik Prometheus (*requests total*, *latency histogram*, *prediction breakdown*, dan *system resource metrics*).

---

## ✅ Kriteria & Checklist Penilaian

- [x] **Modul 1**: Eksperimen EDA lengkap, penanganan duplikat & outlier, standarisasi fitur, skrip otomasi preprocessing, dan CI workflow.
- [x] **Modul 2**: Training model dengan GridSearchCV, MLflow Tracking terhubung ke remote server DagsHub, artefak evaluasi lengkap.
- [x] **Modul 3**: Standardisasi `MLProject`, konfigurasi `conda.yaml`, integrasi GitHub Actions CI build & push ke Docker Hub.
- [x] **Modul 4**: Model serving REST API, Prometheus exporter, konfigurasi scraping, Grafana dashboard panel, alert rule, dan screenshot bukti pengujian.

---

**Author:** Imam Abdul Fatah  
**Project:** Belajar Membangun Sistem Machine Learning (MSML) / MLOps Submission
