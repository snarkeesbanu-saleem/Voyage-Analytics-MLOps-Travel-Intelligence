# Voyage Analytics — End-to-End MLOps Travel Intelligence Platform

🌐 **Live Streamlit Dashboard**: [Voyage Analytics Live Web App](https://voyage-analytics-mlops-travel-intelligence-opkcnvduve4jryt2ugr.streamlit.app/)

A modular, production-ready MLOps platform for travel booking analytics, price optimization, and customer retention. The project integrates customer transactional history (users, flights, hotels) to run unified feature engineering, model training, MLflow tracking, and live deployment.

## 🚀 Key Features

* **Three ML Pipelines**:
  1. **Flight Price Regressor**: Predicts ticket costs dynamically using route distance, time, flight type, and travel agency features.
  2. **Gender Classifier**: Predicts traveler demographics based on transactional patterns.
  3. **Churn Predictor**: Classifies customer retention risk (Active vs. Churned) using Recency, Frequency, and Monetary (RFM) transaction behavior.
* **Modular Pipeline Architecture**: Pure Python modules for data ingestion schema verification, outlier treatments, categorical mapping, feature engineering, and model evaluation.
* **Interactive Streamlit Dashboard**: A dark-themed analytics portal with 15+ Plotly charts, model metrics summaries, feature importances, and interactive forms for live predictions.
* **Flask REST API**: Light API service exposing endpoints for real-time inference and model status queries.
* **MLflow Tracking Integration**: Automatically logs hyperparameter runs and registers trained estimators to a local SQLite backend.
* **Jupyter Notebook Template**: A step-by-step walk-through following standardized submission guidelines.

---

## 📁 Project Structure

```bash
voyage-analytics/
├── data/                  # Users, flights, and hotels source datasets
├── src/                   # Core modular Python packages
│   ├── data_ingestion.py  # Enforces types and maps schemas
│   ├── preprocessing.py   # Normalized scaling and outlier capping
│   ├── feature_eng.py     # Aggregates RFM metrics
│   ├── model_training.py  # Establishes RF estimators
│   ├── model_eval.py      # Produces confusion matrix and metrics JSONs
│   └── utils.py           # Custom decorators, configuration YAML loader
├── pipelines/             # Pipeline orchestrators
│   ├── training_pipeline.py
│   └── inference_pipeline.py
├── api/                   # REST API server (Flask app & schemas)
├── streamlit_app/         # Dashboard code and thematic banners
├── notebooks/             # Guideline-aligned .ipynb template
├── reports/               # Pre-rendered evaluation charts and metrics JSONs
├── config.yaml            # Hyperparameters and folder paths
└── requirements.txt       # Dependencies
```

---

## 🛠️ Quick Start

### 1. Installation
```powershell
pip install -r requirements.txt
```

### 2. Train Models
```powershell
python pipelines/training_pipeline.py
```

### 3. Run Flask REST API
```powershell
python api/app.py
```

### 4. Run Streamlit Dashboard
```powershell
python -m streamlit run streamlit_app/app.py --server.port 8502
```

---

## 🧪 Tech Stack
* **Language**: Python 3.10+
* **ML/Data**: Pandas, Numpy, Scikit-Learn, Joblib
* **Tracking**: MLflow (SQLite backend)
* **Serving**: Flask
* **UI**: Streamlit, Plotly, Seaborn, Matplotlib
