"""
voyage_dag.py
=============
Apache Airflow DAG for orchestrating the Voyage Analytics MLOps training pipeline.
Runs weekly to ingest data, preprocess, engineer features, train and evaluate models,
and log the results to MLflow.
"""

from datetime import datetime, timedelta
import os
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator

# Ensure project modules are importable in Airflow worker paths
PROJECT_ROOT = "/opt/airflow/dags/voyage-analytics"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

default_args = {
    "owner": "voyage_analytics",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2026, 1, 1),
}


def run_ingestion():
    from src.data_ingestion import DataIngestion
    from src.utils import load_config
    config = load_config(f"{PROJECT_ROOT}/config.yaml")
    ingestion = DataIngestion(config)
    users, flights, hotels = ingestion.load_all()
    # Save temp states if needed, or rely on pipeline database/storage
    print(f"Data Ingestion Complete. Users: {len(users)}, Flights: {len(flights)}, Hotels: {len(hotels)}")


def run_preprocessing_and_training():
    from pipelines.training_pipeline import TrainingPipeline
    pipeline = TrainingPipeline(config_path=f"{PROJECT_ROOT}/config.yaml")
    pipeline.run_all()
    print("Pre-processing, training, evaluation, and registration completed successfully.")


with DAG(
    "voyage_analytics_pipeline",
    default_args=default_args,
    description="End-to-end retraining pipeline for Voyage Analytics models",
    schedule_interval="@weekly",
    catchup=False,
) as dag:

    task_ingest = PythonOperator(
        task_id="ingest_data",
        python_callable=run_ingestion,
    )

    task_train = PythonOperator(
        task_id="preprocess_train_evaluate_register",
        python_callable=run_preprocessing_and_training,
    )

    task_ingest >> task_train
