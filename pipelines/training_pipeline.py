"""
training_pipeline.py
===================
Orchestration script that runs the end-to-end training pipeline:
ingestion, preprocessing, feature engineering, model training, evaluation,
and MLflow tracking for all three models.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlflow_tracking.tracking_config import MLflowTracker
from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_evaluation import ModelEvaluator
from src.model_training import ModelTrainer
from src.utils import load_config, setup_logging

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Orchestrates end-to-end model training, evaluation, and registry."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        self.config = load_config(config_path)
        
        # Load directories
        paths = self.config.get("paths", {})
        self.models_dir = Path(paths.get("models_dir", "models")).resolve()
        self.reports_dir = Path(paths.get("reports_dir", "reports")).resolve()
        self.logs_dir = Path(paths.get("logs_dir", "logs")).resolve()
        
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging using utils.py
        setup_logging(
            level=self.config.get("logging", {}).get("level", "INFO"),
            format_str=self.config.get("logging", {}).get("format", "%(asctime)s — %(name)s — %(levelname)s — %(message)s"),
            log_file=self.config.get("logging", {}).get("log_file", "logs/voyage_analytics.log"),
        )
        
        # Initialize modules
        self.ingestion = DataIngestion(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.trainer = ModelTrainer(self.config)
        self.evaluator = ModelEvaluator(self.config)
        self.tracker = MLflowTracker(self.config)

    def run_flight_price_pipeline(self, flights_df: Any) -> None:
        """Run training pipeline for the Flight Price regression model."""
        logger.info("==================================================")
        logger.info("STARTING FLIGHT PRICE MODEL TRAINING PIPELINE")
        logger.info("==================================================")

        # 1. Preprocess
        preprocessor = DataPreprocessor(self.config)
        df_clean = preprocessor.handle_missing_values(flights_df, strategy="median")
        df_clean = preprocessor.encode_categoricals(df_clean, columns=["flightType", "agency", "from", "to"])
        df_clean = preprocessor.remove_outliers(df_clean, columns=["price"])
        
        # Save preprocessor state
        preprocessor.save_state(str(self.models_dir / "flight_price_preprocessor.pkl"))

        # 2. Feature engineering
        X, y = self.feature_engineer.build_flight_price_features(df_clean)

        # 3. Model training
        model, X_train, X_test, y_train, y_test = self.trainer.train_flight_price_model(X, y)

        # 4. Model evaluation
        metrics = self.evaluator.evaluate_regression(model, X_test, y_test, model_name="flight_price")

        # 5. MLflow log
        model_config = self.config.get("models", {}).get("flight_price_model", {})
        params = model_config.get("estimators", {}).get("random_forest", {}).get("params", {})
        self.tracker.log_run(
            experiment_name="Flight_Price_Regression",
            run_name="RandomForest_Regressor_Run",
            params=params,
            metrics=metrics,
            model=model,
            model_artifact_name="flight_price_model",
            artifacts=[
                str(self.reports_dir / "flight_price_metrics.json"),
                str(self.reports_dir / "flight_price_actual_vs_predicted.png"),
                str(self.reports_dir / "flight_price_feature_importance.png"),
            ]
        )
        logger.info("Flight Price pipeline finished successfully.")

    def run_gender_pipeline(self, users_df: Any, flights_df: Any, hotels_df: Any) -> None:
        """Run training pipeline for the Gender classification model."""
        logger.info("==================================================")
        logger.info("STARTING GENDER MODEL TRAINING PIPELINE")
        logger.info("==================================================")

        # 1. Feature Engineering (Raw Merge)
        X_raw, y = self.feature_engineer.build_gender_features(users_df, flights_df, hotels_df)
        
        # 2. Preprocess (categorical encoding & normalisation)
        preprocessor = DataPreprocessor(self.config)
        X_clean = preprocessor.handle_missing_values(X_raw, strategy="median")
        X_clean = preprocessor.encode_categoricals(X_clean, columns=["flightType", "agency"])
        
        # Normalise numericals
        num_cols = ["age", "price", "distance", "time", "days", "hotel_price", "hotel_total"]
        X_clean = preprocessor.normalize_numericals(X_clean, columns=num_cols)
        
        # Save preprocessor state
        preprocessor.save_state(str(self.models_dir / "gender_preprocessor.pkl"))

        # 3. Model training
        model, X_train, X_test, y_train, y_test = self.trainer.train_gender_classifier(X_clean, y)

        # 4. Model evaluation
        metrics = self.evaluator.evaluate_classification(model, X_test, y_test, model_name="gender")

        # 5. MLflow log
        model_config = self.config.get("models", {}).get("gender_classifier", {})
        params = model_config.get("estimators", {}).get("random_forest", {}).get("params", {})
        self.tracker.log_run(
            experiment_name="Gender_Classification",
            run_name="RandomForest_Classifier_Run",
            params=params,
            metrics=metrics,
            model=model,
            model_artifact_name="gender_classifier",
            artifacts=[
                str(self.reports_dir / "gender_metrics.json"),
                str(self.reports_dir / "gender_confusion_matrix.png"),
                str(self.reports_dir / "gender_feature_importance.png"),
            ]
        )
        logger.info("Gender classification pipeline finished successfully.")

    def run_churn_pipeline(self, users_df: Any, flights_df: Any, hotels_df: Any) -> None:
        """Run training pipeline for the Churn classification model."""
        logger.info("==================================================")
        logger.info("STARTING CHURN MODEL TRAINING PIPELINE")
        logger.info("==================================================")

        # 1. Feature Engineering (Aggregations)
        X_raw, y = self.feature_engineer.build_churn_features(users_df, flights_df, hotels_df)

        # 2. Preprocess (categorical encoding & normalisation)
        preprocessor = DataPreprocessor(self.config)
        X_clean = preprocessor.handle_missing_values(X_raw, strategy="median")
        X_clean = preprocessor.encode_categoricals(X_clean, columns=["preferred_flight_type", "preferred_agency"])
        
        num_cols = [
            "total_flights", "total_hotel_bookings", "total_spend", "avg_flight_price",
            "avg_hotel_price", "unique_destinations", "age", "days_since_last_trip"
        ]
        X_clean = preprocessor.normalize_numericals(X_clean, columns=num_cols)

        # Save preprocessor state
        preprocessor.save_state(str(self.models_dir / "churn_preprocessor.pkl"))

        # 3. Model training
        model, X_train, X_test, y_train, y_test = self.trainer.train_churn_classifier(X_clean, y)

        # 4. Model evaluation
        metrics = self.evaluator.evaluate_classification(model, X_test, y_test, model_name="churn")

        # 5. MLflow log
        model_config = self.config.get("models", {}).get("churn_classifier", {})
        params = model_config.get("estimators", {}).get("random_forest", {}).get("params", {})
        self.tracker.log_run(
            experiment_name="Churn_Classification",
            run_name="RandomForest_Classifier_Run",
            params=params,
            metrics=metrics,
            model=model,
            model_artifact_name="churn_classifier",
            artifacts=[
                str(self.reports_dir / "churn_metrics.json"),
                str(self.reports_dir / "churn_confusion_matrix.png"),
                str(self.reports_dir / "churn_feature_importance.png"),
            ]
        )
        logger.info("Churn classification pipeline finished successfully.")

    def run_all(self) -> None:
        """Run all pipelines in sequence."""
        logger.info("Executing full Voyage Analytics training pipeline...")
        users_df, flights_df, hotels_df = self.ingestion.load_all()

        # Run individual pipelines
        self.run_flight_price_pipeline(flights_df)
        self.run_gender_pipeline(users_df, flights_df, hotels_df)
        self.run_churn_pipeline(users_df, flights_df, hotels_df)

        logger.info("All model training pipelines completed successfully.")


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_all()
