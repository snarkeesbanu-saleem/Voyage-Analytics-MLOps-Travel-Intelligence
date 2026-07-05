"""
model_training.py
=================
Training logic for all 3 models in the Voyage Analytics MLOps pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .utils import save_model, timer

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains the Voyage Analytics machine learning models.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.models_dir = Path(config.get("paths", {}).get("models_dir", "models")).resolve()
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = config.get("preprocessing", {}).get("random_state", 42)
        self.test_size = config.get("preprocessing", {}).get("test_size", 0.2)
        logger.info("ModelTrainer initialised – models_dir=%s", self.models_dir)

    @timer
    def train_flight_price_model(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[Union[RandomForestRegressor, GradientBoostingRegressor], pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Train a flight price regression model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target series.

        Returns
        -------
        Tuple
            (model, X_train, X_test, y_train, y_test)
        """
        logger.info("Training Flight Price Regression Model...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        model_config = self.config.get("models", {}).get("flight_price_model", {})
        # Choose estimator
        estimator_name = "random_forest"  # Default
        estimator_config = model_config.get("estimators", {}).get(estimator_name, {})
        params = estimator_config.get("params", {})
        
        # Strip parameters not recognized by sklearn
        clean_params = {k: v for k, v in params.items() if k != "type"}

        logger.info("Selected model: %s with params: %s", estimator_name, clean_params)
        model = RandomForestRegressor(**clean_params)
        
        logger.info("Fitting Flight Price model...")
        model.fit(X_train, y_train)

        # Save model
        save_path = self.models_dir / "flight_price_model.joblib"
        save_model(model, str(save_path))
        logger.info("Flight Price model trained and saved successfully.")

        return model, X_train, X_test, y_train, y_test

    @timer
    def train_gender_classifier(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[Union[RandomForestClassifier, LogisticRegression], pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Train a gender classifier model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target series.

        Returns
        -------
        Tuple
            (model, X_train, X_test, y_train, y_test)
        """
        logger.info("Training Gender Classification Model...")
        # Stratify by gender (y) since it is classification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        model_config = self.config.get("models", {}).get("gender_classifier", {})
        estimator_name = "random_forest"  # Default
        estimator_config = model_config.get("estimators", {}).get(estimator_name, {})
        params = estimator_config.get("params", {})
        
        clean_params = {k: v for k, v in params.items() if k != "type"}

        logger.info("Selected model: %s with params: %s", estimator_name, clean_params)
        model = RandomForestClassifier(**clean_params)

        logger.info("Fitting Gender classifier model...")
        model.fit(X_train, y_train)

        # Save model
        save_path = self.models_dir / "gender_classifier.joblib"
        save_model(model, str(save_path))
        logger.info("Gender classifier model trained and saved successfully.")

        return model, X_train, X_test, y_train, y_test

    @timer
    def train_churn_classifier(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[RandomForestClassifier, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Train a churn classifier model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target series.

        Returns
        -------
        Tuple
            (model, X_train, X_test, y_train, y_test)
        """
        logger.info("Training Churn Classification Model...")
        # Stratify by churn status
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        model_config = self.config.get("models", {}).get("churn_classifier", {})
        estimator_name = "random_forest"  # Default
        estimator_config = model_config.get("estimators", {}).get(estimator_name, {})
        params = estimator_config.get("params", {})
        
        clean_params = {k: v for k, v in params.items() if k != "type"}

        logger.info("Selected model: %s with params: %s", estimator_name, clean_params)
        model = RandomForestClassifier(**clean_params)

        logger.info("Fitting Churn classifier model...")
        model.fit(X_train, y_train)

        # Save model
        save_path = self.models_dir / "churn_classifier.joblib"
        save_model(model, str(save_path))
        logger.info("Churn classifier model trained and saved successfully.")

        return model, X_train, X_test, y_train, y_test
