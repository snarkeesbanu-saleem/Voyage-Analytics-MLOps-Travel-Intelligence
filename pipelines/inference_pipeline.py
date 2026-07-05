"""
inference_pipeline.py
====================
Provides prediction services using trained models and fitted preprocessors
for flight price prediction, gender classification, and churn classification.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import DataPreprocessor
from src.utils import load_config, load_model

logger = logging.getLogger(__name__)


class InferencePipeline:
    """Loads saved models and preprocessors to serve predictions on incoming requests."""

    def __init__(self, config_path: str = "config.yaml", models_dir: Optional[str] = None) -> None:
        self.config = load_config(config_path)
        if models_dir is not None:
            self.models_dir = Path(models_dir).resolve()
        else:
            self.models_dir = Path(self.config.get("paths", {}).get("models_dir", "models")).resolve()
        
        # Paths to saved objects
        self.flight_model_path = self.models_dir / "flight_price_model.joblib"
        self.flight_prep_path = self.models_dir / "flight_price_preprocessor.pkl"
        
        self.gender_model_path = self.models_dir / "gender_classifier.joblib"
        self.gender_prep_path = self.models_dir / "gender_preprocessor.pkl"

        self.churn_model_path = self.models_dir / "churn_classifier.joblib"
        self.churn_prep_path = self.models_dir / "churn_preprocessor.pkl"

        # Lazy loading holders
        self._flight_model: Any = None
        self._flight_prep: Optional[DataPreprocessor] = None
        
        self._gender_model: Any = None
        self._gender_prep: Optional[DataPreprocessor] = None

        self._churn_model: Any = None
        self._churn_prep: Optional[DataPreprocessor] = None
        
        self.load_errors: Dict[str, str] = {}
        self.check_models()

    def check_models(self) -> None:
        """Check if all required model and preprocessor files exist."""
        self.load_errors = {}
        for name, paths in [
            ("flight_price", [self.flight_model_path, self.flight_prep_path]),
            ("gender", [self.gender_model_path, self.gender_prep_path]),
            ("churn", [self.churn_model_path, self.churn_prep_path])
        ]:
            for p in paths:
                if not p.exists():
                    self.load_errors[name] = f"Missing file: {p.name}"

    # ── Lazy Loading Getters ──────────────────────────────────────────────────

    @property
    def flight_model(self) -> Any:
        if self._flight_model is None:
            self._flight_model = load_model(str(self.flight_model_path))
        return self._flight_model

    @property
    def flight_prep(self) -> DataPreprocessor:
        if self._flight_prep is None:
            prep = DataPreprocessor(self.config)
            prep.load_state(str(self.flight_prep_path))
            self._flight_prep = prep
        return self._flight_prep

    @property
    def gender_model(self) -> Any:
        if self._gender_model is None:
            self._gender_model = load_model(str(self.gender_model_path))
        return self._gender_model

    @property
    def gender_prep(self) -> DataPreprocessor:
        if self._gender_prep is None:
            prep = DataPreprocessor(self.config)
            prep.load_state(str(self.gender_prep_path))
            self._gender_prep = prep
        return self._gender_prep

    @property
    def churn_model(self) -> Any:
        if self._churn_model is None:
            self._churn_model = load_model(str(self.churn_model_path))
        return self._churn_model

    @property
    def churn_prep(self) -> DataPreprocessor:
        if self._churn_prep is None:
            prep = DataPreprocessor(self.config)
            prep.load_state(str(self.churn_prep_path))
            self._churn_prep = prep
        return self._churn_prep

    # ── Inference Methods ─────────────────────────────────────────────────────

    def predict_flight_price(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict flight price based on flight details.

        Parameters
        ----------
        input_data : dict
            Requires keys: flightType, time, distance, agency, from, to

        Returns
        -------
        dict
            Dict containing predicted_price and confidence_interval.
        """
        # Load features from config
        features = self.config.get("models", {}).get("flight_price_model", {}).get(
            "features", ["flightType", "time", "distance", "agency", "from", "to"]
        )
        
        # Build DataFrame
        df = pd.DataFrame([input_data])
        
        # Transform categoricals and numericals using saved state
        df_trans = self.flight_prep.transform(
            df,
            columns_to_encode=["flightType", "agency", "from", "to"],
            columns_to_normalize=[]
        )
        
        X = df_trans[features]
        prediction = self.flight_model.predict(X)[0]
        return {
            "predicted_price": float(prediction),
            "confidence_interval": {
                "lower": float(prediction * 0.95),
                "upper": float(prediction * 1.05)
            }
        }

    def predict_gender(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict user gender based on transaction features.

        Parameters
        ----------
        input_data : dict
            Requires keys: age, flightType, price, distance, time, agency, days, hotel_price, hotel_total

        Returns
        -------
        dict
            Dict containing predicted_gender and probability.
        """
        features = self.config.get("models", {}).get("gender_classifier", {}).get(
            "features", ["age", "flightType", "price", "distance", "time", "agency", "days", "hotel_price", "hotel_total"]
        )
        
        df = pd.DataFrame([input_data])
        
        # Transform categoricals and numericals
        df_trans = self.gender_prep.transform(
            df,
            columns_to_encode=["flightType", "agency"],
            columns_to_normalize=["age", "price", "distance", "time", "days", "hotel_price", "hotel_total"]
        )
        
        X = df_trans[features]
        pred_code = int(self.gender_model.predict(X)[0])
        gender_map = {0: "male", 1: "female"}
        gender = gender_map.get(pred_code, "unknown")
        
        prob = 1.0
        if hasattr(self.gender_model, "predict_proba"):
            probs = self.gender_model.predict_proba(X)[0]
            prob = float(probs[pred_code])

        return {
            "predicted_gender": gender,
            "probability": prob
        }

    def predict_churn(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict user churn probability.

        Parameters
        ----------
        input_data : dict
            Requires keys: total_flights, total_hotel_bookings, total_spend, avg_flight_price,
            avg_hotel_price, unique_destinations, preferred_flight_type, preferred_agency,
            age, days_since_last_trip

        Returns
        -------
        dict
            Dict containing churned and churn_probability.
        """
        features = self.config.get("models", {}).get("churn_classifier", {}).get(
            "features", [
                "total_flights", "total_hotel_bookings", "total_spend", "avg_flight_price",
                "avg_hotel_price", "unique_destinations", "preferred_flight_type",
                "preferred_agency", "age", "days_since_last_trip"
            ]
        )
        
        df = pd.DataFrame([input_data])
        
        # Transform categoricals and numericals
        df_trans = self.churn_prep.transform(
            df,
            columns_to_encode=["preferred_flight_type", "preferred_agency"],
            columns_to_normalize=[
                "total_flights", "total_hotel_bookings", "total_spend", "avg_flight_price",
                "avg_hotel_price", "unique_destinations", "age", "days_since_last_trip"
            ]
        )
        
        X = df_trans[features]
        pred_code = int(self.churn_model.predict(X)[0])
        
        prob = 1.0
        if hasattr(self.churn_model, "predict_proba"):
            probs = self.churn_model.predict_proba(X)[0]
            prob = float(probs[1])  # probability of churned=1

        return {
            "churned": pred_code,
            "churn_probability": prob
        }
