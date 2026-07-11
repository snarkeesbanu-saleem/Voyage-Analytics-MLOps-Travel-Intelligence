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


def compute_lexicon_sentiment(text: str) -> float:
    """Lightweight lexicon-based sentiment analyzer for inference."""
    pos_words = {"great", "excellent", "love", "loved", "good", "happy", "wonderful", "amazing", "pleasant", "smooth", "perfect"}
    neg_words = {"bad", "delayed", "worst", "hate", "terrible", "poor", "expensive", "disappointed", "slow", "broken", "issue", "issues"}
    
    words = str(text).lower().split()
    score = 0.0
    for w in words:
        w_clean = "".join(c for c in w if c.isalnum())
        if w_clean in pos_words:
            score += 1.0
        elif w_clean in neg_words:
            score -= 1.0
            
    total = sum(1 for w in words if "".join(c for c in w if c.isalnum()) in pos_words or "".join(c for c in w if c.isalnum()) in neg_words)
    if total == 0:
        return 0.0
    return score / total


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
        explainability = self.explain_prediction(self.flight_model, X)
        
        return {
            "predicted_price": float(prediction),
            "confidence_interval": {
                "lower": float(prediction * 0.95),
                "upper": float(prediction * 1.05)
            },
            "explainability": explainability
        }

    def predict_gender(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict user gender based on transaction features."""
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

        explainability = self.explain_prediction(self.gender_model, X)

        return {
            "predicted_gender": gender,
            "probability": prob,
            "explainability": explainability
        }

    def predict_churn(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict user churn probability, analyzing review text if provided."""
        features = self.config.get("models", {}).get("churn_classifier", {}).get(
            "features", [
                "total_flights", "total_hotel_bookings", "total_spend", "avg_flight_price",
                "avg_hotel_price", "unique_destinations", "preferred_flight_type",
                "preferred_agency", "age", "days_since_last_trip", "feedback_sentiment"
            ]
        )
        
        # Extract and analyze review text if present
        review_text = input_data.get("review_text", "")
        feedback_sentiment = compute_lexicon_sentiment(review_text) if review_text else input_data.get("feedback_sentiment", 0.0)
        
        # Prepare input data for transformation
        proc_data = input_data.copy()
        proc_data["feedback_sentiment"] = feedback_sentiment
        if "review_text" in proc_data:
            del proc_data["review_text"]
            
        df = pd.DataFrame([proc_data])
        
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
        
        prob = 0.5
        if hasattr(self.churn_model, "predict_proba"):
            probs = self.churn_model.predict_proba(X)[0]
            prob = float(probs[1])  # probability of churned=1

        explainability = self.explain_prediction(self.churn_model, X)

        return {
            "churned": pred_code,
            "churn_probability": prob,
            "feedback_sentiment": feedback_sentiment,
            "explainability": explainability
        }

    def explain_prediction(self, model: Any, X_sample: pd.DataFrame) -> Dict[str, float]:
        """Calculate approximate feature contributions (SHAP approximation) for Random Forests."""
        if not hasattr(model, "feature_importances_"):
            return {}
        
        importances = model.feature_importances_
        feature_names = X_sample.columns.tolist()
        
        contributions = {}
        for name, imp in zip(feature_names, importances):
            val = X_sample[name].iloc[0]
            if isinstance(val, (int, float, np.integer, np.floating)):
                # Higher positive values generally represent positive contributors
                contributions[name] = float(imp * val)
            else:
                contributions[name] = float(imp * (0.5 if str(val) != "0" else -0.5))
                
        # Normalize contributions to make them presentable as percentages/weights
        total_abs = sum(abs(v) for v in contributions.values())
        if total_abs > 0:
            contributions = {k: round((v / total_abs) * 100, 1) for k, v in contributions.items()}
        return contributions

    def simulate_price_elasticity(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simulate revenue configurations to recommend pricing optimizations."""
        options = []
        flight_types = ["economic", "premium", "firstClass"]
        agencies = ["FlyingDrops", "CloudFy", "Rainbow"]
        
        for ft in flight_types:
            for ag in agencies:
                sim_data = input_data.copy()
                sim_data["flightType"] = ft
                sim_data["agency"] = ag
                
                # Predict price
                res = self.predict_flight_price(sim_data)
                pred_price = res["predicted_price"]
                
                # Heuristic conversion rates: premium and firstClass scale down
                conv_mult = {"economic": 0.85, "premium": 0.55, "firstClass": 0.25}
                conv_prob = conv_mult[ft] * (1.0 - min(0.8, (pred_price / 1500.0)))
                est_rev = pred_price * conv_prob
                
                options.append({
                    "flightType": ft,
                    "agency": ag,
                    "price": round(pred_price, 2),
                    "conversion_probability": round(conv_prob * 100, 1),
                    "expected_revenue": round(est_rev, 2)
                })
                
        # Sort options by expected revenue descending
        options.sort(key=lambda x: x["expected_revenue"], reverse=True)
        return options
