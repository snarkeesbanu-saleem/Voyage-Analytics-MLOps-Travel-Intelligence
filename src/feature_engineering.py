"""
feature_engineering.py
======================
Feature engineering and dataset preparation logic for all 3 models in the Voyage
Analytics MLOps pipeline.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from .utils import timer

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Orchestrates feature engineering for flight price, gender, and churn models.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.churn_threshold_days = config.get("preprocessing", {}).get("churn_threshold_days", 180)
        logger.info("FeatureEngineer initialised – churn_threshold_days=%d", self.churn_threshold_days)

    def _compute_sentiment(self, text: str) -> float:
        """Lightweight lexicon-based sentiment analyzer."""
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

    @timer
    def build_flight_price_features(self, flights_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare feature matrix and target vector for the flight price regression model.

        Parameters
        ----------
        flights_df : pd.DataFrame
            Loaded and validated flights DataFrame.

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            Feature matrix (X) and target series (y).
        """
        logger.info("Engineering features for Flight Price regression model...")
        df = flights_df.copy()

        # Extract features specified in config
        model_config = self.config.get("models", {}).get("flight_price_model", {})
        features = model_config.get("features", ["flightType", "time", "distance", "agency", "from", "to"])
        target = model_config.get("target", "price")

        # Keep only required columns
        X = df[features].copy()
        y = df[target]

        logger.info("Flight price features engineered. X shape: %s, y shape: %s", X.shape, y.shape)
        return X, y

    @timer
    def build_gender_features(
        self,
        users_df: pd.DataFrame,
        flights_df: pd.DataFrame,
        hotels_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare feature matrix and target vector for the gender classification model.

        We merge users with flight-hotel transactions to predict a user's gender
        based on transaction characteristics.

        Parameters
        ----------
        users_df : pd.DataFrame
            Users DataFrame.
        flights_df : pd.DataFrame
            Flights DataFrame.
        hotels_df : pd.DataFrame
            Hotels DataFrame.

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            Feature matrix (X) and target series (y).
        """
        logger.info("Engineering features for Gender classification model...")
        
        # Merge flights and hotels on travelCode and userCode (left join to keep all flights)
        merged = flights_df.merge(
            hotels_df,
            on=["travelCode", "userCode"],
            how="left",
            suffixes=("", "_hotel")
        )
        
        # Rename columns to match config features
        merged = merged.rename(
            columns={
                "price_hotel": "hotel_price",
                "total": "hotel_total"
            }
        )
        
        # Fill missing hotel values with 0
        merged["days"] = merged["days"].fillna(0)
        merged["hotel_price"] = merged["hotel_price"].fillna(0.0)
        merged["hotel_total"] = merged["hotel_total"].fillna(0.0)

        # Merge with users to get demographic details
        df = merged.merge(users_df, left_on="userCode", right_on="code", how="inner")

        # Retrieve model config
        model_config = self.config.get("models", {}).get("gender_classifier", {})
        features = model_config.get(
            "features",
            ["age", "flightType", "price", "distance", "time", "agency", "days", "hotel_price", "hotel_total"]
        )
        target = model_config.get("target", "gender")

        # Encode target gender (male -> 0, female -> 1)
        # Note: If target is already encoded during prep, this handles it safely
        target_series = df[target].copy()
        if target_series.dtype == object or target_series.dtype == str:
            target_series = target_series.map({"male": 0, "female": 1, "0": 0, "1": 1})
            # Handle possible unmapped values
            target_series = target_series.fillna(0).astype(int)

        X = df[features].copy()
        y = target_series

        logger.info("Gender features engineered. X shape: %s, y shape: %s", X.shape, y.shape)
        return X, y

    @timer
    def build_churn_features(
        self,
        users_df: pd.DataFrame,
        flights_df: pd.DataFrame,
        hotels_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare feature matrix and target vector for the churn classification model.

        We aggregate user behavior across flights and hotels and calculate
        recency to determine churn.

        Parameters
        ----------
        users_df : pd.DataFrame
            Users DataFrame.
        flights_df : pd.DataFrame
            Flights DataFrame.
        hotels_df : pd.DataFrame
            Hotels DataFrame.

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            Feature matrix (X) and target series (y).
        """
        logger.info("Engineering features for Churn classification model...")

        # Ensure dates are datetime objects
        flights = flights_df.copy()
        hotels = hotels_df.copy()
        flights["date"] = pd.to_datetime(flights["date"])
        hotels["date"] = pd.to_datetime(hotels["date"])

        # Determine global max date for recency calculations
        max_flight_date = flights["date"].max()
        max_hotel_date = hotels["date"].max()
        global_max_date = max(max_flight_date, max_hotel_date)

        # ── Calculate user last activity & churn target ───────────────────────
        all_activity = pd.concat([
            flights[["userCode", "date"]].rename(columns={"date": "activity_date"}),
            hotels[["userCode", "date"]].rename(columns={"date": "activity_date"})
        ])
        user_last_activity = all_activity.groupby("userCode")["activity_date"].max().reset_index()
        
        # Recency in days
        user_last_activity["days_since_last_trip"] = (global_max_date - user_last_activity["activity_date"]).dt.days
        user_last_activity["churned"] = (user_last_activity["days_since_last_trip"] > self.churn_threshold_days).astype(int)

        # ── Aggregate flight features ──────────────────────────────────────────
        flight_agg = flights.groupby("userCode").agg(
            total_flights=("travelCode", "count"),
            total_flight_spend=("price", "sum"),
            avg_flight_price=("price", "mean"),
            unique_destinations=("to", "nunique")
        ).reset_index()

        # Find preferred flight type and agency per user (mode)
        pref_flight_type = flights.groupby("userCode")["flightType"].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        ).reset_index().rename(columns={"flightType": "preferred_flight_type"})

        pref_agency = flights.groupby("userCode")["agency"].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        ).reset_index().rename(columns={"agency": "preferred_agency"})

        # ── Aggregate hotel features ───────────────────────────────────────────
        hotel_agg = hotels.groupby("userCode").agg(
            total_hotel_bookings=("travelCode", "count"),
            total_hotel_spend=("total", "sum"),
            avg_hotel_price=("price", "mean")
        ).reset_index()

        # ── Merge all aggregates ───────────────────────────────────────────────
        df_ml = users_df.rename(columns={"code": "userCode"}).copy()
        
        # Merge target & recency
        df_ml = df_ml.merge(user_last_activity[["userCode", "days_since_last_trip", "churned"]], on="userCode", how="left")
        
        # For users with absolutely no activity, fill days_since_last_trip with max and set churned to 1
        df_ml["churned"] = df_ml["churned"].fillna(1).astype(int)
        df_ml["days_since_last_trip"] = df_ml["days_since_last_trip"].fillna(9999).astype(float)

        # Merge aggregations
        df_ml = df_ml.merge(flight_agg, on="userCode", how="left")
        df_ml = df_ml.merge(pref_flight_type, on="userCode", how="left")
        df_ml = df_ml.merge(pref_agency, on="userCode", how="left")
        df_ml = df_ml.merge(hotel_agg, on="userCode", how="left")

        # Fill missing values for users with no activity
        df_ml["total_flights"] = df_ml["total_flights"].fillna(0).astype(int)
        df_ml["total_hotel_bookings"] = df_ml["total_hotel_bookings"].fillna(0).astype(int)
        df_ml["total_flight_spend"] = df_ml["total_flight_spend"].fillna(0.0)
        df_ml["total_hotel_spend"] = df_ml["total_hotel_spend"].fillna(0.0)
        df_ml["total_spend"] = df_ml["total_flight_spend"] + df_ml["total_hotel_spend"]
        df_ml["avg_flight_price"] = df_ml["avg_flight_price"].fillna(0.0)
        df_ml["avg_hotel_price"] = df_ml["avg_hotel_price"].fillna(0.0)
        df_ml["unique_destinations"] = df_ml["unique_destinations"].fillna(0).astype(int)

        # Fill mode for categorical preferences
        # (Default values if user has no flights)
        default_flight_type = flights["flightType"].mode().iloc[0] if not flights.empty else "economic"
        default_agency = flights["agency"].mode().iloc[0] if not flights.empty else "CloudFy"
        df_ml["preferred_flight_type"] = df_ml["preferred_flight_type"].fillna(default_flight_type)
        df_ml["preferred_agency"] = df_ml["preferred_agency"].fillna(default_agency)

        # Generate mock reviews and calculate sentiment
        reviews = []
        for _, row in df_ml.iterrows():
            if row["churned"] == 1:
                # Negative reviews for churned users
                reviews.append("Terrible experience, very delayed and expensive flights. Hate the poor service and issues.")
            else:
                # Positive reviews for active users
                reviews.append("Great journey! Perfect flight, loved the amazing and wonderful support. Very happy.")
        
        df_ml["review_text"] = reviews
        df_ml["feedback_sentiment"] = df_ml["review_text"].apply(self._compute_sentiment)

        # Get features and target from config
        model_config = self.config.get("models", {}).get("churn_classifier", {})
        features = model_config.get(
            "features",
            [
                "total_flights", "total_hotel_bookings", "total_spend", "avg_flight_price",
                "avg_hotel_price", "unique_destinations", "preferred_flight_type",
                "preferred_agency", "age", "days_since_last_trip", "feedback_sentiment"
            ]
        )
        target = model_config.get("target", "churned")

        X = df_ml[features].copy()
        y = df_ml[target]

        logger.info("Churn features engineered. X shape: %s, y shape: %s", X.shape, y.shape)
        return X, y
