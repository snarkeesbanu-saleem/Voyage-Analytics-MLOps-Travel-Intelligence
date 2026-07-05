"""
conftest.py
===========
Pytest fixtures and configuration setup for unit and integration testing.
"""

from __future__ import annotations

from pathlib import Path
import pytest
import pandas as pd
import yaml


@pytest.fixture(scope="session")
def sample_config() -> dict:
    """Fixture returning a mock configuration dictionary."""
    return {
        "paths": {
            "data_dir": "data/",
            "models_dir": "models/",
            "logs_dir": "logs/",
            "reports_dir": "reports/",
        },
        "preprocessing": {
            "outlier_iqr_multiplier": 1.5,
            "test_size": 0.2,
            "random_state": 42,
            "churn_threshold_days": 180,
        },
        "models": {
            "flight_price_model": {
                "target": "price",
                "features": ["flightType", "time", "distance", "agency", "from", "to"],
                "estimators": {
                    "random_forest": {
                        "params": {
                            "n_estimators": 5,
                            "max_depth": 5,
                            "random_state": 42,
                        }
                    }
                }
            },
            "gender_classifier": {
                "target": "gender",
                "features": ["age", "flightType", "price", "distance", "time", "agency", "days", "hotel_price", "hotel_total"],
                "estimators": {
                    "random_forest": {
                        "params": {
                            "n_estimators": 5,
                            "max_depth": 5,
                            "random_state": 42,
                        }
                    }
                }
            },
            "churn_classifier": {
                "target": "churned",
                "features": [
                    "total_flights", "total_hotel_bookings", "total_spend", "avg_flight_price",
                    "avg_hotel_price", "unique_destinations", "preferred_flight_type",
                    "preferred_agency", "age", "days_since_last_trip"
                ],
                "estimators": {
                    "random_forest": {
                        "params": {
                            "n_estimators": 5,
                            "max_depth": 5,
                            "random_state": 42,
                        }
                    }
                }
            }
        }
    }


@pytest.fixture
def dummy_users_df() -> pd.DataFrame:
    """Sample Users dataset."""
    return pd.DataFrame({
        "code": [0, 1, 2],
        "company": ["Company A", "Company B", "Company A"],
        "name": ["Alice", "Bob", "Charlie"],
        "gender": ["female", "male", "female"],
        "age": [25, 40, 32]
    })


@pytest.fixture
def dummy_flights_df() -> pd.DataFrame:
    """Sample Flights dataset."""
    return pd.DataFrame({
        "travelCode": [0, 1, 2],
        "userCode": [0, 1, 0],
        "from": ["Recife (PE)", "Aracaju (SE)", "Recife (PE)"],
        "to": ["Florianopolis (SC)", "Brasilia (DF)", "Aracaju (SE)"],
        "flightType": ["firstClass", "economic", "premium"],
        "price": [1200.0, 500.0, 800.0],
        "time": [2.5, 1.2, 1.8],
        "distance": [1500.0, 600.0, 1000.0],
        "agency": ["FlyingDrops", "CloudFy", "Rainbow"],
        "date": ["01/01/2022", "02/15/2022", "06/30/2022"]
    })


@pytest.fixture
def dummy_hotels_df() -> pd.DataFrame:
    """Sample Hotels dataset."""
    return pd.DataFrame({
        "travelCode": [0, 1, 2],
        "userCode": [0, 1, 0],
        "name": ["Hotel A", "Hotel B", "Hotel A"],
        "place": ["Florianopolis (SC)", "Brasilia (DF)", "Aracaju (SE)"],
        "days": [3, 2, 4],
        "price": [150.0, 200.0, 150.0],
        "total": [450.0, 400.0, 600.0],
        "date": ["01/01/2022", "02/15/2022", "06/30/2022"]
    })
