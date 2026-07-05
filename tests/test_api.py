"""
test_api.py
===========
Flask API endpoints verification with pytest client.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from api.app import create_app


@pytest.fixture
def client():
    """Setup Flask test client with mocks for InferencePipeline."""
    with patch("api.app.InferencePipeline") as MockPipeline:
        # Configure the mock instance
        mock_instance = MockPipeline.return_value
        mock_instance.load_errors = {}
        mock_instance.predict_flight_price.return_value = {
            "predicted_price": 750.0,
            "confidence_interval": {"lower": 712.5, "upper": 787.5}
        }
        mock_instance.predict_gender.return_value = {
            "predicted_gender": "female",
            "probability": 0.85
        }
        mock_instance.predict_churn.return_value = {
            "churned": 1,
            "churn_probability": 0.92
        }
        
        app = create_app()
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client


def test_api_health(client):
    """Verify healthcheck endpoint response."""
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "healthy"


def test_flight_price_endpoint(client):
    """Verify flight price prediction endpoint validation and response."""
    payload = {
        "from_city": "Recife (PE)",
        "to_city": "Florianopolis (SC)",
        "flight_type": "economic",
        "agency": "CloudFy",
        "distance": 1200.0,
        "time": 2.5,
        "month": 6,
        "day_of_week": 3
    }
    response = client.post("/predict/flight-price", json=payload)
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "success"
    assert json_data["data"]["predicted_price"] == 750.0


def test_gender_endpoint(client):
    """Verify gender prediction endpoint validation and response."""
    payload = {
        "age": 32,
        "flightType": "economic",
        "price": 450.0,
        "distance": 800.0,
        "time": 1.5,
        "agency": "CloudFy",
        "days": 0,
        "hotel_price": 0.0,
        "hotel_total": 0.0
    }
    response = client.post("/predict/gender", json=payload)
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "success"
    assert json_data["data"]["predicted_gender"] == "female"
    assert json_data["data"]["probability"] == 0.85


def test_churn_endpoint(client):
    """Verify churn prediction endpoint validation and response."""
    payload = {
        "total_flights": 3,
        "total_hotel_bookings": 1,
        "total_spend": 2500.0,
        "avg_flight_price": 500.0,
        "avg_hotel_price": 200.0,
        "unique_destinations": 2,
        "preferred_flight_type": "economic",
        "preferred_agency": "CloudFy",
        "age": 45,
        "days_since_last_trip": 200.0
    }
    response = client.post("/predict/churn", json=payload)
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "success"
    assert json_data["data"]["churn_probability"] == 0.92
    assert json_data["data"]["risk_level"] == "high"
