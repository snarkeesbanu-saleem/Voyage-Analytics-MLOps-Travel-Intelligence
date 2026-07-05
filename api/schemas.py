"""
Voyage Analytics - API Request/Response Schemas
===============================================
Dataclass-based schemas for validating API requests and serializing responses.
Maps to the underlying dataset columns:
  - flights.csv: travelCode, userCode, from, to, flightType, price, time, distance, agency, date
  - users.csv: code, company, name, gender, age
  - hotels.csv: travelCode, userCode, name, place, days, price, total, date
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Flight Price Prediction
# ---------------------------------------------------------------------------

@dataclass
class FlightPriceRequest:
    """Input features for flight price prediction.

    Attributes:
        from_city: Departure city (e.g. 'Recife (PE)').
        to_city: Destination city (e.g. 'Florianopolis (SC)').
        flight_type: Class of travel – 'economic', 'firstClass', 'premium'.
        agency: Booking agency name (e.g. 'FlyingDrops', 'Rainbow').
        distance: Route distance in km.
        time: Flight duration in hours.
        month: Month of travel (1-12).
        day_of_week: Day of week (0=Mon … 6=Sun).
    """

    from_city: str
    to_city: str
    flight_type: str
    agency: str
    distance: float
    time: float
    month: int
    day_of_week: int

    def validate(self) -> List[str]:
        """Return a list of validation error messages (empty == valid)."""
        errors: List[str] = []
        if not self.from_city or not self.from_city.strip():
            errors.append("from_city is required")
        if not self.to_city or not self.to_city.strip():
            errors.append("to_city is required")
        valid_types = {"economic", "firstClass", "premium"}
        if self.flight_type not in valid_types:
            errors.append(f"flight_type must be one of {valid_types}")
        if self.distance <= 0:
            errors.append("distance must be positive")
        if self.time <= 0:
            errors.append("time must be positive")
        if not 1 <= self.month <= 12:
            errors.append("month must be between 1 and 12")
        if not 0 <= self.day_of_week <= 6:
            errors.append("day_of_week must be between 0 (Mon) and 6 (Sun)")
        return errors

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FlightPriceRequest":
        """Construct from a JSON-decoded dict with type coercion."""
        try:
            return cls(
                from_city=str(data["from_city"]),
                to_city=str(data["to_city"]),
                flight_type=str(data["flight_type"]),
                agency=str(data["agency"]),
                distance=float(data["distance"]),
                time=float(data["time"]),
                month=int(data["month"]),
                day_of_week=int(data["day_of_week"]),
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"Invalid FlightPriceRequest payload: {exc}") from exc

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FlightPriceResponse:
    """Predicted flight price with confidence interval."""

    predicted_price: float
    confidence_interval: Dict[str, float] = field(
        default_factory=lambda: {"lower": 0.0, "upper": 0.0}
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Gender Classification
# ---------------------------------------------------------------------------

@dataclass
class GenderRequest:
    """User-behaviour features used for gender classification.

    Attributes:
        age: User age in years.
        flightType: Class of flight (e.g. 'economic', 'firstClass', 'premium').
        price: Price of flight.
        distance: Distance of flight.
        time: Time of flight.
        agency: Flight agency.
        days: Days stayed in hotel.
        hotel_price: Price per night of hotel.
        hotel_total: Total spending on hotel.
    """

    age: int
    flightType: str
    price: float
    distance: float
    time: float
    agency: str
    days: int
    hotel_price: float
    hotel_total: float

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.age <= 0 or self.age > 150:
            errors.append("age must be between 1 and 150")
        valid_types = {"economic", "firstClass", "premium"}
        if self.flightType not in valid_types:
            errors.append(f"flightType must be one of {valid_types}")
        if self.price < 0:
            errors.append("price must be non-negative")
        if self.distance < 0:
            errors.append("distance must be non-negative")
        if self.time < 0:
            errors.append("time must be non-negative")
        if self.days < 0:
            errors.append("days must be non-negative")
        if self.hotel_price < 0:
            errors.append("hotel_price must be non-negative")
        if self.hotel_total < 0:
            errors.append("hotel_total must be non-negative")
        return errors

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenderRequest":
        try:
            return cls(
                age=int(data["age"]),
                flightType=str(data["flightType"]),
                price=float(data["price"]),
                distance=float(data["distance"]),
                time=float(data["time"]),
                agency=str(data["agency"]),
                days=int(data["days"]),
                hotel_price=float(data["hotel_price"]),
                hotel_total=float(data["hotel_total"]),
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"Invalid GenderRequest payload: {exc}") from exc

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GenderResponse:
    """Gender prediction result."""

    predicted_gender: str
    probability: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Churn Prediction
# ---------------------------------------------------------------------------

@dataclass
class ChurnRequest:
    """User-behaviour features for churn prediction.

    Attributes:
        total_flights: Total flights booked.
        total_hotel_bookings: Total hotel stays.
        total_spend: Cumulative spend across flights and hotels.
        avg_flight_price: Average flight ticket price.
        avg_hotel_price: Average hotel price.
        unique_destinations: Distinct cities visited.
        preferred_flight_type: Class of flight most taken.
        preferred_agency: Most used agency.
        age: User age.
        days_since_last_trip: Recency metric.
    """

    total_flights: int
    total_hotel_bookings: int
    total_spend: float
    avg_flight_price: float
    avg_hotel_price: float
    unique_destinations: int
    preferred_flight_type: str
    preferred_agency: str
    age: int
    days_since_last_trip: float

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.age <= 0 or self.age > 150:
            errors.append("age must be between 1 and 150")
        if self.total_flights < 0:
            errors.append("total_flights must be non-negative")
        if self.total_hotel_bookings < 0:
            errors.append("total_hotel_bookings must be non-negative")
        if self.total_spend < 0:
            errors.append("total_spend must be non-negative")
        if self.avg_flight_price < 0:
            errors.append("avg_flight_price must be non-negative")
        if self.avg_hotel_price < 0:
            errors.append("avg_hotel_price must be non-negative")
        if self.unique_destinations < 0:
            errors.append("unique_destinations must be non-negative")
        if self.days_since_last_trip < 0:
            errors.append("days_since_last_trip must be non-negative")
        return errors

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChurnRequest":
        try:
            return cls(
                total_flights=int(data["total_flights"]),
                total_hotel_bookings=int(data["total_hotel_bookings"]),
                total_spend=float(data["total_spend"]),
                avg_flight_price=float(data["avg_flight_price"]),
                avg_hotel_price=float(data["avg_hotel_price"]),
                unique_destinations=int(data["unique_destinations"]),
                preferred_flight_type=str(data["preferred_flight_type"]),
                preferred_agency=str(data["preferred_agency"]),
                age=int(data["age"]),
                days_since_last_trip=float(data["days_since_last_trip"]),
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"Invalid ChurnRequest payload: {exc}") from exc

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChurnResponse:
    """Churn prediction result with risk categorisation."""

    churn_probability: float
    risk_level: str  # "low" | "medium" | "high"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def classify_risk(probability: float) -> str:
        """Map a churn probability to a human-readable risk level."""
        if probability < 0.3:
            return "low"
        elif probability < 0.7:
            return "medium"
        return "high"


# ---------------------------------------------------------------------------
# Generic Prediction Wrapper
# ---------------------------------------------------------------------------

@dataclass
class PredictionResponse:
    """Standardised API response envelope.

    Every endpoint returns data wrapped in this structure so clients
    always see a consistent shape.
    """

    status: str  # "success" | "error"
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def success(cls, data: Dict[str, Any]) -> "PredictionResponse":
        return cls(status="success", data=data)

    @classmethod
    def error(cls, message: str) -> "PredictionResponse":
        return cls(status="error", message=message)
