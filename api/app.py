"""
Voyage Analytics - Flask REST API
=================================
Production-grade REST API for serving flight-price, gender, and churn
predictions from pre-trained models persisted in the ``models/`` directory.

Endpoints
---------
POST /predict/flight-price   Flight price regression
POST /predict/gender          Gender classification
POST /predict/churn           Churn probability
GET  /health                  Liveness / readiness probe
GET  /model-info              Loaded model metadata
GET  /api/docs                Machine-readable endpoint catalogue
"""

from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request, Response
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Ensure project root is on ``sys.path`` so ``from src.xxx`` imports resolve
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.schemas import (
    ChurnRequest,
    ChurnResponse,
    FlightPriceRequest,
    FlightPriceResponse,
    GenderRequest,
    GenderResponse,
    PredictionResponse,
)
from pipelines.inference_pipeline import InferencePipeline

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "logs" / "api.log", mode="a"),
    ],
)
logger = logging.getLogger("voyage_analytics.api")

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(config: Optional[Dict[str, Any]] = None) -> Flask:
    """Create and configure the Flask application.

    Parameters
    ----------
    config : dict, optional
        Extra configuration overrides merged into ``app.config``.

    Returns
    -------
    Flask
        The configured application instance, ready to serve.
    """

    app = Flask(__name__)
    app.config.update(
        {
            "JSON_SORT_KEYS": False,
            "MODELS_DIR": str(PROJECT_ROOT / "models"),
            "MAX_CONTENT_LENGTH": 1 * 1024 * 1024,  # 1 MB request limit
        }
    )
    if config:
        app.config.update(config)

    # CORS – allow all origins in dev; restrict in production via env var
    allowed_origins = os.getenv("CORS_ORIGINS", "*")
    CORS(app, resources={r"/*": {"origins": allowed_origins}})

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    inference_pipeline: Optional[InferencePipeline] = None
    model_load_errors: Dict[str, str] = {}
    models_loaded_at: Optional[str] = None

    def _load_models() -> None:
        """Attempt to load all models from disk into the inference pipeline."""
        nonlocal inference_pipeline, model_load_errors, models_loaded_at
        try:
            inference_pipeline = InferencePipeline(
                models_dir=app.config["MODELS_DIR"]
            )
            model_load_errors = inference_pipeline.load_errors
            models_loaded_at = datetime.utcnow().isoformat()
            logger.info(
                "Models loaded successfully. Errors: %s",
                model_load_errors or "none",
            )
        except Exception as exc:
            logger.error("Critical failure loading models: %s", exc)
            model_load_errors = {"critical": str(exc)}

    with app.app_context():
        _load_models()

    # ------------------------------------------------------------------
    # Request / Response hooks
    # ------------------------------------------------------------------

    @app.before_request
    def _log_request() -> None:
        request._start_time = time.time()  # type: ignore[attr-defined]
        logger.info(
            "REQ  %s %s  from=%s",
            request.method,
            request.path,
            request.remote_addr,
        )

    @app.after_request
    def _log_response(response: Response) -> Response:
        elapsed = time.time() - getattr(request, "_start_time", time.time())
        logger.info(
            "RESP %s %s  status=%s  %.3fs",
            request.method,
            request.path,
            response.status_code,
            elapsed,
        )
        return response

    # ------------------------------------------------------------------
    # Error handlers
    # ------------------------------------------------------------------

    @app.errorhandler(400)
    def _bad_request(error: Any) -> tuple:
        return (
            jsonify(PredictionResponse.error(str(error)).to_dict()),
            400,
        )

    @app.errorhandler(404)
    def _not_found(error: Any) -> tuple:
        return (
            jsonify(
                PredictionResponse.error(
                    f"Endpoint not found: {request.path}"
                ).to_dict()
            ),
            404,
        )

    @app.errorhandler(405)
    def _method_not_allowed(error: Any) -> tuple:
        return (
            jsonify(
                PredictionResponse.error(
                    f"Method {request.method} not allowed on {request.path}"
                ).to_dict()
            ),
            405,
        )

    @app.errorhandler(500)
    def _internal_error(error: Any) -> tuple:
        logger.error("Unhandled 500: %s\n%s", error, traceback.format_exc())
        return (
            jsonify(
                PredictionResponse.error("Internal server error").to_dict()
            ),
            500,
        )

    # ------------------------------------------------------------------
    # Health & meta endpoints
    # ------------------------------------------------------------------

    @app.route("/health", methods=["GET"])
    def health() -> tuple:
        """Liveness / readiness probe with model status."""
        models_ready = inference_pipeline is not None and not model_load_errors
        status_code = 200 if models_ready else 503

        payload = {
            "status": "healthy" if models_ready else "degraded",
            "models_loaded": inference_pipeline is not None,
            "models_loaded_at": models_loaded_at,
            "model_errors": model_load_errors or None,
            "timestamp": datetime.utcnow().isoformat(),
        }
        return jsonify(payload), status_code

    @app.route("/model-info", methods=["GET"])
    def model_info() -> tuple:
        """Return metadata for every loaded model."""
        if inference_pipeline is None:
            return (
                jsonify(
                    PredictionResponse.error("Models not loaded").to_dict()
                ),
                503,
            )

        info = inference_pipeline.get_model_metadata()
        return (
            jsonify(
                PredictionResponse.success(
                    {
                        "models": info,
                        "models_dir": app.config["MODELS_DIR"],
                        "loaded_at": models_loaded_at,
                    }
                ).to_dict()
            ),
            200,
        )

    @app.route("/api/docs", methods=["GET"])
    def api_docs() -> tuple:
        """Machine-readable API documentation."""
        docs = {
            "service": "Voyage Analytics Prediction API",
            "version": "1.0.0",
            "endpoints": [
                {
                    "path": "/predict/flight-price",
                    "method": "POST",
                    "description": "Predict the price of a flight ticket",
                    "request_body": {
                        "from_city": "string  – departure city",
                        "to_city": "string  – arrival city",
                        "flight_type": "string  – economic | firstClass | premium",
                        "agency": "string  – booking agency name",
                        "distance": "float   – route distance (km)",
                        "time": "float   – flight duration (hours)",
                        "month": "int     – month (1-12)",
                        "day_of_week": "int     – day of week (0=Mon … 6=Sun)",
                    },
                    "response": {
                        "predicted_price": "float",
                        "confidence_interval": {"lower": "float", "upper": "float"},
                    },
                },
                {
                    "path": "/predict/gender",
                    "method": "POST",
                    "description": "Classify user gender from travel behaviour",
                    "request_body": {
                        "age": "int",
                        "num_flights": "int",
                        "total_spend": "float",
                        "avg_price": "float",
                        "unique_destinations": "int",
                        "num_stays": "int",
                        "hotel_spend": "float",
                    },
                    "response": {
                        "predicted_gender": "string – male | female",
                        "probability": "float",
                    },
                },
                {
                    "path": "/predict/churn",
                    "method": "POST",
                    "description": "Predict user churn probability",
                    "request_body": {
                        "age": "int",
                        "num_flights": "int",
                        "total_spend": "float",
                        "avg_price": "float",
                        "unique_destinations": "int",
                        "num_stays": "int",
                        "hotel_spend": "float",
                        "days_since_last_flight": "int",
                        "booking_frequency": "float",
                        "avg_trip_duration": "float",
                    },
                    "response": {
                        "churn_probability": "float",
                        "risk_level": "string – low | medium | high",
                    },
                },
                {
                    "path": "/health",
                    "method": "GET",
                    "description": "Service health check with model status",
                },
                {
                    "path": "/model-info",
                    "method": "GET",
                    "description": "Metadata for loaded models",
                },
                {
                    "path": "/api/docs",
                    "method": "GET",
                    "description": "This endpoint – API documentation",
                },
            ],
        }
        return jsonify(PredictionResponse.success(docs).to_dict()), 200

    # ------------------------------------------------------------------
    # Prediction endpoints
    # ------------------------------------------------------------------

    @app.route("/predict/flight-price", methods=["POST"])
    def predict_flight_price() -> tuple:
        """Predict flight ticket price."""
        if inference_pipeline is None:
            return (
                jsonify(
                    PredictionResponse.error("Models not loaded").to_dict()
                ),
                503,
            )

        payload = request.get_json(silent=True)
        if not payload:
            return (
                jsonify(
                    PredictionResponse.error(
                        "Request body must be valid JSON"
                    ).to_dict()
                ),
                400,
            )

        # Parse & validate
        try:
            req = FlightPriceRequest.from_dict(payload)
        except ValueError as exc:
            return (
                jsonify(PredictionResponse.error(str(exc)).to_dict()),
                400,
            )

        validation_errors = req.validate()
        if validation_errors:
            return (
                jsonify(
                    PredictionResponse.error(
                        f"Validation failed: {'; '.join(validation_errors)}"
                    ).to_dict()
                ),
                422,
            )

        # Predict
        try:
            result = inference_pipeline.predict_flight_price(req.to_dict())
            resp = FlightPriceResponse(
                predicted_price=round(result["predicted_price"], 2),
                confidence_interval=result.get(
                    "confidence_interval", {"lower": 0.0, "upper": 0.0}
                ),
            )
            return (
                jsonify(PredictionResponse.success(resp.to_dict()).to_dict()),
                200,
            )
        except Exception as exc:
            logger.exception("Flight-price prediction failed")
            return (
                jsonify(
                    PredictionResponse.error(
                        f"Prediction failed: {exc}"
                    ).to_dict()
                ),
                500,
            )

    @app.route("/predict/gender", methods=["POST"])
    def predict_gender() -> tuple:
        """Classify user gender from behavioural features."""
        if inference_pipeline is None:
            return (
                jsonify(
                    PredictionResponse.error("Models not loaded").to_dict()
                ),
                503,
            )

        payload = request.get_json(silent=True)
        if not payload:
            return (
                jsonify(
                    PredictionResponse.error(
                        "Request body must be valid JSON"
                    ).to_dict()
                ),
                400,
            )

        try:
            req = GenderRequest.from_dict(payload)
        except ValueError as exc:
            return (
                jsonify(PredictionResponse.error(str(exc)).to_dict()),
                400,
            )

        validation_errors = req.validate()
        if validation_errors:
            return (
                jsonify(
                    PredictionResponse.error(
                        f"Validation failed: {'; '.join(validation_errors)}"
                    ).to_dict()
                ),
                422,
            )

        try:
            result = inference_pipeline.predict_gender(req.to_dict())
            resp = GenderResponse(
                predicted_gender=result["predicted_gender"],
                probability=round(result["probability"], 4),
            )
            return (
                jsonify(PredictionResponse.success(resp.to_dict()).to_dict()),
                200,
            )
        except Exception as exc:
            logger.exception("Gender prediction failed")
            return (
                jsonify(
                    PredictionResponse.error(
                        f"Prediction failed: {exc}"
                    ).to_dict()
                ),
                500,
            )

    @app.route("/predict/churn", methods=["POST"])
    def predict_churn() -> tuple:
        """Predict user churn probability."""
        if inference_pipeline is None:
            return (
                jsonify(
                    PredictionResponse.error("Models not loaded").to_dict()
                ),
                503,
            )

        payload = request.get_json(silent=True)
        if not payload:
            return (
                jsonify(
                    PredictionResponse.error(
                        "Request body must be valid JSON"
                    ).to_dict()
                ),
                400,
            )

        try:
            req = ChurnRequest.from_dict(payload)
        except ValueError as exc:
            return (
                jsonify(PredictionResponse.error(str(exc)).to_dict()),
                400,
            )

        validation_errors = req.validate()
        if validation_errors:
            return (
                jsonify(
                    PredictionResponse.error(
                        f"Validation failed: {'; '.join(validation_errors)}"
                    ).to_dict()
                ),
                422,
            )

        try:
            result = inference_pipeline.predict_churn(req.to_dict())
            probability = result["churn_probability"]
            resp = ChurnResponse(
                churn_probability=round(probability, 4),
                risk_level=ChurnResponse.classify_risk(probability),
            )
            return (
                jsonify(PredictionResponse.success(resp.to_dict()).to_dict()),
                200,
            )
        except Exception as exc:
            logger.exception("Churn prediction failed")
            return (
                jsonify(
                    PredictionResponse.error(
                        f"Prediction failed: {exc}"
                    ).to_dict()
                ),
                500,
            )

    return app


# ---------------------------------------------------------------------------
# Development entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = create_app()
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    logger.info("Starting Voyage Analytics API on port %d (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)
