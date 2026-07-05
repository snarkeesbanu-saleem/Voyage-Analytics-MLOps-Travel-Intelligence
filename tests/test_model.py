"""
test_model.py
=============
Unit tests for model training logic and saving/loading roundtrips.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
import pandas as pd
from src.model_training import ModelTrainer
from src.utils import load_model


def test_model_training_roundtrip(sample_config):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Override models dir to temporary directory
        config = sample_config.copy()
        config["paths"] = {"models_dir": tmpdir}
        
        trainer = ModelTrainer(config)
        
        # Flight price mock data
        X = pd.DataFrame({
            "flightType": [0, 1, 0, 2, 1],
            "time": [2.5, 1.2, 1.8, 3.0, 1.5],
            "distance": [1500.0, 600.0, 1000.0, 2000.0, 800.0],
            "agency": [0, 1, 2, 0, 1],
            "from": [1, 2, 1, 3, 2],
            "to": [2, 1, 3, 2, 1]
        })
        y = pd.Series([1200.0, 500.0, 800.0, 1500.0, 650.0])
        
        model, X_train, X_test, y_train, y_test = trainer.train_flight_price_model(X, y)
        
        # Verify save worked and can load
        model_file = Path(tmpdir) / "flight_price_model.joblib"
        assert model_file.exists()
        
        loaded_model = load_model(str(model_file))
        assert hasattr(loaded_model, "predict")
        
        preds = loaded_model.predict(X_test)
        assert len(preds) == len(X_test)
