"""
test_data_ingestion.py
======================
Unit tests for raw data loading and schema validation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
import pandas as pd
from src.data_ingestion import DataIngestion


def test_data_ingestion_schema_validation(sample_config):
    # Create temporal dataset files to test loading
    with tempfile.TemporaryDirectory() as tmpdir:
        config = sample_config.copy()
        config["data_dir"] = tmpdir
        
        # 1. Invalid columns schema test
        df_invalid = pd.DataFrame({"bad_col": [1, 2, 3]})
        df_invalid.to_csv(Path(tmpdir) / "users.csv", index=False)
        
        ingestion = DataIngestion(config)
        with pytest.raises(ValueError, match="Schema validation failed"):
            ingestion.load_users()
            
        # 2. Valid users schema
        df_valid = pd.DataFrame({
            "code": [1, 2],
            "company": ["C1", "C2"],
            "name": ["N1", "N2"],
            "gender": ["male", "female"],
            "age": [20, 30]
        })
        df_valid.to_csv(Path(tmpdir) / "users.csv", index=False)
        users = ingestion.load_users()
        assert len(users) == 2
        assert list(users.columns) == ["code", "company", "name", "gender", "age"]
