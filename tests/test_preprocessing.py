"""
test_preprocessing.py
====================
Unit tests for the DataPreprocessor class including categorical encoding,
standard scaling, missing values handling, and outlier removal.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from src.data_preprocessing import DataPreprocessor


def test_preprocessing_flow(sample_config):
    df = pd.DataFrame({
        "gender": ["male", "female", np.nan, "male"],
        "price": [10.0, 20.0, 15.0, 1000.0],  # 1000 is an outlier
        "age": [20, 22, 21, 23]
    })
    
    preprocessor = DataPreprocessor(sample_config)
    
    # 1. Fill missing values
    df_imputed = preprocessor.handle_missing_values(df, strategy="median")
    assert df_imputed["gender"].isna().sum() == 0
    assert df_imputed["gender"].iloc[2] == "male"  # mode
    
    # 2. Encode categoricals
    df_encoded = preprocessor.encode_categoricals(df_imputed, columns=["gender"])
    assert df_encoded["gender"].dtype in (np.int32, np.int64, int)
    
    # 3. Normalise numericals
    df_norm = preprocessor.normalize_numericals(df_encoded, columns=["age"])
    assert np.allclose(df_norm["age"].mean(), 0.0, atol=1e-5)
    
    # 4. Outliers
    df_clean = preprocessor.remove_outliers(df_encoded, columns=["price"], multiplier=1.5)
    # price of 1000 should be removed
    assert len(df_clean) == 3
    assert 1000.0 not in df_clean["price"].values
