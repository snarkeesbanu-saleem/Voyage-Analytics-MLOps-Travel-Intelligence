"""
data_preprocessing.py
=====================
Preprocessing pipeline for the Voyage Analytics MLOps project.

Provides a :class:`DataPreprocessor` that handles missing values, categorical
encoding (LabelEncoder), numerical normalisation (StandardScaler), and
IQR-based outlier removal.  Encoder / scaler state can be persisted to disk
with :pyfunc:`joblib` so the same transforms are reproducible at inference time.

Classes
-------
DataPreprocessor
    Stateful preprocessor that stores encoders and scalers for train → serve
    reproducibility.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .utils import save_model, load_model, timer

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Stateful data-preprocessing pipeline.

    Parameters
    ----------
    config : dict
        Configuration dictionary.  Recognised keys:
        - ``iqr_multiplier`` (float) – multiplier for IQR outlier detection
          (default ``1.5``).
        - Any extra keys are stored but not consumed here.

    Attributes
    ----------
    config : dict
        Full configuration dictionary.
    encoders : dict[str, LabelEncoder]
        Mapping of column name → fitted ``LabelEncoder``.
    scalers : dict[str, StandardScaler]
        Mapping of column name → fitted ``StandardScaler``.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Dict[str, Any] = config
        self.encoders: Dict[str, LabelEncoder] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        logger.info("DataPreprocessor initialised.")

    # ──────────────────────────────────────────────────────────────────────────
    # Missing values
    # ──────────────────────────────────────────────────────────────────────────

    @timer
    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = "median",
    ) -> pd.DataFrame:
        """Fill missing values in *df* using the chosen strategy.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame (a **copy** is returned; the original is not
            mutated).
        strategy : ``"median"`` | ``"mean"``
            Imputation strategy for numeric columns.  Categorical (object /
            category) columns are always filled with the mode.

        Returns
        -------
        pd.DataFrame
            DataFrame with missing values imputed.

        Raises
        ------
        ValueError
            If *strategy* is not ``"median"`` or ``"mean"``.
        """
        if strategy not in ("median", "mean"):
            raise ValueError(
                f"Invalid strategy '{strategy}' – choose 'median' or 'mean'."
            )

        df = df.copy()
        total_missing_before = int(df.isna().sum().sum())
        logger.info(
            "Handling missing values (strategy=%s) – total NaN before: %d",
            strategy,
            total_missing_before,
        )

        # Numeric columns
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        for col in numeric_cols:
            n_missing = int(df[col].isna().sum())
            if n_missing == 0:
                continue
            try:
                fill_value = (
                    df[col].median() if strategy == "median" else df[col].mean()
                )
                df[col] = df[col].fillna(fill_value)
                logger.debug(
                    "Filled %d NaN in numeric '%s' with %s=%s",
                    n_missing,
                    col,
                    strategy,
                    fill_value,
                )
            except Exception as exc:
                logger.warning("Could not fill '%s': %s", col, exc)

        # Categorical columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in cat_cols:
            n_missing = int(df[col].isna().sum())
            if n_missing == 0:
                continue
            try:
                mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else ""
                df[col] = df[col].fillna(mode_value)
                logger.debug(
                    "Filled %d NaN in categorical '%s' with mode='%s'",
                    n_missing,
                    col,
                    mode_value,
                )
            except Exception as exc:
                logger.warning("Could not fill '%s': %s", col, exc)

        total_missing_after = int(df.isna().sum().sum())
        logger.info(
            "Missing-value handling complete – NaN remaining: %d", total_missing_after
        )
        return df

    # ──────────────────────────────────────────────────────────────────────────
    # Categorical encoding
    # ──────────────────────────────────────────────────────────────────────────

    @timer
    def encode_categoricals(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Label-encode categorical columns and store fitted encoders.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame (a **copy** is returned).
        columns : list[str] | None
            Columns to encode.  Defaults to
            ``["gender", "flightType", "agency"]`` when *None*.

        Returns
        -------
        pd.DataFrame
            DataFrame with specified columns replaced by integer codes.

        Notes
        -----
        Fitted ``LabelEncoder`` instances are stored in ``self.encoders``
        keyed by column name, enabling inverse transforms and consistent
        encoding during inference via :meth:`transform`.
        """
        if columns is None:
            columns = ["gender", "flightType", "agency"]

        df = df.copy()
        logger.info("Encoding categoricals: %s", columns)

        for col in columns:
            if col not in df.columns:
                logger.warning(
                    "Column '%s' not found in DataFrame – skipping encoding.", col
                )
                continue
            try:
                le = LabelEncoder()
                # Handle potential NaN by converting to string first
                df[col] = df[col].astype(str)
                df[col] = le.fit_transform(df[col])
                self.encoders[col] = le
                logger.info(
                    "Encoded '%s' – %d unique classes: %s",
                    col,
                    len(le.classes_),
                    list(le.classes_),
                )
            except Exception as exc:
                logger.error("Failed to encode '%s': %s", col, exc)
                raise

        return df

    # ──────────────────────────────────────────────────────────────────────────
    # Numerical normalisation
    # ──────────────────────────────────────────────────────────────────────────

    @timer
    def normalize_numericals(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Standardise numeric columns to zero mean / unit variance.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame (a **copy** is returned).
        columns : list[str] | None
            Columns to normalise.  If *None*, all numeric columns are used.

        Returns
        -------
        pd.DataFrame
            DataFrame with normalised columns.

        Notes
        -----
        A single ``StandardScaler`` is fitted across *all* specified columns
        and stored in ``self.scalers["standard"]``.
        """
        df = df.copy()

        if columns is None:
            columns = df.select_dtypes(include="number").columns.tolist()

        # Filter to columns actually present
        present = [c for c in columns if c in df.columns]
        missing = set(columns) - set(present)
        if missing:
            logger.warning(
                "Columns not found for normalisation (skipped): %s", sorted(missing)
            )

        if not present:
            logger.warning("No columns to normalise – returning DataFrame unchanged.")
            return df

        logger.info("Normalising columns: %s", present)

        try:
            scaler = StandardScaler()
            df[present] = scaler.fit_transform(df[present])
            self.scalers["standard"] = scaler
            logger.info(
                "Normalisation complete – means ≈ %s, stds ≈ %s",
                np.round(scaler.mean_, 4).tolist(),
                np.round(scaler.scale_, 4).tolist(),
            )
        except Exception as exc:
            logger.error("Normalisation failed: %s", exc)
            raise

        return df

    # ──────────────────────────────────────────────────────────────────────────
    # Outlier removal
    # ──────────────────────────────────────────────────────────────────────────

    @timer
    def remove_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        multiplier: Optional[float] = None,
    ) -> pd.DataFrame:
        """Remove rows containing IQR-based outliers.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame (a **copy** is returned).
        columns : list[str] | None
            Numeric columns to check.  Defaults to all numeric columns.
        multiplier : float | None
            IQR multiplier.  Falls back to ``config["iqr_multiplier"]`` then
            ``1.5``.

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame without outlier rows.
        """
        df = df.copy()
        initial_len = len(df)

        if multiplier is None:
            multiplier = self.config.get("iqr_multiplier", 1.5)

        if columns is None:
            columns = df.select_dtypes(include="number").columns.tolist()

        logger.info(
            "Removing outliers (IQR × %.2f) on columns: %s", multiplier, columns
        )

        for col in columns:
            if col not in df.columns:
                logger.warning("Column '%s' not in DataFrame – skipping.", col)
                continue
            try:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - multiplier * iqr
                upper = q3 + multiplier * iqr
                before = len(df)
                df = df[(df[col] >= lower) & (df[col] <= upper)]
                removed = before - len(df)
                if removed:
                    logger.info(
                        "Column '%s': removed %d outliers (bounds [%.2f, %.2f]).",
                        col,
                        removed,
                        lower,
                        upper,
                    )
            except Exception as exc:
                logger.warning("Outlier removal failed for '%s': %s", col, exc)

        total_removed = initial_len - len(df)
        logger.info(
            "Outlier removal complete – %d / %d rows removed (%.1f%%).",
            total_removed,
            initial_len,
            100 * total_removed / max(initial_len, 1),
        )
        return df

    # ──────────────────────────────────────────────────────────────────────────
    # State persistence
    # ──────────────────────────────────────────────────────────────────────────

    def save_state(self, filepath: str) -> None:
        """Persist encoders and scalers to disk via ``joblib``.

        Parameters
        ----------
        filepath : str
            Destination path (e.g. ``"models/preprocessor_state.pkl"``).
        """
        state = {
            "encoders": self.encoders,
            "scalers": self.scalers,
        }
        try:
            joblib.dump(state, filepath)
            logger.info("Preprocessor state saved to %s", filepath)
        except Exception as exc:
            logger.error("Failed to save preprocessor state: %s", exc)
            raise

    def load_state(self, filepath: str) -> None:
        """Load previously saved encoders and scalers from disk.

        Parameters
        ----------
        filepath : str
            Path to a joblib-serialised state dict.
        """
        try:
            state = joblib.load(filepath)
            self.encoders = state.get("encoders", {})
            self.scalers = state.get("scalers", {})
            logger.info(
                "Preprocessor state loaded from %s – %d encoders, %d scalers.",
                filepath,
                len(self.encoders),
                len(self.scalers),
            )
        except Exception as exc:
            logger.error("Failed to load preprocessor state: %s", exc)
            raise

    # ──────────────────────────────────────────────────────────────────────────
    # Inference transform
    # ──────────────────────────────────────────────────────────────────────────

    @timer
    def transform(
        self,
        df: pd.DataFrame,
        columns_to_encode: Optional[List[str]] = None,
        columns_to_normalize: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Apply saved encoders / scalers to new data (inference mode).

        Uses the ``LabelEncoder`` and ``StandardScaler`` instances previously
        fitted (or loaded via :meth:`load_state`) to transform *df*
        consistently.

        Parameters
        ----------
        df : pd.DataFrame
            New / unseen data to transform (a **copy** is returned).
        columns_to_encode : list[str] | None
            Categorical columns to label-encode using stored encoders.
            Defaults to all keys in ``self.encoders``.
        columns_to_normalize : list[str] | None
            Numeric columns to normalise using the stored scaler.
            Defaults to the columns the scaler was originally fitted on (if
            available).

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame.

        Raises
        ------
        RuntimeError
            If no fitted encoders / scalers are available.
        """
        df = df.copy()

        # ── Encode categoricals ───────────────────────────────────────────
        if columns_to_encode is None:
            columns_to_encode = list(self.encoders.keys())

        for col in columns_to_encode:
            if col not in df.columns:
                logger.warning(
                    "Transform: column '%s' not in DataFrame – skipping.", col
                )
                continue
            encoder = self.encoders.get(col)
            if encoder is None:
                logger.warning(
                    "No fitted encoder for '%s' – skipping.", col
                )
                continue
            try:
                df[col] = df[col].astype(str)
                df[col] = encoder.transform(df[col])
                logger.debug("Transformed categorical '%s'.", col)
            except Exception as exc:
                logger.error(
                    "Transform failed for encoder '%s': %s", col, exc
                )
                raise

        # ── Normalise numericals ──────────────────────────────────────────
        scaler: Optional[StandardScaler] = self.scalers.get("standard")
        if scaler is not None:
            if columns_to_normalize is None:
                # Use the same columns the scaler was fitted on
                columns_to_normalize = list(
                    getattr(scaler, "feature_names_in_", [])
                )
            present = [c for c in columns_to_normalize if c in df.columns]
            if present:
                try:
                    df[present] = scaler.transform(df[present])
                    logger.debug("Normalised columns: %s", present)
                except Exception as exc:
                    logger.error("Scaler transform failed: %s", exc)
                    raise
            else:
                logger.warning(
                    "No matching columns for normalisation in incoming data."
                )
        else:
            if columns_to_normalize:
                logger.warning(
                    "No fitted scaler available – normalisation skipped."
                )

        logger.info("Inference transform complete – shape %s.", df.shape)
        return df
