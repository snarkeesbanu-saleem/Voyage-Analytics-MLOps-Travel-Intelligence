"""
data_ingestion.py
=================
Handles loading and initial validation of raw CSV datasets (users, flights, hotels)
for the Voyage Analytics MLOps pipeline.

Classes
-------
DataIngestion
    Configurable loader that reads CSVs, validates schemas, enforces dtypes,
    parses dates, and logs diagnostics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .utils import load_config, setup_logging, DataValidator, timer

logger = logging.getLogger(__name__)

# ── Expected schemas ──────────────────────────────────────────────────────────
_USERS_COLUMNS: List[str] = ["code", "company", "name", "gender", "age"]
_FLIGHTS_COLUMNS: List[str] = [
    "travelCode", "userCode", "from", "to",
    "flightType", "price", "time", "distance", "agency", "date",
]
_HOTELS_COLUMNS: List[str] = [
    "travelCode", "userCode", "name", "place",
    "days", "price", "total", "date",
]


class DataIngestion:
    """Load, validate, and return raw DataFrames from the Voyage Analytics CSVs.

    Parameters
    ----------
    config : dict
        Configuration dictionary.  Expected keys:
        - ``data_dir``  – path to the folder containing CSV files (default ``"data"``).
        - Any additional keys are stored but not required by this class.

    Attributes
    ----------
    data_dir : Path
        Resolved path to the data directory.
    config : dict
        Full configuration dictionary.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Dict[str, Any] = config
        self.data_dir: Path = Path(config.get("data_dir", "data")).resolve()
        logger.info("DataIngestion initialised – data_dir=%s", self.data_dir)

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _validate_schema(
        df: pd.DataFrame,
        expected_columns: List[str],
        dataset_name: str,
    ) -> None:
        """Raise ``ValueError`` if *df* is missing any expected columns.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to check.
        expected_columns : list[str]
            Column names that **must** be present.
        dataset_name : str
            Human-readable name used in error / log messages.

        Raises
        ------
        ValueError
            If one or more expected columns are absent.
        """
        missing = set(expected_columns) - set(df.columns)
        if missing:
            msg = (
                f"Schema validation failed for '{dataset_name}': "
                f"missing columns {sorted(missing)}"
            )
            logger.error(msg)
            raise ValueError(msg)
        logger.info(
            "Schema OK for '%s' – found %d columns.", dataset_name, len(df.columns)
        )

    @staticmethod
    def _log_summary(df: pd.DataFrame, dataset_name: str) -> None:
        """Log shape and quick descriptive stats for *df*."""
        logger.info(
            "'%s' shape: %s  |  dtypes: %s",
            dataset_name,
            df.shape,
            dict(df.dtypes),
        )
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            stats = df[numeric_cols].describe().loc[["mean", "std", "min", "max"]]
            logger.debug("'%s' numeric stats:\n%s", dataset_name, stats.to_string())

    # ── Public loaders ────────────────────────────────────────────────────────

    @timer
    def load_users(self) -> pd.DataFrame:
        """Load ``users.csv`` and return a validated DataFrame.

        Returns
        -------
        pd.DataFrame
            Users data with enforced dtypes:
            - ``code`` → ``int``
            - ``age``  → ``int`` (or ``float`` if NaN present)
            - ``gender`` → ``str``

        Raises
        ------
        FileNotFoundError
            If ``users.csv`` does not exist at the configured path.
        ValueError
            If required columns are missing.
        """
        filepath = self.data_dir / "users.csv"
        logger.info("Loading users from %s", filepath)

        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            logger.error("File not found: %s", filepath)
            raise
        except Exception as exc:
            logger.error("Unexpected error reading %s: %s", filepath, exc)
            raise

        self._validate_schema(df, _USERS_COLUMNS, "users")

        # Enforce dtypes
        try:
            df["code"] = pd.to_numeric(df["code"], errors="coerce").astype("Int64")
            df["age"] = pd.to_numeric(df["age"], errors="coerce").astype("Float64")
            df["gender"] = df["gender"].astype(str)
        except Exception as exc:
            logger.warning("Dtype enforcement issue in users: %s", exc)

        self._log_summary(df, "users")
        return df

    @timer
    def load_flights(self) -> pd.DataFrame:
        """Load ``flights.csv`` and return a validated DataFrame.

        Returns
        -------
        pd.DataFrame
            Flights data with parsed ``date`` column and numeric enforcement
            on ``price``, ``time``, and ``distance``.

        Raises
        ------
        FileNotFoundError
            If ``flights.csv`` does not exist at the configured path.
        ValueError
            If required columns are missing.
        """
        filepath = self.data_dir / "flights.csv"
        logger.info("Loading flights from %s", filepath)

        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            logger.error("File not found: %s", filepath)
            raise
        except Exception as exc:
            logger.error("Unexpected error reading %s: %s", filepath, exc)
            raise

        self._validate_schema(df, _FLIGHTS_COLUMNS, "flights")

        # Parse dates (MM/DD/YYYY)
        try:
            df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
            logger.info("Parsed flights 'date' column successfully.")
        except Exception as exc:
            logger.warning("Date parsing issue in flights: %s", exc)

        # Enforce numeric dtypes
        for col in ("price", "time", "distance"):
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception as exc:
                logger.warning("Numeric coercion issue for flights.%s: %s", col, exc)

        self._log_summary(df, "flights")
        return df

    @timer
    def load_hotels(self) -> pd.DataFrame:
        """Load ``hotels.csv`` and return a validated DataFrame.

        Returns
        -------
        pd.DataFrame
            Hotels data with parsed ``date`` column and numeric enforcement
            on ``days``, ``price``, and ``total``.

        Raises
        ------
        FileNotFoundError
            If ``hotels.csv`` does not exist at the configured path.
        ValueError
            If required columns are missing.
        """
        filepath = self.data_dir / "hotels.csv"
        logger.info("Loading hotels from %s", filepath)

        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            logger.error("File not found: %s", filepath)
            raise
        except Exception as exc:
            logger.error("Unexpected error reading %s: %s", filepath, exc)
            raise

        self._validate_schema(df, _HOTELS_COLUMNS, "hotels")

        # Parse dates (MM/DD/YYYY)
        try:
            df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
            logger.info("Parsed hotels 'date' column successfully.")
        except Exception as exc:
            logger.warning("Date parsing issue in hotels: %s", exc)

        # Enforce numeric dtypes
        for col in ("days", "price", "total"):
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception as exc:
                logger.warning("Numeric coercion issue for hotels.%s: %s", col, exc)

        self._log_summary(df, "hotels")
        return df

    @timer
    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all three datasets in sequence.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            ``(users_df, flights_df, hotels_df)``
        """
        logger.info("Loading all datasets …")
        users_df = self.load_users()
        flights_df = self.load_flights()
        hotels_df = self.load_hotels()
        logger.info(
            "All datasets loaded – users=%d, flights=%d, hotels=%d",
            len(users_df),
            len(flights_df),
            len(hotels_df),
        )
        return users_df, flights_df, hotels_df
