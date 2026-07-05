"""
utils.py
========
Shared utility functions and classes for configuration, logging, serialisation,
timing, and data validation across the Voyage Analytics pipeline.
"""

from __future__ import annotations

import logging
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar

import joblib
import yaml

# ── Logging Setup ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


def setup_logging(
    level: str = "INFO",
    format_str: str = "%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    log_file: Optional[str] = None,
) -> None:
    """Configure the root logger to log to console and optionally to a file.

    Parameters
    ----------
    level : str
        Logging level (e.g. "INFO", "DEBUG", "WARNING").
    format_str : str
        Format string for log records.
    log_file : str | None
        Path to a file to log to.  Will be created if it does not exist.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file:
        log_path = Path(log_file).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=format_str,
        handlers=handlers,
        force=True,  # Reset existing configuration
    )
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logger.info("Logging configured.  Console output enabled. File logging: %s", log_file)


# ── Configuration Loader ──────────────────────────────────────────────────────


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    path = Path(config_path).resolve()
    if not path.exists():
        logger.warning("Config file not found at %s. Returning empty dict.", path)
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info("Loaded configuration from %s", path)
        return config or {}
    except Exception as exc:
        logger.error("Failed to parse config file %s: %s", path, exc)
        raise


# ── Model Serialization ───────────────────────────────────────────────────────


def save_model(model: Any, filepath: str) -> None:
    """Save an arbitrary python object (model, scaler, state) using joblib.

    Parameters
    ----------
    model : Any
        The object to serialise.
    filepath : str
        Destination path.
    """
    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(model, path)
        logger.info("Saved artifact to %s", path)
    except Exception as exc:
        logger.error("Failed to save artifact to %s: %s", path, exc)
        raise


def load_model(filepath: str) -> Any:
    """Load a joblib-serialised python object from disk.

    Parameters
    ----------
    filepath : str
        Path to the joblib file.

    Returns
    -------
    Any
        The loaded object.
    """
    path = Path(filepath).resolve()
    if not path.exists():
        logger.error("Artifact file not found at %s", path)
        raise FileNotFoundError(f"Artifact not found at {path}")
    try:
        obj = joblib.load(path)
        logger.info("Loaded artifact from %s", path)
        return obj
    except Exception as exc:
        logger.error("Failed to load artifact from %s: %s", path, exc)
        raise


# ── Performance / Timing Decorator ──────────────────────────────────────────


F = TypeVar("F", bound=Callable[..., Any])


def timer(func: F) -> F:
    """Decorator to measure and log the execution time of a function.

    Parameters
    ----------
    func : Callable
        The function to measure.

    Returns
    -------
    Callable
        The wrapped function.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start_time
            logger.info("Function '%s' executed in %.4f seconds", func.__name__, elapsed)

    return wrapper  # type: ignore


# ── Data Schema Validation ────────────────────────────────────────────────────


class DataValidator:
    """Utility class to validate pandas DataFrame structure and content."""

    @staticmethod
    def check_non_empty(df: Any, df_name: str = "DataFrame") -> None:
        """Verify that the DataFrame is not None and contains rows."""
        import pandas as pd
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{df_name} must be a pandas DataFrame, got {type(df)}")
        if df.empty:
            raise ValueError(f"{df_name} is empty")

    @staticmethod
    def check_missing_percentage(
        df: pd.DataFrame, threshold: float = 0.5
    ) -> Dict[str, float]:
        """Identify columns with missing value percentage higher than a threshold."""
        import pandas as pd
        ratios = df.isna().mean()
        problematic = {col: float(ratio) for col, ratio in ratios.items() if ratio > threshold}
        if problematic:
            logger.warning(
                "Columns with missing values exceeding %.1f%%: %s",
                threshold * 100,
                problematic,
            )
        return problematic
