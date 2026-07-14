"""
src/utils.py — Shared Utilities for ChurnPredict AI
====================================================
Logger factory, timing decorator, data validation helpers, and
miscellaneous utilities shared across every pipeline module.

Author  : ChurnPredict AI Engineering Team
Version : 1.0.0
"""

from __future__ import annotations

import functools
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import pandas as pd

from config.config import LOG_FORMAT, LOG_DATE_FORMAT, LOG_LEVEL
from config.paths import LOGS_DIR

# ---------------------------------------------------------------------------
# Ensure log directory exists
# ---------------------------------------------------------------------------
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Generic TypeVar for decorator
# ---------------------------------------------------------------------------
F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Logger Factory
# ---------------------------------------------------------------------------
def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    """
    Create and return a named logger.

    Parameters
    ----------
    name : str
        Logger name (typically ``__name__``).
    log_file : Path | None
        Optional path to a file handler. Defaults to ``logs/pipeline.log``.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Avoid adding duplicate handlers on repeated calls
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    log_path = log_file or (LOGS_DIR / "pipeline.log")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Timing Decorator
# ---------------------------------------------------------------------------
def timer(func: F) -> F:
    """
    Decorator that logs the execution time of any function.

    Usage
    -----
    @timer
    def my_function():
        ...
    """
    logger = get_logger(func.__module__ or __name__)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.info("▶  Started  : %s", func.__qualname__)
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info("✔  Completed: %s  (%.3f s)", func.__qualname__, elapsed)
            return result
        except Exception as exc:
            elapsed = time.perf_counter() - start
            logger.error(
                "✘  Failed   : %s  (%.3f s) — %s", func.__qualname__, elapsed, exc
            )
            raise

    return wrapper  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# JSON Utilities
# ---------------------------------------------------------------------------
def save_json(data: dict, path: Path) -> None:
    """Serialise *data* to JSON at *path*, handling numpy types."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=4, default=_default)


def load_json(path: Path) -> dict:
    """Load and return JSON from *path*."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# DataFrame Helpers
# ---------------------------------------------------------------------------
def memory_usage(df: pd.DataFrame) -> str:
    """Return human-readable memory usage of a DataFrame."""
    bytes_used = df.memory_usage(deep=True).sum()
    for unit in ("B", "KB", "MB", "GB"):
        if bytes_used < 1024:
            return f"{bytes_used:.2f} {unit}"
        bytes_used /= 1024
    return f"{bytes_used:.2f} TB"


def describe_dataframe(df: pd.DataFrame, logger: logging.Logger | None = None) -> None:
    """Log key DataFrame diagnostics."""
    log = logger or get_logger(__name__)
    log.info("Shape       : %s", df.shape)
    log.info("Memory      : %s", memory_usage(df))
    log.info("Dtypes      :\n%s", df.dtypes.value_counts().to_string())
    nulls = df.isnull().sum()
    if nulls.any():
        log.info("Null counts :\n%s", nulls[nulls > 0].to_string())


def validate_columns(df: pd.DataFrame, required: list[str], context: str = "") -> None:
    """
    Raise ``ValueError`` if any required column is missing from *df*.

    Parameters
    ----------
    df       : pd.DataFrame
    required : list[str]   — expected column names
    context  : str         — descriptive label for better error messages
    """
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"[{context}] Missing required columns: {sorted(missing)}"
        )


# ---------------------------------------------------------------------------
# Numeric Helpers
# ---------------------------------------------------------------------------
def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Return numerator / denominator, or *default* if denominator is zero."""
    return numerator / denominator if denominator != 0 else default


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* between *lo* and *hi* (inclusive)."""
    return max(lo, min(value, hi))


# ---------------------------------------------------------------------------
# Risk-Label Utility
# ---------------------------------------------------------------------------
def probability_to_risk_label(probability: float) -> str:
    """
    Map a churn probability in [0, 1] to a business risk label.

    Returns
    -------
    str — one of "Low Risk", "Medium Risk", "High Risk", "Very High Risk"
    """
    from config.config import RISK_THRESHOLDS  # local import to avoid circularity

    for label, upper_bound in RISK_THRESHOLDS.items():
        if probability < upper_bound:
            return label
    return "Very High Risk"


def risk_label_to_color(label: str) -> str:
    """Return the hex colour associated with a risk label."""
    from config.config import RISK_COLORS

    return RISK_COLORS.get(label, "#718096")


# ---------------------------------------------------------------------------
# Business Recommendation Generator
# ---------------------------------------------------------------------------
def generate_recommendation(probability: float, row: dict | None = None) -> str:
    """
    Generate a plain-English retention recommendation given a churn probability.

    Parameters
    ----------
    probability : float   — churn probability [0, 1]
    row         : dict    — optional feature dict for context-aware advice

    Returns
    -------
    str — business recommendation text
    """
    recommendations = []

    if probability < 0.25:
        recommendations.append(
            "✅ Customer shows strong loyalty signals. Continue standard engagement."
        )
    elif probability < 0.50:
        recommendations.append(
            "⚠️  Monitor closely. Offer a loyalty reward or check-in survey."
        )
    elif probability < 0.75:
        recommendations.append(
            "🚨 High churn risk. Proactively contact the customer with a personalised retention offer."
        )
    else:
        recommendations.append(
            "🔴 Critical churn risk. Escalate to retention team immediately. "
            "Consider contract upgrade incentive and complimentary service add-on."
        )

    if row:
        contract = row.get("Contract", "")
        if "Month" in str(contract):
            recommendations.append(
                "💡 Upgrade prompt: Offer a 1-year or 2-year contract with a 15% discount."
            )
        tech_support = row.get("TechSupport", "")
        if tech_support in ("No", 0):
            recommendations.append(
                "💡 Add TechSupport bundle — customers with support churn 25% less."
            )
        security = row.get("OnlineSecurity", "")
        if security in ("No", 0):
            recommendations.append(
                "💡 Offer Online Security package — strong retention driver."
            )

    return "  \n".join(recommendations)
