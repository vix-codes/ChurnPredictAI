"""
src/inference.py — Inference Engine for ChurnPredict AI
========================================================
Exposes a clean API for loading trained artifacts and making
single-row, batch, and streaming predictions with business-friendly
risk labels and recommendations.

Author  : ChurnPredict AI Engineering Team
Version : 1.0.0
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import paths
from config.config import DEFAULT_THRESHOLD, RISK_THRESHOLDS
from src.utils import (
    get_logger,
    timer,
    probability_to_risk_label,
    risk_label_to_color,
    generate_recommendation,
)
from src.preprocess import preprocess_input

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Artifact Loading
# ---------------------------------------------------------------------------
def load_model(model_path: Path | None = None) -> Any:
    """
    Load the trained model from disk.

    Parameters
    ----------
    model_path : Path | None  — defaults to models/random_forest.pkl

    Returns
    -------
    Fitted sklearn-compatible estimator
    """
    path = model_path or paths.MODEL_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}.\n"
            "Run: python src/pipeline.py  (or python src/train.py) first."
        )
    model = joblib.load(path)
    logger.info("Model loaded from: %s", path)
    return model


def load_encoder(encoder_path: Path | None = None) -> OneHotEncoder:
    """Load the fitted OneHotEncoder."""
    path = encoder_path or paths.ENCODER_FILE
    encoder = joblib.load(path)
    logger.info("Encoder loaded from: %s", path)
    return encoder


def load_scaler(scaler_path: Path | None = None) -> StandardScaler:
    """Load the fitted StandardScaler."""
    path = scaler_path or paths.SCALER_FILE
    scaler = joblib.load(path)
    logger.info("Scaler loaded from: %s", path)
    return scaler


def load_feature_columns(path: Path | None = None) -> list[str]:
    """Load the ordered feature column list."""
    p = path or paths.FEATURE_COLUMNS_FILE
    cols = joblib.load(p)
    logger.info("Feature columns loaded: %d columns", len(cols))
    return cols


def load_all_artifacts() -> tuple[Any, OneHotEncoder, StandardScaler, list[str]]:
    """
    Convenience function to load all inference artifacts in one call.

    Returns
    -------
    model, encoder, scaler, feature_columns
    """
    model = load_model()
    encoder = load_encoder()
    scaler = load_scaler()
    feature_columns = load_feature_columns()
    return model, encoder, scaler, feature_columns


# ---------------------------------------------------------------------------
# Input Validation
# ---------------------------------------------------------------------------
def validate_input(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Validate that *df* contains the minimum required columns for inference.

    Parameters
    ----------
    df : pd.DataFrame  — raw customer data

    Returns
    -------
    (is_valid, list_of_errors)
    """
    required_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Contract",
        "InternetService",
    ]
    errors: list[str] = []

    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing required column: '{col}'")

    if "tenure" in df.columns:
        if (df["tenure"] < 0).any():
            errors.append("'tenure' contains negative values")

    if "MonthlyCharges" in df.columns:
        if (df["MonthlyCharges"] <= 0).any():
            errors.append("'MonthlyCharges' must be > 0")

    return (len(errors) == 0), errors


# ---------------------------------------------------------------------------
# Core Prediction Functions
# ---------------------------------------------------------------------------
def predict_probability(
    input_df: pd.DataFrame,
    model: Any,
    encoder: OneHotEncoder,
    scaler: StandardScaler,
    feature_columns: list[str],
) -> np.ndarray:
    """
    Return churn probability for each row in *input_df*.

    Parameters
    ----------
    input_df        : pd.DataFrame  — raw (unprocessed) customer data
    model           : fitted estimator
    encoder         : fitted OneHotEncoder
    scaler          : fitted StandardScaler
    feature_columns : list[str]     — training column order

    Returns
    -------
    np.ndarray of shape (n_samples,)  — P(churn) for each row
    """
    is_valid, errors = validate_input(input_df)
    if not is_valid:
        raise ValueError("Input validation failed:\n" + "\n".join(errors))

    X = preprocess_input(input_df, encoder, scaler, feature_columns)
    proba = model.predict_proba(X)[:, 1]
    return proba


def predict(
    input_df: pd.DataFrame,
    model: Any,
    encoder: OneHotEncoder,
    scaler: StandardScaler,
    feature_columns: list[str],
    threshold: float = DEFAULT_THRESHOLD,
) -> np.ndarray:
    """
    Return binary churn predictions (0 = No Churn, 1 = Churn).

    Parameters
    ----------
    threshold : float  — decision threshold (default 0.5)

    Returns
    -------
    np.ndarray of shape (n_samples,)  — binary predictions
    """
    proba = predict_probability(input_df, model, encoder, scaler, feature_columns)
    return (proba >= threshold).astype(int)


def predict_batch(
    input_df: pd.DataFrame,
    model: Any,
    encoder: OneHotEncoder,
    scaler: StandardScaler,
    feature_columns: list[str],
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """
    Batch prediction with full business intelligence output.

    Returns a DataFrame with original columns plus:
    - ChurnProbability
    - ChurnPrediction
    - RiskLabel
    - RiskColor
    - Recommendation

    Parameters
    ----------
    input_df : pd.DataFrame  — one or more customer rows (raw format)
    """
    is_valid, errors = validate_input(input_df)
    if not is_valid:
        raise ValueError("Batch input validation failed:\n" + "\n".join(errors))

    proba = predict_probability(input_df, model, encoder, scaler, feature_columns)
    preds = (proba >= threshold).astype(int)

    results = input_df.copy()
    results["ChurnProbability"] = np.round(proba * 100, 2)
    results["ChurnPrediction"] = preds
    results["ChurnPrediction_Label"] = results["ChurnPrediction"].map(
        {0: "No Churn", 1: "Churn"}
    )
    results["RiskLabel"] = [probability_to_risk_label(p) for p in proba]
    results["RiskColor"] = [risk_label_to_color(lbl) for lbl in results["RiskLabel"]]
    results["Recommendation"] = [
        generate_recommendation(p, row)
        for p, row in zip(proba, input_df.to_dict("records"))
    ]

    logger.info(
        "Batch prediction: %d rows | Churn rate: %.2f%%",
        len(results),
        results["ChurnPrediction"].mean() * 100,
    )
    return results


# ---------------------------------------------------------------------------
# Single-Row Prediction (dashboard use)
# ---------------------------------------------------------------------------
def predict_single(
    customer_dict: dict,
    model: Any,
    encoder: OneHotEncoder,
    scaler: StandardScaler,
    feature_columns: list[str],
) -> dict:
    """
    Predict churn for a single customer supplied as a dictionary.

    Parameters
    ----------
    customer_dict : dict  — raw field values (matches raw dataset columns)

    Returns
    -------
    dict with keys:
        probability, risk_label, risk_color, churn_binary,
        recommendation, retention_impact_usd, estimated_savings_usd
    """
    df = pd.DataFrame([customer_dict])
    proba_arr = predict_probability(df, model, encoder, scaler, feature_columns)
    prob = float(proba_arr[0])

    risk_label = probability_to_risk_label(prob)
    color = risk_label_to_color(risk_label)
    recommendation = generate_recommendation(prob, customer_dict)

    # Business metrics
    monthly_charge = float(customer_dict.get("MonthlyCharges", 50))
    avg_tenure_remaining = max(0, 24 - int(customer_dict.get("tenure", 12)))
    retention_impact_usd = round(monthly_charge * avg_tenure_remaining, 2)
    estimated_savings = round(retention_impact_usd * 0.80, 2)

    return {
        "probability": round(prob, 4),
        "probability_pct": round(prob * 100, 2),
        "risk_label": risk_label,
        "risk_color": color,
        "churn_binary": int(prob >= DEFAULT_THRESHOLD),
        "recommendation": recommendation,
        "retention_impact_usd": retention_impact_usd,
        "estimated_savings_usd": estimated_savings,
        "avg_tenure_remaining_months": avg_tenure_remaining,
    }
