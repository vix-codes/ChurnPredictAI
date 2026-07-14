"""
tests/test_preprocess.py — Unit Tests for Data Preprocessing
=============================================================
Tests covering loading, cleaning, type conversion, and encoding steps.

Author  : ChurnPredict AI Engineering Team
Version : 1.0.0
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure root on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import clean_data, convert_types, encode_and_scale
from config.config import BINARY_YES_NO_COLS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def raw_sample() -> pd.DataFrame:
    """Minimal raw-format DataFrame mimicking IBM Telco schema."""
    return pd.DataFrame({
        "customerID": ["C001", "C002", "C003"],
        "gender":     ["Male", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0],
        "Partner":    ["Yes", "No", "Yes"],
        "Dependents": ["No", "No", "Yes"],
        "tenure":     [1, 24, 60],
        "PhoneService": ["Yes", "No", "Yes"],
        "MultipleLines":  ["No", "No phone service", "Yes"],
        "InternetService": ["Fiber optic", "DSL", "No"],
        "OnlineSecurity":  ["No", "Yes", "No internet service"],
        "OnlineBackup":    ["No", "Yes", "No internet service"],
        "DeviceProtection":["No", "Yes", "No internet service"],
        "TechSupport":     ["No", "No", "No internet service"],
        "StreamingTV":     ["Yes", "No", "No internet service"],
        "StreamingMovies": ["Yes", "No", "No internet service"],
        "Contract":        ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling":["Yes", "No", "No"],
        "PaymentMethod":   ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
        "MonthlyCharges":  [89.5, 35.2, 20.0],
        "TotalCharges":    ["89.5", " ", "1200.0"],
        "Churn":           ["Yes", "No", "No"],
    })


# ---------------------------------------------------------------------------
# Tests: clean_data
# ---------------------------------------------------------------------------
class TestCleanData:
    def test_removes_blank_total_charges(self, raw_sample):
        """TotalCharges blanks should be filled (not left as NaN)."""
        df = clean_data(raw_sample)
        assert df["TotalCharges"].isnull().sum() == 0

    def test_no_duplicates_introduced(self, raw_sample):
        dup = pd.concat([raw_sample, raw_sample.iloc[[0]]], ignore_index=True)
        df = clean_data(dup)
        assert len(df) == len(raw_sample)

    def test_strips_whitespace(self, raw_sample):
        raw_sample.loc[0, "Contract"] = "  Month-to-month  "
        df = clean_data(raw_sample)
        assert df.loc[0, "Contract"] == "Month-to-month"

    def test_total_charges_numeric(self, raw_sample):
        df = clean_data(raw_sample)
        assert pd.api.types.is_float_dtype(df["TotalCharges"])


# ---------------------------------------------------------------------------
# Tests: convert_types
# ---------------------------------------------------------------------------
class TestConvertTypes:
    def test_churn_binary(self, raw_sample):
        df = clean_data(raw_sample)
        df = convert_types(df)
        assert set(df["Churn"].unique()).issubset({0, 1})

    def test_partner_binary(self, raw_sample):
        df = clean_data(raw_sample)
        df = convert_types(df)
        assert set(df["Partner"].unique()).issubset({0, 1})

    def test_gender_binary(self, raw_sample):
        df = clean_data(raw_sample)
        df = convert_types(df)
        assert set(df["gender"].unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# Tests: encode_and_scale
# ---------------------------------------------------------------------------
class TestEncodeAndScale:
    def test_output_shape(self, raw_sample):
        from src.features import engineer_features

        df = clean_data(raw_sample)
        df = convert_types(df)
        df = engineer_features(df)
        df_features = df.drop(columns=["Churn", "customerID"], errors="ignore")

        result, feature_cols, encoder, scaler = encode_and_scale(df_features, fit=True)
        # Should have more columns than input (OHE expands categoricals)
        assert result.shape[0] == 3
        assert len(feature_cols) > 0

    def test_no_nan_in_output(self, raw_sample):
        from src.features import engineer_features

        df = clean_data(raw_sample)
        df = convert_types(df)
        df = engineer_features(df)
        df_features = df.drop(columns=["Churn", "customerID"], errors="ignore")

        result, _, _, _ = encode_and_scale(df_features, fit=True)
        assert result.isnull().sum().sum() == 0

    def test_inference_same_columns(self, raw_sample):
        """Re-encoding with fit=False must produce the same columns."""
        from src.features import engineer_features

        df = clean_data(raw_sample)
        df = convert_types(df)
        df = engineer_features(df)
        df_features = df.drop(columns=["Churn", "customerID"], errors="ignore")

        train_result, train_cols, encoder, scaler = encode_and_scale(df_features, fit=True)
        infer_result, infer_cols, _, _ = encode_and_scale(
            df_features, fit=False, encoder=encoder, scaler=scaler
        )
        assert train_cols == infer_cols
