"""
tests/test_inference.py — Unit Tests for the Inference Engine
=============================================================
Tests validate_input, probability_to_risk_label, predict_single output schema.

Author  : ChurnPredict AI Engineering Team
Version : 1.0.0
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import validate_input
from src.utils import probability_to_risk_label, risk_label_to_color, generate_recommendation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def valid_customer_df() -> pd.DataFrame:
    return pd.DataFrame([{
        "customerID": "T001",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 90.0,
        "TotalCharges": 450.0,
    }])


# ---------------------------------------------------------------------------
# validate_input
# ---------------------------------------------------------------------------
class TestValidateInput:
    def test_valid_input_passes(self, valid_customer_df):
        is_valid, errors = validate_input(valid_customer_df)
        assert is_valid
        assert len(errors) == 0

    def test_missing_column_fails(self, valid_customer_df):
        df = valid_customer_df.drop(columns=["tenure"])
        is_valid, errors = validate_input(df)
        assert not is_valid
        assert any("tenure" in e for e in errors)

    def test_negative_tenure_fails(self, valid_customer_df):
        df = valid_customer_df.copy()
        df["tenure"] = -5
        is_valid, errors = validate_input(df)
        assert not is_valid

    def test_zero_monthly_charge_fails(self, valid_customer_df):
        df = valid_customer_df.copy()
        df["MonthlyCharges"] = 0
        is_valid, errors = validate_input(df)
        assert not is_valid


# ---------------------------------------------------------------------------
# Risk Label Mapping
# ---------------------------------------------------------------------------
class TestProbabilityToRiskLabel:
    def test_low_risk(self):
        assert probability_to_risk_label(0.10) == "Low Risk"

    def test_medium_risk(self):
        assert probability_to_risk_label(0.35) == "Medium Risk"

    def test_high_risk(self):
        assert probability_to_risk_label(0.65) == "High Risk"

    def test_very_high_risk(self):
        assert probability_to_risk_label(0.90) == "Very High Risk"

    def test_boundary_zero(self):
        label = probability_to_risk_label(0.0)
        assert label == "Low Risk"

    def test_boundary_one(self):
        label = probability_to_risk_label(1.0)
        assert label == "Very High Risk"


# ---------------------------------------------------------------------------
# Risk Color
# ---------------------------------------------------------------------------
class TestRiskLabelToColor:
    def test_known_label(self):
        color = risk_label_to_color("Low Risk")
        assert color.startswith("#")

    def test_unknown_label_fallback(self):
        color = risk_label_to_color("Unknown Risk")
        assert color.startswith("#")


# ---------------------------------------------------------------------------
# Recommendation Generator
# ---------------------------------------------------------------------------
class TestGenerateRecommendation:
    def test_low_risk_no_action(self):
        rec = generate_recommendation(0.10)
        assert "loyalty" in rec.lower() or "standard" in rec.lower()

    def test_very_high_risk_escalation(self):
        rec = generate_recommendation(0.90)
        assert "critical" in rec.lower() or "escalate" in rec.lower()

    def test_month_to_month_contract_tip(self):
        row = {"Contract": "Month-to-month"}
        rec = generate_recommendation(0.60, row)
        assert "contract" in rec.lower()

    def test_no_tech_support_tip(self):
        row = {"TechSupport": "No"}
        rec = generate_recommendation(0.55, row)
        assert "techsupport" in rec.lower() or "support" in rec.lower()
