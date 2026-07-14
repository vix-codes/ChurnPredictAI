"""
tests/test_features.py — Unit Tests for Feature Engineering
============================================================
Tests all business feature engineering transforms.

Author  : ChurnPredict AI Engineering Team
Version : 1.0.0
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import (
    add_tenure_group,
    add_spend_category,
    add_avg_monthly_spend,
    add_high_value_flag,
    add_contract_length,
    add_support_usage_score,
    add_streaming_usage_score,
    add_family_customer_flag,
    add_retention_risk_bucket,
    add_interaction_features,
    engineer_features,
)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------
@pytest.fixture()
def base_df() -> pd.DataFrame:
    return pd.DataFrame({
        "tenure": [1, 12, 36, 60],
        "MonthlyCharges": [20.0, 50.0, 75.0, 95.0],
        "TotalCharges": [20.0, 600.0, 2700.0, 5700.0],
        "Partner": ["No", "Yes", "No", "Yes"],
        "Dependents": ["No", "No", "No", "Yes"],
        "Contract": ["Month-to-month", "One year", "Month-to-month", "Two year"],
        "InternetService": ["Fiber optic", "DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["No", "Yes", "No", "No internet service"],
        "DeviceProtection": ["No", "Yes", "No", "No internet service"],
        "TechSupport": ["No", "No", "Yes", "No internet service"],
        "StreamingTV": ["Yes", "No", "Yes", "No internet service"],
        "StreamingMovies": ["Yes", "No", "No", "No internet service"],
        "PaperlessBilling": ["Yes", "No", "Yes", "No"],
    })


# ---------------------------------------------------------------------------
# Tenure Group
# ---------------------------------------------------------------------------
class TestTenureGroup:
    def test_new_customer(self, base_df):
        df = add_tenure_group(base_df.copy())
        assert df.loc[0, "TenureGroup"] == "New"

    def test_champion_customer(self, base_df):
        df = add_tenure_group(base_df.copy())
        assert df.loc[3, "TenureGroup"] == "Champion"

    def test_no_nulls(self, base_df):
        df = add_tenure_group(base_df.copy())
        assert df["TenureGroup"].isnull().sum() == 0


# ---------------------------------------------------------------------------
# Spend Category
# ---------------------------------------------------------------------------
class TestSpendCategory:
    def test_budget_category(self, base_df):
        df = add_spend_category(base_df.copy())
        assert df.loc[0, "SpendCategory"] == "Budget"

    def test_enterprise_category(self, base_df):
        df = add_spend_category(base_df.copy())
        assert df.loc[3, "SpendCategory"] == "Enterprise"


# ---------------------------------------------------------------------------
# Avg Monthly Spend
# ---------------------------------------------------------------------------
class TestAvgMonthlySpend:
    def test_zero_tenure_fallback(self):
        df = pd.DataFrame({"tenure": [0], "TotalCharges": [0.0], "MonthlyCharges": [55.0]})
        df = add_avg_monthly_spend(df)
        assert df.loc[0, "AvgMonthlySpend"] == 55.0

    def test_normal_calculation(self, base_df):
        df = add_avg_monthly_spend(base_df.copy())
        expected = round(600.0 / 12, 2)
        assert df.loc[1, "AvgMonthlySpend"] == expected


# ---------------------------------------------------------------------------
# High Value Flag
# ---------------------------------------------------------------------------
class TestHighValueFlag:
    def test_high_value_true(self, base_df):
        df = add_high_value_flag(base_df.copy())
        assert df.loc[3, "HighValueCustomer"] == 1

    def test_not_high_value(self, base_df):
        df = add_high_value_flag(base_df.copy())
        assert df.loc[0, "HighValueCustomer"] == 0


# ---------------------------------------------------------------------------
# Contract Length
# ---------------------------------------------------------------------------
class TestContractLength:
    def test_month_to_month(self, base_df):
        df = add_contract_length(base_df.copy())
        assert df.loc[0, "ContractLength"] == 1

    def test_two_year(self, base_df):
        df = add_contract_length(base_df.copy())
        assert df.loc[3, "ContractLength"] == 24


# ---------------------------------------------------------------------------
# Support Usage Score
# ---------------------------------------------------------------------------
class TestSupportUsageScore:
    def test_max_score(self, base_df):
        df = add_support_usage_score(base_df.copy())
        # Row 1 has OnlineSecurity=Yes, DeviceProtection=Yes, TechSupport=No → 2
        assert df.loc[1, "SupportUsageScore"] == 2

    def test_zero_score(self, base_df):
        df = add_support_usage_score(base_df.copy())
        # Row 0 has all No → 0
        assert df.loc[0, "SupportUsageScore"] == 0


# ---------------------------------------------------------------------------
# Family Flag
# ---------------------------------------------------------------------------
class TestFamilyCustomerFlag:
    def test_family_flag_true(self, base_df):
        df = add_family_customer_flag(base_df.copy())
        # Row 3: Partner=Yes AND Dependents=Yes
        assert df.loc[3, "FamilyCustomer"] == 1

    def test_family_flag_false(self, base_df):
        df = add_family_customer_flag(base_df.copy())
        # Row 0: Partner=No, Dependents=No
        assert df.loc[0, "FamilyCustomer"] == 0


# ---------------------------------------------------------------------------
# Master engineer_features
# ---------------------------------------------------------------------------
class TestEngineerFeatures:
    def test_output_has_more_columns(self, base_df):
        original_cols = set(base_df.columns)
        result = engineer_features(base_df.copy())
        assert len(result.columns) > len(original_cols)

    def test_no_nulls(self, base_df):
        result = engineer_features(base_df.copy())
        assert result.isnull().sum().sum() == 0
