"""
src/features.py — Business Feature Engineering for ChurnPredict AI
===================================================================
Adds meaningful, domain-driven features on top of the cleaned raw data.
All transformations are deterministic and reversible.

Author  : ChurnPredict AI Engineering Team
Version : 1.0.0
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.utils import get_logger, timer
from config.config import (
    HIGH_VALUE_MONTHLY_THRESHOLD,
    MIN_TENURE_LOYAL,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tenure Grouping
# ---------------------------------------------------------------------------
def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin *tenure* (months) into categorical loyalty tiers.

    Tiers
    -----
    New (0-12 m) | Growing (13-24 m) | Established (25-48 m) |
    Loyal (49-60 m) | Champion (60+ m)
    """
    bins = [0, 12, 24, 48, 60, float("inf")]
    labels = ["New", "Growing", "Established", "Loyal", "Champion"]
    df["TenureGroup"] = pd.cut(
        df["tenure"], bins=bins, labels=labels, right=True
    ).astype(str)
    return df


# ---------------------------------------------------------------------------
# Spend Category
# ---------------------------------------------------------------------------
def add_spend_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin *MonthlyCharges* into spend tiers.

    Tiers: Budget | Standard | Premium | Enterprise
    """
    bins = [0, 35, 65, 85, float("inf")]
    labels = ["Budget", "Standard", "Premium", "Enterprise"]
    df["SpendCategory"] = pd.cut(
        df["MonthlyCharges"], bins=bins, labels=labels, right=True
    ).astype(str)
    return df


# ---------------------------------------------------------------------------
# Monthly Charge Bucket (finer granularity)
# ---------------------------------------------------------------------------
def add_monthly_charge_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Split MonthlyCharges into 5 equal-width buckets."""
    df["MonthlyChargeBucket"] = pd.qcut(
        df["MonthlyCharges"],
        q=5,
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
        duplicates="drop",
    ).astype(str)
    return df


# ---------------------------------------------------------------------------
# Average Monthly Spend
# ---------------------------------------------------------------------------
def add_avg_monthly_spend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average monthly spend = TotalCharges / tenure.
    Uses MonthlyCharges as fallback for tenure == 0.
    """
    df["AvgMonthlySpend"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"],
    ).round(2)
    return df


# ---------------------------------------------------------------------------
# High-Value Customer Flag
# ---------------------------------------------------------------------------
def add_high_value_flag(df: pd.DataFrame) -> pd.DataFrame:
    """1 if MonthlyCharges ≥ HIGH_VALUE_MONTHLY_THRESHOLD, else 0."""
    df["HighValueCustomer"] = (
        df["MonthlyCharges"] >= HIGH_VALUE_MONTHLY_THRESHOLD
    ).astype(int)
    return df


# ---------------------------------------------------------------------------
# Contract Length (numeric encoding)
# ---------------------------------------------------------------------------
def add_contract_length(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map Contract text to numeric months.

    Month-to-month → 1 | One year → 12 | Two year → 24
    """
    mapping = {
        "Month-to-month": 1,
        "One year": 12,
        "Two year": 24,
    }
    df["ContractLength"] = df["Contract"].map(mapping).fillna(1).astype(int)
    return df


# ---------------------------------------------------------------------------
# Customer Lifetime Estimate
# ---------------------------------------------------------------------------
def add_customer_lifetime_estimate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate future lifetime revenue.

    Heuristic: AvgMonthlySpend × ContractLength × remaining_tenure_multiplier
    """
    multiplier = np.where(df["ContractLength"] > 1, 1.5, 1.0)
    df["CustomerLifetimeEstimate"] = (
        df["AvgMonthlySpend"] * df["ContractLength"] * multiplier
    ).round(2)
    return df


# ---------------------------------------------------------------------------
# Support Usage Score
# ---------------------------------------------------------------------------
def add_support_usage_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite score (0–3) for support / security service usage.

    + 1 if TechSupport == Yes
    + 1 if OnlineSecurity == Yes
    + 1 if DeviceProtection == Yes
    """
    score = pd.Series(0, index=df.index, dtype=int)
    for col in ["TechSupport", "OnlineSecurity", "DeviceProtection"]:
        if col in df.columns:
            score += df[col].isin(["Yes", 1]).astype(int)
    df["SupportUsageScore"] = score
    return df


# ---------------------------------------------------------------------------
# Streaming Usage Score
# ---------------------------------------------------------------------------
def add_streaming_usage_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite score (0–2) for streaming service adoption.

    + 1 if StreamingTV == Yes
    + 1 if StreamingMovies == Yes
    """
    score = pd.Series(0, index=df.index, dtype=int)
    for col in ["StreamingTV", "StreamingMovies"]:
        if col in df.columns:
            score += df[col].isin(["Yes", 1]).astype(int)
    df["StreamingUsageScore"] = score
    return df


# ---------------------------------------------------------------------------
# Internet Service Category (simplified)
# ---------------------------------------------------------------------------
def add_internet_service_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group InternetService into a simplified 3-tier category.

    Fiber → High-Speed | DSL → Standard | No → None
    """
    mapping = {
        "Fiber optic": "High-Speed",
        "DSL": "Standard",
        "No": "None",
    }
    df["InternetServiceCategory"] = df["InternetService"].map(mapping).fillna("None")
    return df


# ---------------------------------------------------------------------------
# Family Customer Flag
# ---------------------------------------------------------------------------
def add_family_customer_flag(df: pd.DataFrame) -> pd.DataFrame:
    """1 if customer has both Partner and Dependents, else 0."""
    partner = df["Partner"].isin(["Yes", 1]).astype(int)
    dependents = df["Dependents"].isin(["Yes", 1]).astype(int)
    df["FamilyCustomer"] = (partner & dependents).astype(int)
    return df


# ---------------------------------------------------------------------------
# Retention Risk Bucket
# ---------------------------------------------------------------------------
def add_retention_risk_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rule-based retention risk score (0-5) derived from domain knowledge.

    Points are added for each known risk factor.
    """
    risk = pd.Series(0, index=df.index, dtype=int)

    # Month-to-month contract
    if "Contract" in df.columns:
        risk += (df["Contract"] == "Month-to-month").astype(int)

    # No tech support
    if "TechSupport" in df.columns:
        risk += df["TechSupport"].isin(["No", 0]).astype(int)

    # No online security
    if "OnlineSecurity" in df.columns:
        risk += df["OnlineSecurity"].isin(["No", 0]).astype(int)

    # Short tenure (< 12 months)
    if "tenure" in df.columns:
        risk += (df["tenure"] < 12).astype(int)

    # High monthly charges (> 80)
    if "MonthlyCharges" in df.columns:
        risk += (df["MonthlyCharges"] > 80).astype(int)

    # Paperless billing without any service add-on
    if "PaperlessBilling" in df.columns:
        risk += df["PaperlessBilling"].isin(["Yes", 1]).astype(int)

    df["RetentionRiskScore"] = risk

    # Convert to named bucket
    bins = [-1, 1, 2, 3, 4, 6]
    labels = ["Minimal", "Low", "Moderate", "High", "Critical"]
    df["RetentionRiskBucket"] = pd.cut(
        risk, bins=bins, labels=labels
    ).astype(str)
    return df


# ---------------------------------------------------------------------------
# Interaction Features
# ---------------------------------------------------------------------------
def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create pairwise interaction features that capture compound risk signals.
    """
    # High charge AND month-to-month
    df["HighCharge_MTM"] = (
        (df["MonthlyCharges"] > 80) & (df["Contract"] == "Month-to-month")
    ).astype(int)

    # New customer AND no support
    df["NewNoSupport"] = (
        (df["tenure"] < 12) & (df["SupportUsageScore"] == 0)
    ).astype(int)

    # Fiber AND no security
    if "InternetService" in df.columns:
        df["FiberNoSecurity"] = (
            (df["InternetService"] == "Fiber optic")
            & df["OnlineSecurity"].isin(["No", 0])
        ).astype(int)

    # Monthly charges × tenure
    df["ChargeXTenure"] = (df["MonthlyCharges"] * df["tenure"]).round(2)

    return df


# ---------------------------------------------------------------------------
# Master Feature Engineering Function
# ---------------------------------------------------------------------------
@timer
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transforms to *df* in order.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned (but not yet encoded/scaled) customer DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with all engineered features appended.
    """
    logger.info("Starting feature engineering — input shape: %s", df.shape)

    df = add_tenure_group(df)
    df = add_spend_category(df)
    df = add_monthly_charge_bucket(df)
    df = add_avg_monthly_spend(df)
    df = add_high_value_flag(df)
    df = add_contract_length(df)
    df = add_customer_lifetime_estimate(df)
    df = add_support_usage_score(df)
    df = add_streaming_usage_score(df)
    df = add_internet_service_category(df)
    df = add_family_customer_flag(df)
    df = add_retention_risk_bucket(df)
    df = add_interaction_features(df)

    logger.info("Feature engineering complete — output shape: %s", df.shape)
    return df
