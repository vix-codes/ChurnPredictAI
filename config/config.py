"""
config.py — Central Configuration for ChurnPredict AI
======================================================
All project-wide constants, model hyperparameters, column definitions,
and business thresholds are declared here. No magic numbers elsewhere.

Author  : ChurnPredict AI Engineering Team
Version : 1.0.0
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Project Metadata
# ---------------------------------------------------------------------------
PROJECT_NAME: str = "ChurnPredict AI"
VERSION: str = "1.0.0"
DESCRIPTION: str = "Enterprise-Grade Customer Churn Prediction Platform"

# ---------------------------------------------------------------------------
# Random State & Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
STRATIFY_COL: str = "Churn"

# ---------------------------------------------------------------------------
# Target Column
# ---------------------------------------------------------------------------
TARGET_COL: str = "Churn"
CUSTOMER_ID_COL: str = "customerID"

# ---------------------------------------------------------------------------
# Raw Dataset Columns
# ---------------------------------------------------------------------------
RAW_COLUMNS: list[str] = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

# ---------------------------------------------------------------------------
# Binary Yes/No Columns
# ---------------------------------------------------------------------------
BINARY_YES_NO_COLS: list[str] = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn",
]

# ---------------------------------------------------------------------------
# Columns to Drop Before Modelling
# ---------------------------------------------------------------------------
DROP_COLS: list[str] = ["customerID"]

# ---------------------------------------------------------------------------
# Categorical Columns for One-Hot Encoding
# ---------------------------------------------------------------------------
CATEGORICAL_COLS: list[str] = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
    # Engineered categoricals
    "TenureGroup",
    "SpendCategory",
    "MonthlyChargeBucket",
    "InternetServiceCategory",
    "RetentionRiskBucket",
]

# ---------------------------------------------------------------------------
# Numeric Columns for Standard Scaling
# ---------------------------------------------------------------------------
NUMERIC_COLS: list[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    # Engineered numerics
    "AvgMonthlySpend",
    "CustomerLifetimeEstimate",
    "SupportUsageScore",
    "StreamingUsageScore",
]

# ---------------------------------------------------------------------------
# Model Hyperparameters — Random Forest GridSearchCV
# ---------------------------------------------------------------------------
RF_PARAM_GRID: dict = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "class_weight": ["balanced", None],
}

RF_CV_FOLDS: int = 5
RF_SCORING: str = "roc_auc"
RF_N_JOBS: int = -1

# ---------------------------------------------------------------------------
# Model Hyperparameters — XGBoost (optional)
# ---------------------------------------------------------------------------
XGB_PARAM_GRID: dict = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "scale_pos_weight": [1, 2, 3],
}

# ---------------------------------------------------------------------------
# Model Hyperparameters — LightGBM (optional)
# ---------------------------------------------------------------------------
LGB_PARAM_GRID: dict = {
    "n_estimators": [100, 200, 300],
    "max_depth": [-1, 10, 20],
    "learning_rate": [0.01, 0.05, 0.1],
    "num_leaves": [31, 63, 127],
    "subsample": [0.7, 0.85, 1.0],
    "class_weight": ["balanced", None],
}

# ---------------------------------------------------------------------------
# Classification Threshold
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLD: float = 0.50

# ---------------------------------------------------------------------------
# Risk Tier Thresholds (probability → label)
# ---------------------------------------------------------------------------
RISK_THRESHOLDS: dict[str, float] = {
    "Low Risk": 0.25,
    "Medium Risk": 0.50,
    "High Risk": 0.75,
    "Very High Risk": 1.01,
}

RISK_COLORS: dict[str, str] = {
    "Low Risk": "#00C48C",
    "Medium Risk": "#FFB300",
    "High Risk": "#FF6B35",
    "Very High Risk": "#E53E3E",
}

# ---------------------------------------------------------------------------
# SHAP Configuration
# ---------------------------------------------------------------------------
SHAP_MAX_DISPLAY: int = 20
SHAP_PLOT_TYPE: str = "dot"

# ---------------------------------------------------------------------------
# Feature Importance — Top N
# ---------------------------------------------------------------------------
TOP_N_FEATURES: int = 20

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
)
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

# ---------------------------------------------------------------------------
# Streamlit Dashboard
# ---------------------------------------------------------------------------
STREAMLIT_THEME: dict = {
    "primaryColor": "#3B82F6",
    "backgroundColor": "#0F172A",
    "secondaryBackgroundColor": "#1E293B",
    "textColor": "#F1F5F9",
    "font": "sans serif",
}

PAGE_TITLE: str = "ChurnPredict AI | Enterprise Dashboard"
PAGE_ICON: str = "🔮"
LAYOUT: str = "wide"

# ---------------------------------------------------------------------------
# Business Constants
# ---------------------------------------------------------------------------
AVG_CUSTOMER_LTV: float = 1200.0          # Average lifetime value (USD)
CHURN_COST_MULTIPLIER: float = 5.0        # Cost to acquire vs retain
RETENTION_OFFER_DISCOUNT: float = 0.15   # 15% discount offered to at-risk customers
MIN_TENURE_LOYAL: int = 24                # Months — considered loyal customer
HIGH_VALUE_MONTHLY_THRESHOLD: float = 70.0  # USD — high-value customer
