"""
app/utils.py — Dashboard Utility Layer for ChurnPredict AI
===========================================================
Caching wrappers, session-state helpers, and shared UI utilities
for the Streamlit dashboard.

Author  : ChurnPredict AI Engineering Team
Version : 1.0.0
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import json
import pandas as pd
import streamlit as st

from config import paths
from config.config import (
    PAGE_TITLE, PAGE_ICON, LAYOUT,
    AVG_CUSTOMER_LTV, CHURN_COST_MULTIPLIER, RETENTION_OFFER_DISCOUNT,
)


# ---------------------------------------------------------------------------
# Page Configuration (call ONCE at the top of main.py)
# ---------------------------------------------------------------------------
def configure_page() -> None:
    """Set global Streamlit page configuration."""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout=LAYOUT,
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.com/churnpredict-ai",
            "Report a bug": "https://github.com/churnpredict-ai/issues",
            "About": "ChurnPredict AI — Enterprise Churn Prediction Platform v1.0",
        },
    )


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
def inject_css() -> None:
    """Inject global dark-theme CSS into the Streamlit app."""
    st.markdown(
        """
        <style>
        /* ─── Import Google Font ──────────────────────── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* ─── Base Reset ─────────────────────────────── */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* ─── Main Background ────────────────────────── */
        .stApp {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
        }

        /* ─── Sidebar ────────────────────────────────── */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%);
            border-right: 1px solid #334155;
        }

        /* ─── Cards ──────────────────────────────────── */
        .metric-card {
            background: rgba(30, 41, 59, 0.8);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(59, 130, 246, 0.2);
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            margin-bottom: 12px;
        }
        .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 40px rgba(59, 130, 246, 0.15);
        }
        .metric-value {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #3B82F6, #8B5CF6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1.2;
        }
        .metric-label {
            font-size: 0.8rem;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-top: 6px;
            font-weight: 500;
        }

        /* ─── Risk Badges ────────────────────────────── */
        .risk-badge {
            display: inline-block;
            padding: 6px 18px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.85rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }
        .risk-low    { background: rgba(0, 196, 140, 0.18); color: #00C48C; border: 1px solid #00C48C; }
        .risk-medium { background: rgba(255, 179, 0, 0.18);  color: #FFB300; border: 1px solid #FFB300; }
        .risk-high   { background: rgba(255, 107, 53, 0.18); color: #FF6B35; border: 1px solid #FF6B35; }
        .risk-vhigh  { background: rgba(229, 62, 62, 0.18);  color: #E53E3E; border: 1px solid #E53E3E; }

        /* ─── Insight Cards ──────────────────────────── */
        .insight-card {
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid rgba(99, 102, 241, 0.25);
            border-radius: 12px;
            padding: 18px 22px;
            margin-bottom: 12px;
            border-left: 4px solid #6366F1;
        }
        .insight-card h4 {
            color: #A5B4FC;
            margin: 0 0 8px 0;
            font-size: 0.9rem;
            font-weight: 600;
        }
        .insight-card p {
            color: #CBD5E1;
            margin: 0;
            font-size: 0.85rem;
            line-height: 1.5;
        }

        /* ─── Section Headers ────────────────────────── */
        .section-header {
            font-size: 1.3rem;
            font-weight: 700;
            color: #F1F5F9;
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 2px solid rgba(59, 130, 246, 0.4);
        }

        /* ─── KPI Strip ──────────────────────────────── */
        .kpi-strip {
            background: rgba(30, 41, 59, 0.5);
            border-radius: 12px;
            padding: 16px 24px;
            border: 1px solid rgba(59, 130, 246, 0.15);
        }

        /* ─── Prediction Banner ──────────────────────── */
        .pred-banner {
            background: linear-gradient(135deg, rgba(59,130,246,0.15) 0%, rgba(139,92,246,0.15) 100%);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 20px;
            padding: 32px;
            text-align: center;
        }
        .pred-prob {
            font-size: 4rem;
            font-weight: 800;
            background: linear-gradient(135deg, #3B82F6, #8B5CF6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        /* ─── Streamlit overrides ─────────────────────── */
        .stButton > button {
            background: linear-gradient(135deg, #3B82F6, #6366F1);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
        }
        .stSelectbox [data-baseweb="select"],
        .stSlider {
            background: rgba(30, 41, 59, 0.8) !important;
        }
        div[data-testid="stMetricValue"] {
            color: #3B82F6 !important;
            font-weight: 700 !important;
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            color: #94A3B8;
            border: none;
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            color: #3B82F6 !important;
            border-bottom: 2px solid #3B82F6 !important;
        }
        footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Cached Data / Model Loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading ML artifacts…")
def load_ml_artifacts() -> tuple[Any, Any, Any, list[str]]:
    """
    Load and cache all ML artifacts (model, encoder, scaler, feature cols).

    Returns
    -------
    model, encoder, scaler, feature_columns
    """
    from src.inference import load_all_artifacts
    return load_all_artifacts()


@st.cache_data(show_spinner="Loading dataset…")
def load_processed_dataset() -> pd.DataFrame | None:
    """Load and cache the processed dataset from parquet."""
    if paths.PROCESSED_DATA_FILE.exists():
        return pd.read_parquet(paths.PROCESSED_DATA_FILE)
    if paths.PROCESSED_CSV_FILE.exists():
        return pd.read_csv(paths.PROCESSED_CSV_FILE)
    return None


@st.cache_data(show_spinner="Loading raw dataset…")
def load_raw_dataset() -> pd.DataFrame | None:
    """Load and cache the raw dataset."""
    if paths.RAW_DATA_FILE.exists():
        return pd.read_csv(paths.RAW_DATA_FILE)
    if paths.SAMPLE_DATA_FILE.exists():
        return pd.read_csv(paths.SAMPLE_DATA_FILE)
    return None


@st.cache_data
def load_model_metrics() -> dict | None:
    """Load and cache model performance metrics."""
    if paths.METRICS_FILE.exists():
        with open(paths.METRICS_FILE, "r") as f:
            return json.load(f)
    return None


@st.cache_data
def load_feature_importance() -> pd.DataFrame | None:
    """Load pre-computed feature importance CSV."""
    if paths.FEATURE_IMPORTANCE_CSV.exists():
        return pd.read_csv(paths.FEATURE_IMPORTANCE_CSV)
    return None


# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    """Initialise all required Streamlit session state keys."""
    defaults: dict[str, Any] = {
        "prediction_history": [],
        "current_prediction": None,
        "current_probability": None,
        "current_risk": None,
        "dark_mode": True,
        "threshold": 0.5,
        "batch_results": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_to_history(customer: dict, prediction: dict) -> None:
    """Append the latest prediction to session history."""
    record = {**customer, **prediction}
    st.session_state["prediction_history"].append(record)
    # Cap history at 50 entries
    if len(st.session_state["prediction_history"]) > 50:
        st.session_state["prediction_history"] = st.session_state["prediction_history"][-50:]


# ---------------------------------------------------------------------------
# Business Calculations
# ---------------------------------------------------------------------------
def compute_business_impact(
    probability: float,
    monthly_charge: float,
    tenure: int,
) -> dict:
    """
    Compute retention-related business impact metrics.

    Returns
    -------
    dict with revenue_at_risk, retention_cost, estimated_savings, roi
    """
    avg_remaining_tenure = max(0, 24 - tenure)
    revenue_at_risk = round(monthly_charge * avg_remaining_tenure * probability, 2)
    retention_cost = round(monthly_charge * RETENTION_OFFER_DISCOUNT * 12, 2)
    estimated_savings = round(revenue_at_risk - retention_cost, 2)
    roi = round(
        (estimated_savings / retention_cost * 100) if retention_cost > 0 else 0, 1
    )

    return {
        "revenue_at_risk": revenue_at_risk,
        "retention_cost": retention_cost,
        "estimated_savings": estimated_savings,
        "roi_pct": roi,
        "avg_remaining_tenure_months": avg_remaining_tenure,
    }


# ---------------------------------------------------------------------------
# Artifacts Status Check
# ---------------------------------------------------------------------------
def check_artifacts_exist() -> bool:
    """Return True if all required ML artifacts are present."""
    required = [
        paths.MODEL_FILE,
        paths.ENCODER_FILE,
        paths.SCALER_FILE,
        paths.FEATURE_COLUMNS_FILE,
    ]
    return all(p.exists() for p in required)
