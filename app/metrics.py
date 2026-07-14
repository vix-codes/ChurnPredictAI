"""
app/metrics.py — KPI & Metric Display Components for ChurnPredict AI
=====================================================================
Renders model performance KPIs, dataset statistics, and business metric
cards in a clean, reusable way.

Author  : ChurnPredict AI Engineering Team
Version : 1.0.0
"""

from __future__ import annotations

import streamlit as st
import pandas as pd


# ---------------------------------------------------------------------------
# Generic Metric Card (HTML)
# ---------------------------------------------------------------------------
def metric_card(label: str, value: str, delta: str | None = None, icon: str = "") -> str:
    """Return an HTML metric card string."""
    delta_html = (
        f'<div style="font-size:0.75rem; color:#94A3B8; margin-top:4px;">{delta}</div>'
        if delta else ""
    )
    return f"""
    <div class="metric-card">
        <div style="font-size:1.4rem; margin-bottom:4px;">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>
    """


# ---------------------------------------------------------------------------
# Model Performance Metrics Row
# ---------------------------------------------------------------------------
def render_model_metrics(metrics: dict) -> None:
    """
    Render a 5-column KPI strip for model performance.

    Parameters
    ----------
    metrics : dict  — loaded from outputs/metrics/model_metrics.json
    """
    test = metrics.get("test", {})

    st.markdown('<p class="section-header">📊 Model Performance</p>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    kpi_data = [
        (col1, "Accuracy",  f"{test.get('accuracy', 0):.2%}",   "Test Split", "🎯"),
        (col2, "Precision", f"{test.get('precision', 0):.2%}",  "Test Split", "⚡"),
        (col3, "Recall",    f"{test.get('recall', 0):.2%}",     "Test Split", "🔍"),
        (col4, "F1 Score",  f"{test.get('f1_score', 0):.2%}",   "Harmonic Mean", "📐"),
        (col5, "ROC AUC",   f"{test.get('roc_auc', 0):.4f}",   "Test Split", "📈"),
    ]

    for col, label, value, delta, icon in kpi_data:
        with col:
            st.markdown(metric_card(label, value, delta, icon), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cross-Validation Summary
# ---------------------------------------------------------------------------
def render_cv_metrics(metrics: dict) -> None:
    """Render cross-validation metric table."""
    cv = metrics.get("cross_validation", {})
    if not cv:
        return

    st.markdown("**5-Fold Cross-Validation Summary**")
    rows = []
    for metric, stats in cv.items():
        rows.append({
            "Metric": metric.replace("_", " ").title(),
            "Mean": f"{stats['mean']:.4f}",
            "Std Dev": f"± {stats['std']:.4f}",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Dataset KPI Cards
# ---------------------------------------------------------------------------
def render_dataset_kpis(df: pd.DataFrame) -> None:
    """
    Render dataset-level statistics (row count, churn rate, etc.).

    Parameters
    ----------
    df : pd.DataFrame  — raw or processed dataset
    """
    st.markdown('<p class="section-header">📦 Dataset Overview</p>', unsafe_allow_html=True)

    total = len(df)
    # Try to compute churn rate from raw or encoded target
    churn_rate = None
    if "Churn" in df.columns:
        col_vals = df["Churn"]
        if col_vals.dtype == object:
            churn_rate = (col_vals == "Yes").mean()
        else:
            churn_rate = col_vals.mean()

    cols = st.columns(4)
    with cols[0]:
        st.metric("Total Customers", f"{total:,}")
    with cols[1]:
        if churn_rate is not None:
            st.metric("Churn Rate", f"{churn_rate:.1%}")
        else:
            st.metric("Total Features", f"{df.shape[1]:,}")
    with cols[2]:
        st.metric("Total Features", f"{df.shape[1]:,}")
    with cols[3]:
        nulls = df.isnull().sum().sum()
        st.metric("Missing Values", f"{nulls:,}")


# ---------------------------------------------------------------------------
# Prediction Result Card
# ---------------------------------------------------------------------------
def render_prediction_card(result: dict) -> None:
    """
    Render the main prediction result with probability, risk badge,
    and business impact.

    Parameters
    ----------
    result : dict  — output from inference.predict_single()
    """
    prob_pct = result["probability_pct"]
    risk = result["risk_label"]
    color = result["risk_color"]
    churn = result["churn_binary"]

    # Map risk to CSS class
    css_map = {
        "Low Risk": "risk-low",
        "Medium Risk": "risk-medium",
        "High Risk": "risk-high",
        "Very High Risk": "risk-vhigh",
    }
    badge_class = css_map.get(risk, "risk-medium")

    verdict = "⚠️ LIKELY TO CHURN" if churn else "✅ LIKELY TO RETAIN"
    verdict_color = "#E53E3E" if churn else "#00C48C"

    st.markdown(
        f"""
        <div class="pred-banner">
            <div style="margin-bottom: 12px;">
                <span class="risk-badge {badge_class}">{risk}</span>
            </div>
            <div class="pred-prob">{prob_pct:.1f}%</div>
            <div style="font-size:0.85rem; color:#94A3B8; margin:4px 0 16px 0;">
                Churn Probability
            </div>
            <div style="font-size:1.1rem; font-weight:700; color:{verdict_color};">
                {verdict}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Business Impact Row
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "Revenue at Risk",
            f"${result.get('retention_impact_usd', 0):,.0f}",
            help="Estimated revenue lost if customer churns",
        )
    with c2:
        st.metric(
            "Est. Savings (Retention)",
            f"${result.get('estimated_savings_usd', 0):,.0f}",
            help="Estimated savings if retention action succeeds",
        )
    with c3:
        months = result.get("avg_tenure_remaining_months", 0)
        st.metric(
            "Months Remaining",
            f"{months} mo",
            help="Estimated remaining tenure if retained",
        )


# ---------------------------------------------------------------------------
# Business Insight Cards
# ---------------------------------------------------------------------------
BUSINESS_INSIGHTS = [
    {
        "icon": "📅",
        "title": "Month-to-Month = Higher Churn",
        "text": (
            "Customers on month-to-month contracts churn at 3× the rate "
            "of those on 2-year contracts. Offer contract upgrades proactively."
        ),
    },
    {
        "icon": "⏳",
        "title": "Loyalty Builds Retention",
        "text": (
            "Customers with 24+ months tenure have a churn rate under 10%. "
            "Invest in loyalty programmes during the first year."
        ),
    },
    {
        "icon": "💸",
        "title": "High Charges Drive Churn",
        "text": (
            "Monthly charges above $80 correlate with 25% higher churn probability. "
            "Bundling services can offset perceived cost."
        ),
    },
    {
        "icon": "🛡️",
        "title": "Tech Support Reduces Churn",
        "text": (
            "Customers without Tech Support churn 35% more frequently. "
            "Upsell support bundles to high-risk segments."
        ),
    },
    {
        "icon": "🔒",
        "title": "Online Security = Loyalty Signal",
        "text": (
            "Customers with Online Security have 28% lower churn probability. "
            "It's a high-impact add-on to offer at-risk customers."
        ),
    },
    {
        "icon": "👨‍👩‍👧",
        "title": "Family Customers Are Loyal",
        "text": (
            "Customers with both a partner and dependents show 40% lower churn. "
            "Tailor family bundle promotions to this segment."
        ),
    },
]


def render_business_insights() -> None:
    """Render all business insight cards in a 2-column grid."""
    st.markdown('<p class="section-header">💡 Business Insights</p>', unsafe_allow_html=True)

    cols = st.columns(2)
    for i, insight in enumerate(BUSINESS_INSIGHTS):
        with cols[i % 2]:
            st.markdown(
                f"""
                <div class="insight-card">
                    <h4>{insight['icon']} {insight['title']}</h4>
                    <p>{insight['text']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Recommendation Panel
# ---------------------------------------------------------------------------
def render_recommendation(recommendation: str) -> None:
    """Display the AI-generated retention recommendation."""
    st.markdown("**🤖 AI Retention Recommendation**")
    st.info(recommendation)


# ---------------------------------------------------------------------------
# Prediction History Table
# ---------------------------------------------------------------------------
def render_prediction_history(history: list[dict]) -> None:
    """Display the session prediction history as a formatted table."""
    if not history:
        st.info("No predictions made yet in this session.")
        return

    import pandas as pd

    rows = []
    for i, h in enumerate(history[::-1], 1):
        rows.append({
            "#": i,
            "Tenure": h.get("tenure", "—"),
            "Monthly $": f"${h.get('MonthlyCharges', 0):.0f}",
            "Contract": h.get("Contract", "—"),
            "Internet": h.get("InternetService", "—"),
            "Churn %": f"{h.get('probability_pct', 0):.1f}%",
            "Risk": h.get("risk_label", "—"),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
