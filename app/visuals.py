"""
app/visuals.py — Plotly Visualisation Library for ChurnPredict AI
==================================================================
All interactive Plotly charts used by the dashboard.
Every chart follows the dark enterprise theme.

Author  : ChurnPredict AI Engineering Team
Version : 1.0.0
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.config import RISK_COLORS

# ---------------------------------------------------------------------------
# Shared Theme
# ---------------------------------------------------------------------------
THEME = dict(
    paper_bgcolor="#0F172A",
    plot_bgcolor="#1E293B",
    font=dict(family="Inter, sans-serif", color="#CBD5E1"),
    title_font=dict(color="#F1F5F9", size=15),
    colorway=["#3B82F6", "#8B5CF6", "#10B981", "#F59E0B", "#EF4444", "#EC4899"],
    margin=dict(l=40, r=20, t=50, b=40),
)

GRID_STYLE = dict(
    gridcolor="#334155",
    gridwidth=0.5,
    showgrid=True,
    zerolinecolor="#475569",
)


def _base_layout(**overrides) -> dict:
    layout = {**THEME, **overrides}
    return layout


# ---------------------------------------------------------------------------
# 1. Churn Probability Gauge
# ---------------------------------------------------------------------------
def gauge_chart(probability: float, risk_label: str, risk_color: str) -> go.Figure:
    """
    Render a half-donut gauge showing churn probability.

    Parameters
    ----------
    probability : float  — churn probability [0, 1]
    risk_label  : str
    risk_color  : str    — hex colour

    Returns
    -------
    go.Figure
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=probability * 100,
            delta={"reference": 50, "valueformat": ".1f", "suffix": "%"},
            title={"text": f"Churn Probability<br><b>{risk_label}</b>",
                   "font": {"size": 16, "color": "#F1F5F9"}},
            number={"suffix": "%", "font": {"size": 42, "color": risk_color}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#475569",
                    "tickfont": {"color": "#94A3B8"},
                },
                "bar": {"color": risk_color, "thickness": 0.3},
                "bgcolor": "#1E293B",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25],  "color": "rgba(0, 196, 140, 0.18)"},
                    {"range": [25, 50], "color": "rgba(255, 179, 0, 0.18)"},
                    {"range": [50, 75], "color": "rgba(255, 107, 53, 0.18)"},
                    {"range": [75, 100], "color": "rgba(229, 62, 62, 0.18)"},
                ],
                "threshold": {
                    "line": {"color": risk_color, "width": 3},
                    "thickness": 0.8,
                    "value": probability * 100,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="#0F172A",
        font=dict(family="Inter, sans-serif", color="#CBD5E1"),
        height=320,
        margin=dict(l=30, r=30, t=60, b=10),
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Feature Importance Bar Chart
# ---------------------------------------------------------------------------
def feature_importance_chart(
    fi_df: pd.DataFrame,
    top_n: int = 15,
) -> go.Figure:
    """
    Horizontal bar chart for feature importances.

    Parameters
    ----------
    fi_df : pd.DataFrame  — columns: feature, importance
    top_n : int
    """
    df = fi_df.head(top_n).copy().sort_values("importance")

    colors = [
        f"rgba(59, 130, 246, {0.4 + 0.6 * (i / len(df))})"
        for i in range(len(df))
    ]

    fig = go.Figure(
        go.Bar(
            x=df["importance"],
            y=df["feature"],
            orientation="h",
            marker=dict(color=colors, line=dict(color="#3B82F6", width=0.5)),
            text=[f"{v:.4f}" for v in df["importance"]],
            textposition="outside",
            textfont=dict(color="#94A3B8", size=10),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        **_base_layout(
            title=f"Top {top_n} Feature Importances",
            height=520,
            xaxis=dict(title="Gini Importance", **GRID_STYLE),
            yaxis=dict(title="", **GRID_STYLE, automargin=True),
        )
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Churn Distribution Pie
# ---------------------------------------------------------------------------
def churn_pie_chart(df: pd.DataFrame, target_col: str = "Churn") -> go.Figure:
    """Donut chart showing churn vs. non-churn distribution."""
    if target_col not in df.columns:
        return go.Figure()

    counts = df[target_col].value_counts()
    labels_map = {0: "No Churn", 1: "Churn", "No": "No Churn", "Yes": "Churn"}
    labels = [labels_map.get(k, str(k)) for k in counts.index]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=counts.values,
            hole=0.55,
            marker=dict(colors=["#10B981", "#EF4444"],
                        line=dict(color="#0F172A", width=2)),
            textinfo="label+percent",
            textfont=dict(size=13, color="#F1F5F9"),
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>",
        )
    )
    fig.update_layout(
        **_base_layout(title="Churn Distribution", height=360),
        showlegend=True,
        legend=dict(font=dict(color="#CBD5E1")),
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Tenure Distribution Histogram
# ---------------------------------------------------------------------------
def tenure_histogram(df: pd.DataFrame, churn_col: str = "Churn") -> go.Figure:
    """Overlaid histogram of tenure split by churn status."""
    fig = go.Figure()

    colors = {"No Churn": "#10B981", "Churn": "#EF4444"}

    if churn_col in df.columns:
        churn_vals = df[churn_col]
        if churn_vals.dtype == object:
            mapping = {"Yes": "Churn", "No": "No Churn"}
        else:
            mapping = {1: "Churn", 0: "No Churn"}

        for raw_val, label in mapping.items():
            mask = churn_vals == raw_val
            subset = df.loc[mask, "tenure"] if "tenure" in df.columns else pd.Series(dtype=float)
            fig.add_trace(
                go.Histogram(
                    x=subset,
                    name=label,
                    marker_color=colors[label],
                    opacity=0.75,
                    nbinsx=30,
                    hovertemplate=f"<b>{label}</b><br>Tenure: %{{x}}<br>Count: %{{y}}<extra></extra>",
                )
            )
    else:
        if "tenure" in df.columns:
            fig.add_trace(go.Histogram(x=df["tenure"], marker_color="#3B82F6", nbinsx=30))

    fig.update_layout(
        **_base_layout(
            title="Tenure Distribution by Churn Status",
            height=380,
            barmode="overlay",
            xaxis=dict(title="Tenure (Months)", **GRID_STYLE),
            yaxis=dict(title="Count", **GRID_STYLE),
        ),
        legend=dict(font=dict(color="#CBD5E1")),
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Monthly Charges Distribution
# ---------------------------------------------------------------------------
def monthly_charges_box(df: pd.DataFrame, churn_col: str = "Churn") -> go.Figure:
    """Box plot of MonthlyCharges segmented by churn."""
    if "MonthlyCharges" not in df.columns:
        return go.Figure()

    fig = go.Figure()
    if churn_col in df.columns:
        churn_vals = df[churn_col]
        mapping = (
            {"Yes": ("Churn", "#EF4444"), "No": ("No Churn", "#10B981")}
            if churn_vals.dtype == object
            else {1: ("Churn", "#EF4444"), 0: ("No Churn", "#10B981")}
        )
        for raw, (label, color) in mapping.items():
            subset = df.loc[churn_vals == raw, "MonthlyCharges"]
            fig.add_trace(
                go.Box(
                    y=subset,
                    name=label,
                    marker_color=color,
                    line_color=color,
                    boxmean=True,
                )
            )
    else:
        fig.add_trace(go.Box(y=df["MonthlyCharges"], marker_color="#3B82F6"))

    fig.update_layout(
        **_base_layout(
            title="Monthly Charges by Churn Status",
            height=400,
            yaxis=dict(title="Monthly Charges ($)", **GRID_STYLE),
            xaxis=dict(**GRID_STYLE),
        )
    )
    return fig


# ---------------------------------------------------------------------------
# 6. Contract Type Distribution
# ---------------------------------------------------------------------------
def contract_bar_chart(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of contract types vs churn."""
    if "Contract" not in df.columns:
        return go.Figure()

    churn_col = "Churn"
    if churn_col in df.columns:
        ct = df.groupby(["Contract", churn_col]).size().reset_index(name="Count")
        ct["ChurnLabel"] = ct[churn_col].map(
            {"Yes": "Churn", "No": "No Churn", 1: "Churn", 0: "No Churn"}
        )
        fig = px.bar(
            ct,
            x="Contract",
            y="Count",
            color="ChurnLabel",
            barmode="group",
            color_discrete_map={"Churn": "#EF4444", "No Churn": "#10B981"},
            title="Contract Type vs Churn",
        )
    else:
        ct = df["Contract"].value_counts().reset_index()
        ct.columns = ["Contract", "Count"]
        fig = px.bar(ct, x="Contract", y="Count", title="Contract Type Distribution")

    fig.update_layout(**_base_layout(height=400))
    return fig


# ---------------------------------------------------------------------------
# 7. Risk Distribution Bar (batch results)
# ---------------------------------------------------------------------------
def risk_distribution_chart(batch_df: pd.DataFrame) -> go.Figure:
    """Bar chart showing distribution of risk labels across a batch."""
    if "RiskLabel" not in batch_df.columns:
        return go.Figure()

    order = ["Low Risk", "Medium Risk", "High Risk", "Very High Risk"]
    counts = batch_df["RiskLabel"].value_counts().reindex(order, fill_value=0)
    colors = [RISK_COLORS.get(r, "#64748B") for r in order]

    fig = go.Figure(
        go.Bar(
            x=order,
            y=counts.values,
            marker=dict(color=colors),
            text=counts.values,
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        **_base_layout(
            title="Churn Risk Distribution",
            height=380,
            xaxis=dict(title="Risk Category", **GRID_STYLE),
            yaxis=dict(title="Customer Count", **GRID_STYLE),
        )
    )
    return fig


# ---------------------------------------------------------------------------
# 8. ROC-like Probability Scatter
# ---------------------------------------------------------------------------
def probability_distribution_chart(
    batch_df: pd.DataFrame,
    prob_col: str = "ChurnProbability",
) -> go.Figure:
    """Histogram of predicted churn probabilities coloured by risk tier."""
    if prob_col not in batch_df.columns:
        return go.Figure()

    fig = px.histogram(
        batch_df,
        x=prob_col,
        nbins=40,
        color="RiskLabel" if "RiskLabel" in batch_df.columns else None,
        color_discrete_map=RISK_COLORS,
        title="Predicted Churn Probability Distribution",
        labels={prob_col: "Churn Probability (%)"},
    )
    fig.update_layout(**_base_layout(height=380))
    return fig


# ---------------------------------------------------------------------------
# 9. Correlation Heatmap
# ---------------------------------------------------------------------------
def correlation_heatmap(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Heatmap of pairwise Pearson correlation for the top-N numeric columns.
    """
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        return go.Figure()

    # Pick top_n most-correlated (by abs variance) columns
    if len(num_df.columns) > top_n:
        variances = num_df.var().nlargest(top_n).index
        num_df = num_df[variances]

    corr = num_df.corr()

    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=9),
            hoverongaps=False,
        )
    )
    fig.update_layout(
        **_base_layout(
            title="Feature Correlation Heatmap",
            height=520,
            xaxis=dict(tickfont=dict(size=9)),
            yaxis=dict(tickfont=dict(size=9)),
        )
    )
    return fig


# ---------------------------------------------------------------------------
# 10. Internet Service Pie
# ---------------------------------------------------------------------------
def internet_service_pie(df: pd.DataFrame) -> go.Figure:
    """Donut chart of internet service type distribution."""
    if "InternetService" not in df.columns:
        return go.Figure()

    counts = df["InternetService"].value_counts()
    fig = go.Figure(
        go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.5,
            marker=dict(colors=["#3B82F6", "#8B5CF6", "#10B981"],
                        line=dict(color="#0F172A", width=2)),
            textinfo="label+percent",
            textfont=dict(size=12, color="#F1F5F9"),
        )
    )
    fig.update_layout(**_base_layout(title="Internet Service Distribution", height=340))
    return fig


# ---------------------------------------------------------------------------
# 11. What-If Sensitivity Chart
# ---------------------------------------------------------------------------
def whatif_sensitivity_chart(
    probabilities: list[float],
    param_values: list[float],
    param_name: str,
) -> go.Figure:
    """
    Line chart showing how churn probability changes as one parameter varies.

    Parameters
    ----------
    probabilities : list[float]  — predicted P(churn) at each param value
    param_values  : list[float]  — the swept parameter values
    param_name    : str          — axis label
    """
    fig = go.Figure(
        go.Scatter(
            x=param_values,
            y=[p * 100 for p in probabilities],
            mode="lines+markers",
            line=dict(color="#3B82F6", width=2.5),
            marker=dict(color="#8B5CF6", size=7),
            fill="tozeroy",
            fillcolor="rgba(59, 130, 246, 0.1)",
            hovertemplate=f"<b>{param_name}</b>: %{{x}}<br>Churn %: %{{y:.1f}}%<extra></extra>",
        )
    )
    fig.update_layout(
        **_base_layout(
            title=f"What-If Analysis: {param_name} vs Churn Probability",
            height=380,
            xaxis=dict(title=param_name, **GRID_STYLE),
            yaxis=dict(title="Churn Probability (%)", range=[0, 100], **GRID_STYLE),
        )
    )
    return fig


# ---------------------------------------------------------------------------
# 12. Model Comparison Bar
# ---------------------------------------------------------------------------
def model_comparison_chart(comparison: dict) -> go.Figure:
    """Bar chart comparing AUC scores across models."""
    if not comparison:
        return go.Figure()

    models = list(comparison.keys())
    aucs = list(comparison.values())
    colors = ["#3B82F6", "#8B5CF6", "#10B981", "#F59E0B"][: len(models)]

    fig = go.Figure(
        go.Bar(
            x=models,
            y=aucs,
            marker=dict(color=colors),
            text=[f"{a:.4f}" for a in aucs],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>ROC AUC: %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        **_base_layout(
            title="Model Comparison — ROC AUC",
            height=380,
            xaxis=dict(title="Model", **GRID_STYLE),
            yaxis=dict(title="ROC AUC", range=[0.5, 1.0], **GRID_STYLE),
        )
    )
    return fig
