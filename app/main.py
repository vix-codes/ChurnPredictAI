"""
app/main.py — ChurnPredict AI Streamlit Dashboard (Enterprise Edition)
=======================================================================
The main entry point for the interactive business dashboard.

Tabs
----
• Dashboard       — Live What-If prediction + customer profiling
• Model Insights  — Performance metrics, ROC, feature importance
• Dataset Overview — EDA charts and statistics
• Business Insights — Domain-driven churn analysis cards

Usage
-----
    streamlit run app/main.py

Author  : ChurnPredict AI Engineering Team
Version : 1.0.0
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path for imports
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ---------------------------------------------------------------------------
# App-level imports
# ---------------------------------------------------------------------------
from app.utils import (
    configure_page,
    inject_css,
    load_ml_artifacts,
    load_raw_dataset,
    load_model_metrics,
    load_feature_importance,
    init_session_state,
    add_to_history,
    compute_business_impact,
    check_artifacts_exist,
)
from app.sidebar import render_sidebar
from app.metrics import (
    render_model_metrics,
    render_cv_metrics,
    render_dataset_kpis,
    render_prediction_card,
    render_business_insights,
    render_recommendation,
    render_prediction_history,
)
from app.visuals import (
    gauge_chart,
    feature_importance_chart,
    churn_pie_chart,
    tenure_histogram,
    monthly_charges_box,
    contract_bar_chart,
    correlation_heatmap,
    internet_service_pie,
    whatif_sensitivity_chart,
    model_comparison_chart,
    risk_distribution_chart,
)
from src.inference import predict_single, predict_batch
from config import paths

# ---------------------------------------------------------------------------
# Page Configuration (MUST be first Streamlit call)
# ---------------------------------------------------------------------------
configure_page()
inject_css()
init_session_state()


# ---------------------------------------------------------------------------
# Helper: Not-Ready Banner
# ---------------------------------------------------------------------------
def _show_not_ready_banner() -> None:
    st.error(
        "⚠️ **Model artifacts not found.**\n\n"
        "Please train the model first:\n"
        "```\n"
        "python src/pipeline.py\n"
        "```\n\n"
        "Or use the sample data generator:\n"
        "```\n"
        "python data/sample/generate_sample.py\n"
        "python src/pipeline.py --data data/sample/sample_telco_churn.csv\n"
        "```"
    )
    st.stop()


# ---------------------------------------------------------------------------
# Load Artifacts
# ---------------------------------------------------------------------------
if not check_artifacts_exist():
    _show_not_ready_banner()

model, encoder, scaler, feature_columns = load_ml_artifacts()
raw_df = load_raw_dataset()
model_metrics = load_model_metrics()
fi_df = load_feature_importance()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:16px; padding:8px 0 24px 0;">
        <div style="font-size:2.8rem;">🔮</div>
        <div>
            <div style="font-size:1.8rem; font-weight:800; color:#F1F5F9; line-height:1.1;">
                ChurnPredict AI
            </div>
            <div style="font-size:0.85rem; color:#64748B; letter-spacing:0.08em;">
                ENTERPRISE CHURN PREDICTION PLATFORM · v1.0
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
customer, predict_btn = render_sidebar()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_dash, tab_model, tab_data, tab_business = st.tabs([
    "🏠 Dashboard",
    "📊 Model Insights",
    "📦 Dataset Overview",
    "💡 Business Insights",
])


# ============================================================
# TAB 1 — DASHBOARD
# ============================================================
with tab_dash:

    # ── Auto-predict on sidebar change (What-If) ──────────────────────────
    if predict_btn or st.session_state.get("current_prediction") is not None:

        try:
            result = predict_single(
                customer, model, encoder, scaler, feature_columns
            )
            st.session_state["current_prediction"] = result
            st.session_state["current_probability"] = result["probability"]
            st.session_state["current_risk"] = result["risk_label"]

            if predict_btn:
                add_to_history(customer, result)

        except Exception as e:
            st.error(f"Prediction error: {e}")
            result = None
    else:
        result = None

    # ── Layout: two columns ───────────────────────────────────────────────
    col_left, col_right = st.columns([1.1, 1], gap="large")

    with col_left:
        st.markdown('<p class="section-header">🎯 Churn Prediction</p>',
                    unsafe_allow_html=True)

        if result:
            # Gauge
            prob = result["probability"]
            fig_gauge = gauge_chart(prob, result["risk_label"], result["risk_color"])
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Prediction Card
            render_prediction_card(result)
        else:
            st.markdown(
                """
                <div style="
                    background:rgba(30,41,59,0.5);
                    border:1px dashed #334155;
                    border-radius:16px;
                    padding:48px;
                    text-align:center;
                    margin-bottom:24px;
                ">
                    <div style="font-size:3rem; margin-bottom:12px;">🔮</div>
                    <div style="color:#94A3B8; font-size:1rem;">
                        Configure customer in the sidebar<br>and click <b>Predict Churn</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_right:
        st.markdown('<p class="section-header">👤 Customer Profile</p>',
                    unsafe_allow_html=True)

        # Profile Summary
        profile_data = {
            "📅 Tenure": f"{customer['tenure']} months",
            "💰 Monthly Charges": f"${customer['MonthlyCharges']:.2f}",
            "📋 Contract": customer["Contract"],
            "📡 Internet": customer["InternetService"],
            "💳 Payment": customer["PaymentMethod"],
            "🛡️ Security": customer["OnlineSecurity"],
            "🔧 Tech Support": customer["TechSupport"],
            "📺 Streaming TV": customer["StreamingTV"],
            "🎬 Streaming Movies": customer["StreamingMovies"],
            "👨‍👩‍👧 Partner": customer["Partner"],
            "🏠 Dependents": customer["Dependents"],
        }

        for label, value in profile_data.items():
            col_a, col_b = st.columns([1.5, 2])
            with col_a:
                st.markdown(
                    f'<div style="color:#64748B; font-size:0.82rem;">{label}</div>',
                    unsafe_allow_html=True,
                )
            with col_b:
                st.markdown(
                    f'<div style="color:#F1F5F9; font-size:0.82rem; font-weight:500;">{value}</div>',
                    unsafe_allow_html=True,
                )

        # What-If Analysis
        if result and model is not None:
            st.markdown("---")
            st.markdown("**📈 What-If: Monthly Charges Sensitivity**")
            charge_range = np.linspace(18.0, 120.0, 20)
            probs_whatif = []
            for charge in charge_range:
                test_customer = {**customer, "MonthlyCharges": float(charge)}
                try:
                    r = predict_single(test_customer, model, encoder, scaler, feature_columns)
                    probs_whatif.append(r["probability"])
                except Exception:
                    probs_whatif.append(0.0)

            fig_wi = whatif_sensitivity_chart(
                probs_whatif, charge_range.tolist(), "Monthly Charges ($)"
            )
            st.plotly_chart(fig_wi, use_container_width=True)

        # Recommendation
        if result:
            st.markdown("---")
            render_recommendation(result["recommendation"])

            # Business Impact
            impact = compute_business_impact(
                result["probability"],
                customer["MonthlyCharges"],
                customer["tenure"],
            )
            st.markdown("**📊 Business Impact**")
            b1, b2 = st.columns(2)
            with b1:
                st.metric("Revenue at Risk", f"${impact['revenue_at_risk']:,.0f}")
                st.metric("Retention Cost", f"${impact['retention_cost']:,.0f}")
            with b2:
                st.metric("Est. Savings", f"${impact['estimated_savings']:,.0f}")
                st.metric("ROI", f"{impact['roi_pct']:.0f}%")

    # ── Prediction History ────────────────────────────────────────────────
    if st.session_state["prediction_history"]:
        st.markdown("---")
        st.markdown('<p class="section-header">🕘 Prediction History</p>',
                    unsafe_allow_html=True)
        render_prediction_history(st.session_state["prediction_history"])

        # Download history
        hist_df = pd.DataFrame(st.session_state["prediction_history"])
        csv_hist = hist_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download History CSV",
            data=csv_hist,
            file_name="churn_prediction_history.csv",
            mime="text/csv",
        )

    # ── Batch Upload ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">📤 Batch Prediction Upload</p>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload a CSV file (same format as Telco dataset)",
        type=["csv"],
        key="batch_upload",
    )

    if uploaded:
        try:
            batch_input = pd.read_csv(uploaded)
            st.success(f"Loaded {len(batch_input):,} rows from '{uploaded.name}'")

            if st.button("🚀 Run Batch Prediction", type="primary"):
                with st.spinner("Running batch inference…"):
                    batch_results = predict_batch(
                        batch_input, model, encoder, scaler, feature_columns
                    )
                st.session_state["batch_results"] = batch_results
                st.success(f"✅ Batch complete — {len(batch_results):,} predictions")

        except Exception as e:
            st.error(f"Batch error: {e}")

    if st.session_state["batch_results"] is not None:
        br = st.session_state["batch_results"]
        st.dataframe(
            br[["customerID", "ChurnProbability", "ChurnPrediction_Label",
                "RiskLabel", "Recommendation"] if "customerID" in br.columns
               else br.head()].head(20),
            use_container_width=True,
        )

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "⬇️ Download Batch Results CSV",
                data=br.drop(columns=["RiskColor", "Recommendation"], errors="ignore")
                       .to_csv(index=False),
                file_name="batch_churn_predictions.csv",
                mime="text/csv",
            )
        with col_dl2:
            st.plotly_chart(
                risk_distribution_chart(br), use_container_width=True
            )


# ============================================================
# TAB 2 — MODEL INSIGHTS
# ============================================================
with tab_model:

    if model_metrics:
        render_model_metrics(model_metrics)
        st.markdown("---")

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            render_cv_metrics(model_metrics)

        with col_m2:
            # Model comparison
            comp = model_metrics.get("model_comparison_auc", {})
            if comp:
                st.markdown("**Model Comparison — ROC AUC**")
                st.plotly_chart(model_comparison_chart(comp), use_container_width=True)

        st.markdown("---")
    else:
        st.info("Run the training pipeline first to see model metrics.")

    # Feature Importance
    st.markdown('<p class="section-header">🏆 Feature Importance</p>',
                unsafe_allow_html=True)

    if fi_df is not None:
        n_top = st.slider("Top N Features", min_value=5, max_value=min(40, len(fi_df)),
                          value=15, step=1)
        fig_fi = feature_importance_chart(fi_df, top_n=n_top)
        st.plotly_chart(fig_fi, use_container_width=True)

        with st.expander("📋 Feature Importance Table"):
            st.dataframe(fi_df, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Download CSV",
                data=fi_df.to_csv(index=False),
                file_name="feature_importance.csv",
                mime="text/csv",
            )
    else:
        st.info("Feature importance will appear after training.")

    # Saved Plots
    st.markdown("---")
    st.markdown('<p class="section-header">📉 Evaluation Plots</p>',
                unsafe_allow_html=True)

    plot_cols = st.columns(3)
    plot_map = [
        (paths.ROC_CURVE_FILE,          "ROC Curve"),
        (paths.PR_CURVE_FILE,           "Precision-Recall Curve"),
        (paths.CONFUSION_MATRIX_FILE,   "Confusion Matrix"),
    ]
    for i, (plot_path, label) in enumerate(plot_map):
        with plot_cols[i % 3]:
            if plot_path.exists():
                st.image(str(plot_path), caption=label, use_container_width=True)
            else:
                st.info(f"{label} not yet generated.")

    # SHAP Plots
    shap_paths = [
        (paths.SHAP_SUMMARY_PLOT,   "SHAP Summary"),
        (paths.SHAP_WATERFALL_PLOT, "SHAP Waterfall"),
    ]
    shap_existing = [(p, l) for p, l in shap_paths if p.exists()]
    if shap_existing:
        st.markdown("**SHAP Explainability**")
        s_cols = st.columns(len(shap_existing))
        for i, (p, l) in enumerate(shap_existing):
            with s_cols[i]:
                st.image(str(p), caption=l, use_container_width=True)


# ============================================================
# TAB 3 — DATASET OVERVIEW
# ============================================================
with tab_data:

    if raw_df is not None:
        render_dataset_kpis(raw_df)
        st.markdown("---")

        row1c1, row1c2 = st.columns(2)
        with row1c1:
            st.plotly_chart(churn_pie_chart(raw_df), use_container_width=True)
        with row1c2:
            st.plotly_chart(internet_service_pie(raw_df), use_container_width=True)

        row2c1, row2c2 = st.columns(2)
        with row2c1:
            st.plotly_chart(tenure_histogram(raw_df), use_container_width=True)
        with row2c2:
            st.plotly_chart(monthly_charges_box(raw_df), use_container_width=True)

        st.plotly_chart(contract_bar_chart(raw_df), use_container_width=True)

        st.markdown("---")
        st.markdown('<p class="section-header">🔥 Correlation Heatmap</p>',
                    unsafe_allow_html=True)

        num_raw = raw_df.copy()
        # Encode churn for correlation
        if "Churn" in num_raw.columns and num_raw["Churn"].dtype == object:
            num_raw["Churn_binary"] = (num_raw["Churn"] == "Yes").astype(int)
        if "TotalCharges" in num_raw.columns:
            num_raw["TotalCharges"] = pd.to_numeric(num_raw["TotalCharges"], errors="coerce")

        st.plotly_chart(correlation_heatmap(num_raw), use_container_width=True)

        st.markdown("---")
        with st.expander("🔍 Raw Data Preview"):
            st.dataframe(raw_df.head(100), use_container_width=True)
            st.download_button(
                "⬇️ Download Raw Data",
                data=raw_df.to_csv(index=False),
                file_name="telco_churn_raw.csv",
                mime="text/csv",
            )
    else:
        st.info(
            "No dataset found. Place `telco_churn.csv` in `data/raw/` "
            "or run `python data/sample/generate_sample.py`."
        )


# ============================================================
# TAB 4 — BUSINESS INSIGHTS
# ============================================================
with tab_business:

    render_business_insights()

    st.markdown("---")

    # Churn Rate by Contract (if data available)
    if raw_df is not None and "Contract" in raw_df.columns and "Churn" in raw_df.columns:
        st.markdown('<p class="section-header">📊 Churn Rate Analysis</p>',
                    unsafe_allow_html=True)

        df_b = raw_df.copy()
        df_b["ChurnBinary"] = (df_b["Churn"] == "Yes").astype(int)

        # By Contract
        contract_churn = (
            df_b.groupby("Contract")["ChurnBinary"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "ChurnRate", "count": "Customers"})
        )
        contract_churn["ChurnRate_pct"] = contract_churn["ChurnRate"] * 100

        import plotly.express as px
        fig_cc = px.bar(
            contract_churn,
            x="Contract", y="ChurnRate_pct",
            color="ChurnRate_pct",
            color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
            text=contract_churn["ChurnRate_pct"].map(lambda x: f"{x:.1f}%"),
            title="Churn Rate by Contract Type",
        )
        fig_cc.update_traces(textposition="outside")
        fig_cc.update_layout(
            paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
            font=dict(color="#CBD5E1"), height=380,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_cc, use_container_width=True)

        # By Tenure Group
        if "tenure" in df_b.columns:
            bins = [0, 12, 24, 48, 60, 73]
            labels = ["0-12 mo", "13-24 mo", "25-48 mo", "49-60 mo", "60+ mo"]
            df_b["TenureGroup"] = pd.cut(df_b["tenure"], bins=bins, labels=labels, right=True)
            tenure_churn = (
                df_b.groupby("TenureGroup", observed=True)["ChurnBinary"]
                .mean()
                .reset_index()
                .rename(columns={"ChurnBinary": "ChurnRate"})
            )
            tenure_churn["ChurnRate_pct"] = tenure_churn["ChurnRate"] * 100

            fig_tc = px.line(
                tenure_churn,
                x="TenureGroup", y="ChurnRate_pct",
                markers=True,
                title="Churn Rate by Tenure Group",
            )
            fig_tc.update_traces(
                line=dict(color="#3B82F6", width=2.5),
                marker=dict(color="#8B5CF6", size=10),
            )
            fig_tc.update_layout(
                paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
                font=dict(color="#CBD5E1"), height=360,
                xaxis=dict(title="Tenure Group", gridcolor="#334155"),
                yaxis=dict(title="Churn Rate (%)", gridcolor="#334155"),
            )
            st.plotly_chart(fig_tc, use_container_width=True)

    st.markdown("---")
    st.markdown(
        """
        <div style="
            background:rgba(30,41,59,0.6);
            border:1px solid rgba(99,102,241,0.25);
            border-radius:12px;
            padding:24px;
            text-align:center;
        ">
            <div style="font-size:1.1rem; font-weight:700; color:#A5B4FC; margin-bottom:8px;">
                🎯 Strategic Retention Focus Areas
            </div>
            <div style="color:#CBD5E1; font-size:0.88rem; line-height:1.8;">
                1. Convert Month-to-Month customers to Annual contracts with 10-15% incentive<br>
                2. Proactively upsell TechSupport + OnlineSecurity to Fiber optic users<br>
                3. Launch loyalty programme for 12-24 month tenure customers<br>
                4. Target high-charge ($80+) month-to-month customers with bundle offers<br>
                5. New customers (&lt;6 months) need dedicated onboarding to reduce early churn
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
