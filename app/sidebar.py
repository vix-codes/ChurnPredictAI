"""
app/sidebar.py — Sidebar Input Panel for ChurnPredict AI Dashboard
==================================================================
Renders all customer input widgets and returns a structured
customer dictionary ready for inference.

Author  : ChurnPredict AI Engineering Team
Version : 1.0.0
"""

from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------------------------
# Example Customer Profiles
# ---------------------------------------------------------------------------
EXAMPLE_CUSTOMERS: dict[str, dict] = {
    "High Risk — Month-to-Month": {
        "customerID": "EX-001",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
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
        "MonthlyCharges": 89.50,
        "TotalCharges": 268.50,
    },
    "Low Risk — Long-Term": {
        "customerID": "EX-002",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 54,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 55.20,
        "TotalCharges": 2980.80,
    },
    "Medium Risk — DSL User": {
        "customerID": "EX-003",
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 14,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 42.30,
        "TotalCharges": 592.20,
    },
}


# ---------------------------------------------------------------------------
# Sidebar Renderer
# ---------------------------------------------------------------------------
def render_sidebar() -> dict:
    """
    Render all sidebar widgets and return a customer input dictionary.

    Returns
    -------
    dict  — customer field values matching the raw dataset schema
    """
    with st.sidebar:
        # ── Logo / Brand ──────────────────────────────────────────────────
        st.markdown(
            """
            <div style="text-align:center; padding: 16px 0 8px 0;">
                <div style="font-size:2.2rem;">🔮</div>
                <div style="font-size:1.1rem; font-weight:700; color:#F1F5F9;">
                    ChurnPredict AI
                </div>
                <div style="font-size:0.72rem; color:#64748B; letter-spacing:0.08em;">
                    ENTERPRISE EDITION
                </div>
            </div>
            <hr style="border-color:#334155; margin:12px 0 20px 0;">
            """,
            unsafe_allow_html=True,
        )

        # ── Quick-Fill Buttons ────────────────────────────────────────────
        st.markdown("**⚡ Quick Examples**")
        col_ex1, col_ex2 = st.columns(2)

        example_key = None
        with col_ex1:
            if st.button("🔴 High Risk", use_container_width=True, key="btn_high"):
                example_key = "High Risk — Month-to-Month"
        with col_ex2:
            if st.button("🟢 Low Risk", use_container_width=True, key="btn_low"):
                example_key = "Low Risk — Long-Term"

        col_ex3, col_ex4 = st.columns(2)
        with col_ex3:
            if st.button("🟡 Medium", use_container_width=True, key="btn_med"):
                example_key = "Medium Risk — DSL User"
        with col_ex4:
            if st.button("↺ Reset", use_container_width=True, key="btn_reset"):
                example_key = None
                for key in list(st.session_state.keys()):
                    if key.startswith("w_"):
                        del st.session_state[key]

        # Load example if button was clicked
        example = EXAMPLE_CUSTOMERS.get(example_key, {}) if example_key else {}

        st.markdown(
            "<hr style='border-color:#334155; margin:16px 0;'>",
            unsafe_allow_html=True,
        )

        # ── Section: Demographics ─────────────────────────────────────────
        st.markdown("**👤 Demographics**")

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"],
            index=0 if example.get("gender", "Male") == "Male" else 1,
            key="w_gender",
        )
        senior = st.selectbox(
            "Senior Citizen",
            ["No", "Yes"],
            index=example.get("SeniorCitizen", 0),
            key="w_senior",
        )
        partner = st.selectbox(
            "Partner",
            ["Yes", "No"],
            index=0 if example.get("Partner", "No") == "Yes" else 1,
            key="w_partner",
        )
        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"],
            index=0 if example.get("Dependents", "No") == "Yes" else 1,
            key="w_dep",
        )

        st.markdown(
            "<hr style='border-color:#334155; margin:16px 0;'>",
            unsafe_allow_html=True,
        )

        # ── Section: Account ──────────────────────────────────────────────
        st.markdown("**📋 Account Information**")

        tenure = st.slider(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=int(example.get("tenure", 12)),
            step=1,
            key="w_tenure",
        )
        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"],
            index=["Month-to-month", "One year", "Two year"].index(
                example.get("Contract", "Month-to-month")
            ),
            key="w_contract",
        )
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            index=[
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ].index(example.get("PaymentMethod", "Electronic check")),
            key="w_payment",
        )
        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"],
            index=0 if example.get("PaperlessBilling", "Yes") == "Yes" else 1,
            key="w_paperless",
        )

        st.markdown(
            "<hr style='border-color:#334155; margin:16px 0;'>",
            unsafe_allow_html=True,
        )

        # ── Section: Charges ──────────────────────────────────────────────
        st.markdown("**💰 Billing**")

        monthly_charges = st.slider(
            "Monthly Charges ($)",
            min_value=18.0,
            max_value=120.0,
            value=float(example.get("MonthlyCharges", 65.0)),
            step=0.5,
            format="$%.1f",
            key="w_monthly",
        )
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=10000.0,
            value=float(example.get("TotalCharges", monthly_charges * max(tenure, 1))),
            step=10.0,
            format="%.2f",
            key="w_total",
        )

        st.markdown(
            "<hr style='border-color:#334155; margin:16px 0;'>",
            unsafe_allow_html=True,
        )

        # ── Section: Services ─────────────────────────────────────────────
        st.markdown("**📡 Services**")

        phone = st.selectbox(
            "Phone Service",
            ["Yes", "No"],
            index=0 if example.get("PhoneService", "Yes") == "Yes" else 1,
            key="w_phone",
        )
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"],
            index=["No", "Yes", "No phone service"].index(
                example.get("MultipleLines", "No")
            ),
            key="w_multi",
        )
        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
            index=["DSL", "Fiber optic", "No"].index(
                example.get("InternetService", "Fiber optic")
            ),
            key="w_internet",
        )
        security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"],
            index=["No", "Yes", "No internet service"].index(
                example.get("OnlineSecurity", "No")
            ),
            key="w_security",
        )
        backup = st.selectbox(
            "Online Backup",
            ["No", "Yes", "No internet service"],
            index=["No", "Yes", "No internet service"].index(
                example.get("OnlineBackup", "No")
            ),
            key="w_backup",
        )
        device = st.selectbox(
            "Device Protection",
            ["No", "Yes", "No internet service"],
            index=["No", "Yes", "No internet service"].index(
                example.get("DeviceProtection", "No")
            ),
            key="w_device",
        )
        tech = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"],
            index=["No", "Yes", "No internet service"].index(
                example.get("TechSupport", "No")
            ),
            key="w_tech",
        )
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"],
            index=["No", "Yes", "No internet service"].index(
                example.get("StreamingTV", "No")
            ),
            key="w_stv",
        )
        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"],
            index=["No", "Yes", "No internet service"].index(
                example.get("StreamingMovies", "No")
            ),
            key="w_smovies",
        )

        st.markdown(
            "<hr style='border-color:#334155; margin:16px 0;'>",
            unsafe_allow_html=True,
        )

        # ── Predict Button ────────────────────────────────────────────────
        predict_btn = st.button(
            "🔮 Predict Churn",
            use_container_width=True,
            type="primary",
            key="btn_predict",
        )

        # ── Advanced Settings ─────────────────────────────────────────────
        with st.expander("⚙️ Advanced Settings"):
            threshold = st.slider(
                "Decision Threshold",
                min_value=0.10,
                max_value=0.90,
                value=float(st.session_state.get("threshold", 0.50)),
                step=0.05,
                format="%.2f",
                help="Probability threshold above which a customer is flagged as churner",
            )
            st.session_state["threshold"] = threshold

        st.markdown(
            "<div style='text-align:center; color:#475569; font-size:0.72rem; margin-top:24px;'>"
            "ChurnPredict AI v1.0 · Enterprise Edition"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Assemble customer dict ────────────────────────────────────────────
    customer = {
        "customerID": "LIVE-001",
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": device,
        "TechSupport": tech,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    return customer, predict_btn
