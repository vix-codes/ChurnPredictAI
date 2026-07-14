"""
data/sample/generate_sample.py — Synthetic Telco Churn Dataset Generator
=========================================================================
Generates a realistic synthetic dataset matching the IBM Telco Customer
Churn schema. Use this when the real dataset is not available.

Usage
-----
    python data/sample/generate_sample.py
    python data/sample/generate_sample.py --rows 5000 --seed 42

Author  : ChurnPredict AI Engineering Team
Version : 1.0.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is importable
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.paths import SAMPLE_DATA_FILE, RAW_DATA_DIR


# ---------------------------------------------------------------------------
# Column option pools
# ---------------------------------------------------------------------------
GENDER_OPTS          = ["Male", "Female"]
YES_NO               = ["Yes", "No"]
INTERNET_OPTS        = ["DSL", "Fiber optic", "No"]
MULTI_LINE_OPTS      = ["Yes", "No", "No phone service"]
ADD_ON_OPTS          = ["Yes", "No", "No internet service"]
CONTRACT_OPTS        = ["Month-to-month", "One year", "Two year"]
PAYMENT_OPTS         = [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)",
]


# ---------------------------------------------------------------------------
# Synthetic Data Generator
# ---------------------------------------------------------------------------
def generate_telco_dataset(n_rows: int = 7043, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic Telco churn dataset.

    The churn probability is modelled realistically:
    - Month-to-month contracts increase churn
    - Higher tenure decreases churn
    - Fiber optic + no security increases churn
    - No tech support increases churn

    Parameters
    ----------
    n_rows : int  — number of customer records
    seed   : int  — random seed

    Returns
    -------
    pd.DataFrame — synthetic dataset with 21 columns
    """
    rng = np.random.default_rng(seed)

    # ── Basic Demographics ────────────────────────────────────────────────
    customer_ids  = [f"CUST-{i:06d}" for i in range(1, n_rows + 1)]
    gender        = rng.choice(GENDER_OPTS, n_rows)
    senior        = rng.choice([0, 1], n_rows, p=[0.84, 0.16])
    partner       = rng.choice(YES_NO, n_rows, p=[0.48, 0.52])
    dependents    = rng.choice(YES_NO, n_rows, p=[0.30, 0.70])
    tenure        = rng.integers(0, 73, n_rows)

    # ── Services ──────────────────────────────────────────────────────────
    phone_service = rng.choice(YES_NO, n_rows, p=[0.90, 0.10])
    multi_lines   = np.where(
        phone_service == "No",
        "No phone service",
        rng.choice(["Yes", "No"], n_rows, p=[0.42, 0.58]),
    )

    internet      = rng.choice(INTERNET_OPTS, n_rows, p=[0.34, 0.44, 0.22])

    def _add_on(internet_col: np.ndarray, yes_p: float = 0.35) -> np.ndarray:
        out = np.where(
            internet_col == "No",
            "No internet service",
            rng.choice(["Yes", "No"], n_rows, p=[yes_p, 1 - yes_p]),
        )
        return out

    security      = _add_on(internet, 0.30)
    backup        = _add_on(internet, 0.34)
    device_prot   = _add_on(internet, 0.34)
    tech_support  = _add_on(internet, 0.30)
    streaming_tv  = _add_on(internet, 0.38)
    streaming_mov = _add_on(internet, 0.39)

    # ── Contract / Billing ────────────────────────────────────────────────
    contract      = rng.choice(CONTRACT_OPTS, n_rows, p=[0.55, 0.21, 0.24])
    paperless     = rng.choice(YES_NO, n_rows, p=[0.59, 0.41])
    payment       = rng.choice(PAYMENT_OPTS, n_rows,
                               p=[0.34, 0.23, 0.22, 0.21])

    # ── Charges ───────────────────────────────────────────────────────────
    base_charge   = rng.uniform(18, 30, n_rows)
    phone_add     = np.where(phone_service == "Yes", rng.uniform(8, 12, n_rows), 0)
    internet_add  = np.where(
        internet == "Fiber optic", rng.uniform(35, 50, n_rows),
        np.where(internet == "DSL", rng.uniform(18, 30, n_rows), 0),
    )
    addon_charge  = (
        (security      == "Yes").astype(float) * rng.uniform(3, 7, n_rows) +
        (backup        == "Yes").astype(float) * rng.uniform(3, 7, n_rows) +
        (device_prot   == "Yes").astype(float) * rng.uniform(3, 7, n_rows) +
        (tech_support  == "Yes").astype(float) * rng.uniform(3, 7, n_rows) +
        (streaming_tv  == "Yes").astype(float) * rng.uniform(5, 8, n_rows) +
        (streaming_mov == "Yes").astype(float) * rng.uniform(5, 8, n_rows)
    )
    monthly_charges = np.round(base_charge + phone_add + internet_add + addon_charge, 2)
    total_charges   = np.round(monthly_charges * np.maximum(tenure, 1) * rng.uniform(0.95, 1.05, n_rows), 2)

    # Introduce a few blank TotalCharges (as in the real dataset)
    blank_idx = rng.choice(n_rows, size=11, replace=False)
    total_charges_str = total_charges.astype(str)
    total_charges_str[blank_idx] = " "

    # ── Churn Probability Model ───────────────────────────────────────────
    churn_score = np.zeros(n_rows, dtype=float)

    # Contract risk
    churn_score += np.where(contract == "Month-to-month", 0.35, 0.0)
    churn_score += np.where(contract == "One year", 0.08, 0.0)

    # Tenure (longer = lower churn)
    churn_score -= np.clip(tenure / 72, 0, 0.30)

    # Internet type
    churn_score += np.where(internet == "Fiber optic", 0.15, 0.0)

    # Missing security / tech support
    churn_score += np.where(security == "No", 0.08, 0.0)
    churn_score += np.where(tech_support == "No", 0.07, 0.0)

    # High monthly charges
    churn_score += np.clip((monthly_charges - 70) / 100, 0, 0.12)

    # Senior citizen
    churn_score += np.where(senior == 1, 0.05, 0.0)

    # Partner / dependents (reduce churn)
    churn_score -= np.where(partner == "Yes", 0.05, 0.0)
    churn_score -= np.where(dependents == "Yes", 0.04, 0.0)

    # Add noise
    churn_score += rng.normal(0, 0.08, n_rows)

    # Sigmoid-like probability
    churn_prob = 1 / (1 + np.exp(-churn_score * 4))
    churn_binary = (rng.random(n_rows) < churn_prob).astype(int)
    churn_label  = np.where(churn_binary == 1, "Yes", "No")

    # ── Assemble DataFrame ────────────────────────────────────────────────
    df = pd.DataFrame({
        "customerID":    customer_ids,
        "gender":        gender,
        "SeniorCitizen": senior,
        "Partner":       partner,
        "Dependents":    dependents,
        "tenure":        tenure,
        "PhoneService":  phone_service,
        "MultipleLines": multi_lines,
        "InternetService": internet,
        "OnlineSecurity":  security,
        "OnlineBackup":    backup,
        "DeviceProtection": device_prot,
        "TechSupport":     tech_support,
        "StreamingTV":     streaming_tv,
        "StreamingMovies": streaming_mov,
        "Contract":        contract,
        "PaperlessBilling": paperless,
        "PaymentMethod":    payment,
        "MonthlyCharges":   monthly_charges,
        "TotalCharges":     total_charges_str,
        "Churn":            churn_label,
    })

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic Telco churn data")
    parser.add_argument("--rows", type=int, default=7043, help="Number of rows (default: 7043)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Generating synthetic Telco dataset — {args.rows:,} rows …")
    df = generate_telco_dataset(n_rows=args.rows, seed=args.seed)

    # Save to sample dir
    SAMPLE_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAMPLE_DATA_FILE, index=False)
    print(f"Sample data saved -> {SAMPLE_DATA_FILE}")

    # Also copy to raw/ so the pipeline finds it automatically
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_target = RAW_DATA_DIR / "telco_churn.csv"
    df.to_csv(raw_target, index=False)
    print(f"Raw data copy     -> {raw_target}")

    churn_rate = (df["Churn"] == "Yes").mean()
    print(f"\nDataset Summary")
    print(f"  Rows       : {len(df):,}")
    print(f"  Columns    : {len(df.columns)}")
    print(f"  Churn Rate : {churn_rate:.1%}")
    print(f"\nNext step:")
    print(f"  python src/pipeline.py")
