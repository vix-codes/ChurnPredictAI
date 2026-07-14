"""
paths.py — Centralised Path Management for ChurnPredict AI
===========================================================
All filesystem paths are resolved here using pathlib.Path.
No hardcoded path strings anywhere else in the project.

Author  : ChurnPredict AI Engineering Team
Version : 1.0.0
"""

from __future__ import annotations
from pathlib import Path

# ---------------------------------------------------------------------------
# Project Root  (two levels up: config/ → ChurnPredictAI/)
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Source & App
# ---------------------------------------------------------------------------
SRC_DIR: Path = ROOT_DIR / "src"
APP_DIR: Path = ROOT_DIR / "app"
CONFIG_DIR: Path = ROOT_DIR / "config"

# ---------------------------------------------------------------------------
# Data Directories
# ---------------------------------------------------------------------------
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
SAMPLE_DATA_DIR: Path = DATA_DIR / "sample"

# ---------------------------------------------------------------------------
# Specific Data Files
# ---------------------------------------------------------------------------
RAW_DATA_FILE: Path = RAW_DATA_DIR / "telco_churn.csv"
SAMPLE_DATA_FILE: Path = SAMPLE_DATA_DIR / "sample_telco_churn.csv"
PROCESSED_DATA_FILE: Path = PROCESSED_DATA_DIR / "processed_churn.parquet"
PROCESSED_CSV_FILE: Path = PROCESSED_DATA_DIR / "processed_churn.csv"

# ---------------------------------------------------------------------------
# Model Artifacts
# ---------------------------------------------------------------------------
MODELS_DIR: Path = ROOT_DIR / "models"
MODEL_FILE: Path = MODELS_DIR / "random_forest.pkl"
XGB_MODEL_FILE: Path = MODELS_DIR / "xgboost_model.pkl"
LGB_MODEL_FILE: Path = MODELS_DIR / "lightgbm_model.pkl"
SCALER_FILE: Path = MODELS_DIR / "scaler.pkl"
ENCODER_FILE: Path = MODELS_DIR / "encoder.pkl"
FEATURE_COLUMNS_FILE: Path = MODELS_DIR / "feature_columns.pkl"
LABEL_ENCODER_FILE: Path = MODELS_DIR / "label_encoder.pkl"

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
OUTPUTS_DIR: Path = ROOT_DIR / "outputs"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
METRICS_DIR: Path = OUTPUTS_DIR / "metrics"

# ---------------------------------------------------------------------------
# Specific Output Files
# ---------------------------------------------------------------------------
METRICS_FILE: Path = METRICS_DIR / "model_metrics.json"
COMPARISON_METRICS_FILE: Path = METRICS_DIR / "model_comparison.json"
CLASSIFICATION_REPORT_FILE: Path = REPORTS_DIR / "classification_report.txt"
FEATURE_IMPORTANCE_CSV: Path = REPORTS_DIR / "feature_importance.csv"

# Figure Files
ROC_CURVE_FILE: Path = FIGURES_DIR / "roc_curve.png"
PR_CURVE_FILE: Path = FIGURES_DIR / "precision_recall_curve.png"
CONFUSION_MATRIX_FILE: Path = FIGURES_DIR / "confusion_matrix.png"
FEATURE_IMPORTANCE_PLOT: Path = FIGURES_DIR / "feature_importance.png"
SHAP_SUMMARY_PLOT: Path = FIGURES_DIR / "shap_summary.png"
SHAP_WATERFALL_PLOT: Path = FIGURES_DIR / "shap_waterfall.png"

# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------
LOGS_DIR: Path = ROOT_DIR / "logs"
PIPELINE_LOG_FILE: Path = LOGS_DIR / "pipeline.log"
APP_LOG_FILE: Path = LOGS_DIR / "app.log"

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
TESTS_DIR: Path = ROOT_DIR / "tests"

# ---------------------------------------------------------------------------
# Notebooks
# ---------------------------------------------------------------------------
NOTEBOOKS_DIR: Path = ROOT_DIR / "notebooks"


def ensure_directories() -> None:
    """Create all required directories if they do not yet exist."""
    dirs = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SAMPLE_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        LOGS_DIR,
        TESTS_DIR,
        NOTEBOOKS_DIR,
    ]
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
