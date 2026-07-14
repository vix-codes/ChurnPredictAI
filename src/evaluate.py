"""
src/evaluate.py — Model Evaluation & Visualisation for ChurnPredict AI
=======================================================================
Generates all evaluation plots: ROC, PR curve, Confusion Matrix,
Feature Importance, and threshold sensitivity analysis.

Author  : ChurnPredict AI Engineering Team
Version : 1.0.0
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix,
)

from config import paths
from config.config import TOP_N_FEATURES
from src.utils import get_logger, timer

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Shared Styling
# ---------------------------------------------------------------------------
STYLE = {
    "figure.facecolor": "#0F172A",
    "axes.facecolor": "#1E293B",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#CBD5E1",
    "xtick.color": "#CBD5E1",
    "ytick.color": "#CBD5E1",
    "text.color": "#F1F5F9",
    "grid.color": "#334155",
    "grid.alpha": 0.5,
}


def _apply_style() -> None:
    plt.rcParams.update(STYLE)


# ---------------------------------------------------------------------------
# ROC Curve
# ---------------------------------------------------------------------------
@timer
def plot_roc_curve(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Plot and save the ROC curve with AUC annotation.

    Parameters
    ----------
    model     : fitted classifier
    X_test    : pd.DataFrame
    y_test    : pd.Series
    save_path : Path | None   — where to save the figure

    Returns
    -------
    plt.Figure
    """
    _apply_style()
    proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#3B82F6", lw=2.5, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="#64748B", lw=1.5, linestyle="--", label="Random")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#3B82F6")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — ChurnPredict AI", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11, facecolor="#1E293B", edgecolor="#3B82F6")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = save_path or paths.ROC_CURVE_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.info("ROC curve saved → %s", out)
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Precision-Recall Curve
# ---------------------------------------------------------------------------
@timer
def plot_precision_recall_curve(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot and save Precision-Recall curve."""
    _apply_style()
    proba = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, proba)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="#10B981", lw=2.5, label=f"PR-AUC = {pr_auc:.4f}")
    ax.fill_between(recall, precision, alpha=0.15, color="#10B981")
    baseline = float(y_test.mean())
    ax.axhline(y=baseline, color="#64748B", lw=1.5, linestyle="--",
               label=f"Baseline = {baseline:.2f}")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve — ChurnPredict AI", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11, facecolor="#1E293B", edgecolor="#10B981")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = save_path or paths.PR_CURVE_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.info("PR curve saved → %s", out)
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------------
@timer
def plot_confusion_matrix(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot and save the Confusion Matrix heatmap."""
    _apply_style()
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    labels = ["No Churn (0)", "Churn (1)"]
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels, fontsize=10)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "#0F172A",
                fontsize=14, fontweight="bold",
            )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix — ChurnPredict AI", fontsize=13, fontweight="bold")
    fig.tight_layout()

    out = save_path or paths.CONFUSION_MATRIX_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=STYLE["figure.facecolor"])
    logger.info("Confusion matrix saved → %s", out)
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------------
@timer
def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    top_n: int = TOP_N_FEATURES,
    save_path: Path | None = None,
) -> pd.DataFrame:
    """
    Plot top-N feature importances as a horizontal bar chart.

    Returns
    -------
    pd.DataFrame  — feature importance table (sorted descending)
    """
    _apply_style()

    importances = model.feature_importances_
    fi_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    # Save CSV
    paths.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fi_df.to_csv(paths.FEATURE_IMPORTANCE_CSV, index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(fi_df)))[::-1]
    bars = ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color=colors[::-1])

    for bar, val in zip(bars, fi_df["importance"][::-1]):
        ax.text(
            bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=8, color="#CBD5E1",
        )

    ax.set_xlabel("Feature Importance (Gini)", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances — ChurnPredict AI",
                 fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    out = save_path or paths.FEATURE_IMPORTANCE_PLOT
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=STYLE["figure.facecolor"])
    logger.info("Feature importance plot saved → %s", out)
    plt.close(fig)
    return fi_df


# ---------------------------------------------------------------------------
# Threshold Sensitivity Analysis
# ---------------------------------------------------------------------------
def threshold_sensitivity(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """
    Compute precision, recall, F1, accuracy across multiple decision thresholds.

    Returns
    -------
    pd.DataFrame  — metrics per threshold
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    thresholds = thresholds or [t / 100 for t in range(10, 91, 5)]
    proba = model.predict_proba(X_test)[:, 1]

    rows = []
    for t in thresholds:
        preds = (proba >= t).astype(int)
        rows.append({
            "threshold": t,
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Master Evaluation Runner
# ---------------------------------------------------------------------------
@timer
def run_evaluation(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str],
) -> None:
    """
    Run all evaluation visualisations and save to outputs/figures/.
    """
    plot_roc_curve(model, X_test, y_test)
    plot_precision_recall_curve(model, X_test, y_test)
    plot_confusion_matrix(model, X_test, y_test)
    plot_feature_importance(model, feature_names)
    logger.info("All evaluation plots saved to %s", paths.FIGURES_DIR)
