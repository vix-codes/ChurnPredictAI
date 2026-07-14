# 🔮 ChurnPredict AI — Enterprise Edition

> An end-to-end customer churn prediction platform featuring advanced feature engineering, hyperparameter tuning, SHAP explainability, interactive what-if analysis, batch predictions, and a modern Streamlit dashboard for real-world business insights.

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red?logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Overview

ChurnPredict AI is a fully modular, production-quality machine learning system that:

- **Preprocesses** raw customer data (IBM Telco schema)
- **Engineers** 15+ business-driven features
- **Trains** a Random Forest with GridSearchCV (optional: XGBoost, LightGBM)
- **Evaluates** with full metrics suite (AUC, F1, Precision, Recall, Confusion Matrix)
- **Explains** predictions with SHAP (summary, waterfall, dependence plots)
- **Serves** predictions through a modern Streamlit dashboard
- **Supports** What-If analysis, batch CSV upload, and prediction history

---

## 🏗️ Architecture

```
ChurnPredictAI/
│
├── app/                    # Streamlit dashboard
│   ├── main.py             # Dashboard entry point (4 tabs)
│   ├── sidebar.py          # Customer input widgets
│   ├── metrics.py          # KPI card components
│   └── visuals.py          # Plotly chart library (12 chart types)
│
├── config/
│   ├── config.py           # All constants, hyperparameters, thresholds
│   └── paths.py            # Centralised pathlib path management
│
├── data/
│   ├── raw/                # Raw CSV input
│   ├── processed/          # Encoded + scaled Parquet output
│   └── sample/             # Synthetic data generator
│
├── models/                 # Saved artifacts (.pkl files)
│
├── outputs/
│   ├── reports/            # Classification report, feature importance CSV
│   ├── figures/            # ROC, PR, CM, SHAP plots
│   └── metrics/            # model_metrics.json, comparison.json
│
├── src/
│   ├── utils.py            # Logger factory, timer decorator, helpers
│   ├── preprocess.py       # Load → Clean → Encode → Scale pipeline
│   ├── features.py         # 15+ business feature engineering functions
│   ├── train.py            # GridSearchCV training, metrics, model saving
│   ├── evaluate.py         # Evaluation plots (matplotlib)
│   ├── inference.py        # load/predict/batch/validate API
│   ├── explain.py          # SHAP + permutation importance
│   └── pipeline.py         # End-to-end CLI orchestrator
│
├── tests/                  # Pytest unit + integration tests
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone and install dependencies

```bash
git clone https://github.com/your-org/churnpredict-ai.git
cd churnpredict-ai
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

### 2. Prepare the dataset

**Option A — Use real IBM Telco dataset:**
```
Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
Place the CSV as: data/raw/telco_churn.csv
```

**Option B — Generate synthetic data:**
```bash
python data/sample/generate_sample.py
```

### 3. Run the full pipeline

```bash
# Full pipeline (preprocessing → training → evaluation → SHAP)
python src/pipeline.py

# Skip SHAP (faster, for quick iteration)
python src/pipeline.py --skip-explain

# Skip model comparison (XGBoost/LightGBM)
python src/pipeline.py --skip-compare

# Custom data path
python src/pipeline.py --data path/to/your/data.csv
```

### 4. Launch the dashboard

```bash
streamlit run app/main.py
```

---

## 📊 Dataset

The system is built for the **IBM Telco Customer Churn** dataset. Key columns:

| Column | Type | Description |
|---|---|---|
| customerID | string | Unique identifier |
| tenure | int | Months as customer |
| MonthlyCharges | float | Current monthly bill |
| TotalCharges | float | Total billed (may contain blanks) |
| Contract | categorical | Month-to-month / One year / Two year |
| InternetService | categorical | DSL / Fiber optic / No |
| Churn | binary | Yes / No (target) |

---

## 🧠 Machine Learning

### Model: Random Forest Classifier

| Step | Detail |
|---|---|
| Split | Stratified 80/20 train/test |
| Tuning | GridSearchCV (5-fold StratifiedKFold) |
| Metric | ROC AUC (primary), F1, Precision, Recall |
| Classes | Balanced class weighting optional |
| Threshold | Configurable via dashboard (default: 0.50) |

### Optional Models (auto-detected if installed)

- **XGBoost** — `pip install xgboost`
- **LightGBM** — `pip install lightgbm`

### Typical Performance (on synthetic data)

| Metric | Score |
|---|---|
| ROC AUC | 0.84 – 0.89 |
| F1 Score | 0.61 – 0.68 |
| Accuracy | 0.79 – 0.83 |
| Recall   | 0.70 – 0.78 |

---

## 🔮 Dashboard Features

### Tab 1 — Dashboard
- **Live Gauge Chart** — Churn probability with colour-coded risk zones
- **What-If Analysis** — Sensitivity chart as monthly charges vary
- **Customer Profiler** — Current input values summarised
- **Business Impact** — Revenue at risk, retention cost, ROI
- **AI Recommendation** — Personalised retention action
- **Batch Upload** — Predict churn for CSV file of customers
- **Prediction History** — Session history with CSV download

### Tab 2 — Model Insights
- **5-column KPI strip** — Accuracy, Precision, Recall, F1, AUC
- **Cross-Validation table** — Mean ± Std for all metrics
- **Feature Importance** — Interactive top-N bar chart
- **Evaluation Plots** — ROC, PR Curve, Confusion Matrix
- **SHAP Plots** — Summary beeswarm, waterfall (if generated)
- **Model Comparison** — RF vs XGBoost vs LightGBM AUC bar

### Tab 3 — Dataset Overview
- Churn distribution pie
- Internet service breakdown
- Tenure histogram (by churn)
- Monthly charges box plot
- Contract type vs churn bar
- Correlation heatmap (top-15 features)

### Tab 4 — Business Insights
- Six insight cards with domain-driven findings
- Churn rate by contract type (bar)
- Churn rate by tenure group (line)
- Strategic retention recommendations

---

## 🎯 Feature Engineering

15+ engineered features including:

| Feature | Description |
|---|---|
| TenureGroup | New / Growing / Established / Loyal / Champion |
| SpendCategory | Budget / Standard / Premium / Enterprise |
| AvgMonthlySpend | TotalCharges ÷ tenure |
| HighValueCustomer | MonthlyCharges ≥ $70 |
| ContractLength | 1 / 12 / 24 months (numeric) |
| CustomerLifetimeEstimate | Projected revenue |
| SupportUsageScore | Sum of security + tech + device add-ons |
| StreamingUsageScore | StreamingTV + StreamingMovies |
| FamilyCustomer | Partner AND Dependents = Yes |
| RetentionRiskBucket | Rule-based risk tier (0–5 scale) |
| HighCharge_MTM | Interaction: high charge + month-to-month |
| NewNoSupport | Interaction: new customer + no support |
| ChargeXTenure | MonthlyCharges × tenure |

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov=config --cov-report=term-missing

# Run specific test file
pytest tests/test_preprocess.py -v
pytest tests/test_features.py -v
pytest tests/test_inference.py -v
pytest tests/test_pipeline.py -v
```

---

## 📁 Output Artifacts

After running the pipeline:

```
models/
├── random_forest.pkl       # Best model from GridSearchCV
├── scaler.pkl              # StandardScaler
├── encoder.pkl             # OneHotEncoder
└── feature_columns.pkl     # Ordered feature list

outputs/
├── metrics/
│   ├── model_metrics.json       # Full metrics (train/test/CV)
│   └── model_comparison.json    # AUC by model
├── reports/
│   ├── classification_report.txt
│   ├── feature_importance.csv
│   └── permutation_importance.csv
└── figures/
    ├── roc_curve.png
    ├── precision_recall_curve.png
    ├── confusion_matrix.png
    ├── feature_importance.png
    ├── shap_summary.png
    └── shap_waterfall.png
```

---

## 🔧 Configuration

All constants live in `config/config.py`:

```python
RANDOM_STATE = 42
TEST_SIZE = 0.20
RF_CV_FOLDS = 5
DEFAULT_THRESHOLD = 0.50
HIGH_VALUE_MONTHLY_THRESHOLD = 70.0
```

All paths live in `config/paths.py` (pathlib-based, no hardcoded strings).

---

## 🚀 Future Improvements

- [ ] MLflow experiment tracking
- [ ] FastAPI REST inference endpoint
- [ ] SMOTE class balancing
- [ ] Optuna hyperparameter optimisation
- [ ] PDF report generation (ReportLab / WeasyPrint)
- [ ] Real-time streaming predictions (Kafka / Redis)
- [ ] Docker containerisation + CI/CD
- [ ] A/B testing framework for model versions
- [ ] Customer segmentation (K-Means clustering)
- [ ] Scheduled retraining pipeline (Airflow / Prefect)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👥 Authors

**ChurnPredict AI Engineering Team**

Built to demonstrate enterprise-grade Machine Learning engineering practices suitable for top analytics firms and technology companies.

---

*"Predict churn before it happens. Retain customers before they leave."*
