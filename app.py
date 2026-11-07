"""
Interactive Streamlit Web Application for BITS Pilani M.Tech (AIML) - Machine Learning Assignment 2.
Demonstrates 5 pre-trained classification models on the Breast Cancer Wisconsin (Diagnostic) dataset
without retraining on uploaded test data.
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from model.ml_models import compute_all_metrics

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Diagnostic ML Evaluation | BITS Pilani AIML",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for cards and tables
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(90deg, #0f172a 0%, #1e3a8a 100%);
        color: #ffffff;
        padding: 1.4rem 1.8rem;
        border-radius: 0.75rem;
        margin-bottom: 1.2rem;
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 1.75rem;
        margin-bottom: 0.25rem;
    }
    .main-header p {
        color: #cbd5e1;
        font-size: 0.95rem;
        margin: 0;
    }
    .metric-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #2563eb;
        border-radius: 0.5rem;
        padding: 0.85rem 1rem;
        text-align: center;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #475569;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .metric-value {
        font-size: 1.55rem;
        color: #0f172a;
        font-weight: 700;
        margin-top: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DEFAULT_TEST_CSV = os.path.join(BASE_DIR, "test_data.csv")

MODEL_REGISTRY = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "svm.pkl",
}


@st.cache_resource
def load_artifacts():
    """Loads the saved StandardScaler and all 5 pre-trained classification models from disk."""
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    models = {}
    for name, fname in MODEL_REGISTRY.items():
        mpath = os.path.join(MODEL_DIR, fname)
        with open(mpath, "rb") as f:
            models[name] = pickle.load(f)
    return scaler, models


scaler, trained_models = load_artifacts()
expected_features = list(scaler.feature_names_in_)

# Header Banner
st.markdown(
    """
    <div class="main-header">
        <h1>🧬 Breast Cancer Wisconsin (Diagnostic) — Multi-Model Classification Suite</h1>
        <p>BITS Pilani WILP | M.Tech (AIML) Machine Learning Assignment 2 — Pre-trained Model Evaluation & Diagnostic Workbench</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Evaluation Controls")
    st.markdown(
        "Upload a held-out test CSV containing the **30 nuclear morphology features** and the **`target`** column (`0 = Benign`, `1 = Malignant`)."
    )

    uploaded_file = st.file_uploader(
        "a. Upload Test Dataset (CSV)",
        type=["csv"],
        help="Upload test_data.csv (or any compatible test CSV with the 30 features and target column). Models are evaluated directly without retraining.",
    )

    selected_model_name = st.selectbox(
        "b. Select Classification Model",
        options=list(MODEL_REGISTRY.keys()),
        index=3,  # Default to Naive Bayes (overall top performer)
        help="Choose one of the 5 pre-trained classifiers to inspect detailed metrics, confusion matrix, and classification report.",
    )

    st.divider()
    st.subheader("📥 Download Sample Test Data")
    if os.path.exists(DEFAULT_TEST_CSV):
        with open(DEFAULT_TEST_CSV, "rb") as f_csv:
            st.download_button(
                label="Download test_data.csv (114 rows)",
                data=f_csv.read(),
                file_name="test_data.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.caption(
        "🔒 **Inference-Only Mode**: In compliance with assignment rules, models are loaded from pre-trained `.pkl` artifacts in `model/` without retraining on uploaded test data."
    )

# Load active dataset (uploaded CSV or default test_data.csv)
if uploaded_file is not None:
    try:
        df_eval = pd.read_csv(uploaded_file)
        data_source_label = f"Uploaded File: `{uploaded_file.name}` ({len(df_eval)} instances)"
    except Exception as e:
        st.error(f"Error reading uploaded CSV file: {e}")
        st.stop()
else:
    if os.path.exists(DEFAULT_TEST_CSV):
        df_eval = pd.read_csv(DEFAULT_TEST_CSV)
        data_source_label = f"Default Repository Test Set: `test_data.csv` ({len(df_eval)} instances)"
    else:
        st.warning("Please upload `test_data.csv` in the sidebar to begin evaluation.")
        st.stop()

# Validate columns
missing_cols = [c for c in expected_features if c not in df_eval.columns]
if missing_cols:
    st.error(
        f"Uploaded CSV is missing {len(missing_cols)} required feature column(s): `{', '.join(missing_cols[:5])}`..."
    )
    st.stop()

target_col = None
for candidate in ["target", "diagnosis", "label", "class", "y"]:
    if candidate in df_eval.columns:
        target_col = candidate
        break

if target_col is None:
    st.error(
        "Uploaded CSV must include the ground-truth target column (`target` or `diagnosis`) to compute evaluation metrics."
    )
    st.stop()

# Prepare features and target
X_raw = df_eval[expected_features].values
y_raw = df_eval[target_col]
if y_raw.dtype == object:
    y_true = y_raw.map({"B": 0, "M": 1, "Benign": 0, "Malignant": 1, "0": 0, "1": 1}).values.astype(int)
else:
    y_true = y_raw.values.astype(int)

X_scaled = scaler.transform(X_raw)

# Compute evaluation metrics across all 5 pre-trained models on the active test data
all_model_results = {}
for m_name, clf in trained_models.items():
    y_pred_m = clf.predict(X_scaled)
    y_prob_m = clf.predict_proba(X_scaled)[:, 1]
    all_model_results[m_name] = compute_all_metrics(y_true, y_pred_m, y_prob_m)

active_metrics = all_model_results[selected_model_name]

# Dataset Overview Bar
col_info1, col_info2, col_info3, col_info4 = st.columns(4)
col_info1.info(f"**Active Dataset**\n\n{data_source_label}")
col_info2.info(f"**Feature Count**\n\n`{len(expected_features)}` continuous features")
col_info3.info(
    f"**Class Distribution**\n\nBenign (`0`): `{int(np.sum(y_true == 0))}` | Malignant (`1`): `{int(np.sum(y_true == 1))}`"
)
col_info4.info(f"**Selected Classifier**\n\n`{selected_model_name}`")

with st.expander("🔍 Preview Uploaded / Active Test Data (First 10 Rows)", expanded=False):
    st.dataframe(df_eval.head(10), use_container_width=True)

# Section C: Display of Evaluation Metrics for Selected Model
st.subheader(f"📊 c. Evaluation Metrics — {selected_model_name}")
m_cols = st.columns(6)
metric_keys = [
    ("Accuracy", "Accuracy"),
    ("AUC Score", "AUC"),
    ("Precision", "Precision"),
    ("Recall", "Recall"),
    ("F1 Score", "F1"),
    ("MCC Score", "MCC"),
]
for col_ui, (label_text, key_name) in zip(m_cols, metric_keys):
    val = active_metrics[key_name]
    col_ui.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">{label_text}</div>
            <div class="metric-value">{val:.4f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# Section D: Confusion Matrix, ROC Curve, and Classification Report
st.subheader(f"🧮 d. Confusion Matrix, ROC Curve & Classification Report — {selected_model_name}")
viz_col1, viz_col2, viz_col3 = st.columns([1.1, 1.1, 1.3])

with viz_col1:
    st.markdown("**Confusion Matrix**")
    cm = np.array(active_metrics["confusion_matrix"])
    fig_cm, ax_cm = plt.subplots(figsize=(4.6, 3.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred: Benign (0)", "Pred: Malignant (1)"],
        yticklabels=["Actual: Benign (0)", "Actual: Malignant (1)"],
        annot_kws={"size": 13, "weight": "bold"},
        ax=ax_cm,
    )
    ax_cm.set_title(f"{selected_model_name} Confusion Matrix", fontsize=11, fontweight="bold", pad=10)
    ax_cm.set_xlabel("Predicted Diagnosis", fontsize=10)
    ax_cm.set_ylabel("Actual Diagnosis", fontsize=10)
    fig_cm.tight_layout()
    st.pyplot(fig_cm)
    plt.close(fig_cm)

with viz_col2:
    st.markdown("**ROC Curve (AUC Analysis)**")
    fig_roc, ax_roc = plt.subplots(figsize=(4.6, 3.8))
    for m_name, res in all_model_results.items():
        lw = 2.6 if m_name == selected_model_name else 1.2
        alpha = 1.0 if m_name == selected_model_name else 0.45
        ax_roc.plot(
            res["fpr"],
            res["tpr"],
            linewidth=lw,
            alpha=alpha,
            label=f"{m_name} ({res['AUC']:.4f})",
        )
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", linewidth=1.1)
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.02])
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=9.5)
    ax_roc.set_ylabel("True Positive Rate (Sensitivity)", fontsize=9.5)
    ax_roc.set_title("Receiver Operating Characteristic (ROC)", fontsize=11, fontweight="bold", pad=10)
    ax_roc.legend(loc="lower right", fontsize=7.8, frameon=True)
    ax_roc.grid(alpha=0.25)
    fig_roc.tight_layout()
    st.pyplot(fig_roc)
    plt.close(fig_roc)

with viz_col3:
    st.markdown("**Detailed Classification Report**")
    report_df = pd.DataFrame(active_metrics["classification_report"])
    st.dataframe(
        report_df.style.format(
            {"Precision": "{:.4f}", "Recall": "{:.4f}", "F1-Score": "{:.4f}", "Support": "{:d}"}
        ),
        use_container_width=True,
        hide_index=True,
    )
    tn, fp = cm[0]
    fn, tp = cm[1]
    st.markdown(
        f"""
        - **True Negatives (Benign correctly identified):** `{tn}`
        - **True Positives (Malignant correctly identified):** `{tp}`
        - **False Positives (Type I Error):** `{fp}`
        - **False Negatives (Type II Error):** `{fn}`
        """
    )

st.divider()

# All 5 Models Comparison Table on Test Data
st.subheader("🏆 All 5 Classification Models — Comparison Table on Test Data")
st.markdown(
    "Comparison of all 5 pre-trained classifiers evaluated side-by-side on the active test dataset across all 6 mandatory assignment metrics:"
)

comparison_rows = []
for m_name in MODEL_REGISTRY.keys():
    res = all_model_results[m_name]
    comparison_rows.append(
        {
            "ML Model Name": m_name,
            "Accuracy": res["Accuracy"],
            "AUC": res["AUC"],
            "Precision": res["Precision"],
            "Recall": res["Recall"],
            "F1": res["F1"],
            "MCC": res["MCC"],
        }
    )

comp_df = pd.DataFrame(comparison_rows)

st.dataframe(
    comp_df.style.format(
        {
            "Accuracy": "{:.4f}",
            "AUC": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1": "{:.4f}",
            "MCC": "{:.4f}",
        }
    ).highlight_max(
        subset=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
        color="#dcfce7",
    ),
    use_container_width=True,
    hide_index=True,
)
