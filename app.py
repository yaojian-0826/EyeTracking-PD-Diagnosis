"""
Eye Movement SHAP Calculator
=============================
An interactive Streamlit web app for Parkinson's disease classification
using Random Forest + SHAP explanation on eye-movement features.

Design reference: DCM SHAP Calculator (yaojian-0826/DCM_shap-calculator)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import shap
import io
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    classification_report, accuracy_score
)

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

# ══════════════════════════════════════════
# Page config
# ══════════════════════════════════════════
st.set_page_config(
    page_title="Eye Movement SHAP Calculator",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════
# Custom CSS
# ══════════════════════════════════════════
st.markdown("""
<style>
    .risk-high {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white; padding: 1.5rem; border-radius: 1rem;
        text-align: center; font-size: 1.5rem; font-weight: bold; margin-bottom: 1rem;
    }
    .risk-low {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white; padding: 1.5rem; border-radius: 1rem;
        text-align: center; font-size: 1.5rem; font-weight: bold; margin-bottom: 1rem;
    }
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #3b82f6 100%);
        color: white; padding: 2rem; border-radius: 1rem; margin-bottom: 2rem;
    }
    .main-header h1 { margin: 0; font-size: 2rem; }
    .main-header p { margin: 0.5rem 0 0 0; opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# Feature Labels
# ══════════════════════════════════════════
FEATURE_LABELS = {
    "visuospatialExecutiveFunction": "Visuospatial Executive Function",
    "attention": "Attention",
    "meanOverlapSaccadeVelocity": "Mean Overlap Saccade Velocity",
    "meanAntiSaccadeVelocity": "Mean Anti-Saccade Velocity",
    "orientation": "Orientation",
}

FEATURE_DESCRIPTIONS = {
    "visuospatialExecutiveFunction": (
        "Visuospatial executive function score. Higher values indicate better performance."
    ),
    "attention": (
        "Attention function score. Higher values indicate better performance."
    ),
    "meanOverlapSaccadeVelocity": (
        "Average velocity of overlap saccades (deg/s). Reflects saccadic control."
    ),
    "meanAntiSaccadeVelocity": (
        "Average velocity of anti-saccades (deg/s). Reflects inhibitory control."
    ),
    "orientation": (
        "Orientation function score. Higher values indicate better performance."
    ),
}

# ══════════════════════════════════════════
# Data & Model Configuration
# ══════════════════════════════════════════
TRAIN_FILE = "shap解释测试集.xlsx"
VAL_FILE = "shap解释验证集.xlsx"
TARGET_COL = "group"
TARGET_MAP = {"X1": 1, "X0": 0}

# ══════════════════════════════════════════
# Model Training (Cached)
# ══════════════════════════════════════════
@st.cache_resource(show_spinner="Training Random Forest model...")
def load_model():
    train_data = pd.read_excel(TRAIN_FILE, index_col=0)
    val_data = pd.read_excel(VAL_FILE, index_col=0)

    X_train_raw = train_data.drop(columns=[TARGET_COL])
    y_train_raw = train_data[TARGET_COL].map(TARGET_MAP)
    X_val_raw = val_data.drop(columns=[TARGET_COL])
    y_val_raw = val_data[TARGET_COL].map(TARGET_MAP)

    feature_names = X_train_raw.columns.tolist()

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_train = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train_raw.index)
    X_val = pd.DataFrame(X_val_scaled, columns=feature_names, index=X_val_raw.index)

    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train_raw)

    # SHAP
    explainer = shap.TreeExplainer(rf_model)
    shap_raw = explainer.shap_values(X_val)
    ev_raw = explainer.expected_value

    # Extract class 1 SHAP values robustly
    if isinstance(shap_raw, list):
        shap_values = shap_raw[1]
        base_value = ev_raw[1] if isinstance(ev_raw, (list, np.ndarray)) else ev_raw
    elif isinstance(shap_raw, np.ndarray) and shap_raw.ndim == 3:
        shap_values = shap_raw[:, :, 1]
        base_value = ev_raw[1] if isinstance(ev_raw, np.ndarray) and len(ev_raw) == 2 else ev_raw
    else:
        shap_values = shap_raw
        base_value = ev_raw

    if isinstance(base_value, np.ndarray):
        base_value = float(base_value.flat[0])
    else:
        base_value = float(base_value)

    return (rf_model, explainer, scaler, feature_names,
            X_train, y_train_raw, X_val, y_val_raw,
            shap_values, base_value)


def get_shap_single(explainer, X_row):
    """Get SHAP values for a single sample (class 1)."""
    sv = explainer.shap_values(X_row)
    if isinstance(sv, list):
        return sv[1][0] if len(sv) == 2 else sv[0]
    return sv[0] if sv.ndim > 1 else sv


# ══════════════════════════════════════════
# Custom Force Plot
# ══════════════════════════════════════════
def plot_force_plot_matplotlib(base_value, shap_values, features, feature_names):
    sorted_idx = np.argsort(np.abs(shap_values))[::-1]
    n_display = min(len(feature_names), len(feature_names))
    top_idx = sorted_idx[:n_display]

    fig, ax = plt.subplots(figsize=(10, 4))
    y_pos = np.arange(n_display)
    colors = ["#ef4444" if shap_values[i] > 0 else "#10b981" for i in top_idx]
    bars = ax.barh(y_pos, [shap_values[i] for i in top_idx],
                   color=colors, height=0.6, alpha=0.85)

    display_names = [FEATURE_LABELS.get(feature_names[i], feature_names[i]) for i in top_idx]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=10)
    ax.set_xlabel("SHAP Value (contribution to prediction)", fontsize=11)
    ax.axvline(0, color="black", linewidth=1)

    for bar, i in zip(bars, top_idx):
        width = bar.get_width()
        label_x = width + 0.005 if width >= 0 else width - 0.005
        ha = "left" if width >= 0 else "right"
        ax.text(label_x, bar.get_y() + bar.get_height() / 2,
                f"{shap_values[i]:+.4f}\n(val={features[i]:.2f})",
                va="center", ha=ha, fontsize=9)

    prediction = base_value + np.sum(shap_values)
    ax.set_title(
        f"SHAP Force Plot\nBase: {base_value:.4f} → Prediction: {prediction:.4f}",
        fontsize=12, fontweight="bold", pad=15
    )

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#ef4444", label="Increases Case risk"),
        Patch(facecolor="#10b981", label="Decreases Case risk"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════
# Load model
# ══════════════════════════════════════════
try:
    (rf_model, explainer, scaler, feature_names,
     X_train, y_train, X_val, y_val,
     shap_values_all, base_val) = load_model()
except FileNotFoundError:
    st.error(
        "❌ **Data files not found.**\n\n"
        "Please make sure the following files are in the same directory as `app.py`:\n"
        "- `shap解释测试集.xlsx` (training set)\n"
        "- `shap解释验证集.xlsx` (validation set)"
    )
    st.stop()

# ══════════════════════════════════════════
# Header
# ══════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>👁 Random Forest + SHAP Eye Movement Calculator</h1>
    <p>Parkinson's Disease Classification Based on Eye Movement Features</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    page = st.radio(
        "",
        [
            "🔮 Predict & Explain",
            "📊 Model Evaluation",
            "🌐 Global SHAP",
            "📋 Validation Samples",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Model Info**")
    st.caption(f"Training samples: **{len(X_train)}**")
    st.caption(f"Validation samples: **{len(X_val)}**")
    st.caption(f"Features: **{len(feature_names)}**")
    st.caption(f"Algorithm: **Random Forest**")
    st.caption(f"Base value: **{base_val:.4f}**")
    st.markdown("---")
    st.caption("Built with Streamlit + Random Forest + SHAP")


# ══════════════════════════════════════════
# PAGE 1 — Predict & Explain
# ══════════════════════════════════════════
if page.startswith("🔮"):
    st.subheader("🧪 Enter Eye Movement Data")
    st.caption(
        "Input eye movement parameters for a new subject. "
        "Values will be standardized automatically using the training set scaler."
    )

    # ---- Default values from validation set median ----
    defaults = X_val.median()

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown("##### Cognitive Function Scores")
        vsef = st.number_input(
            FEATURE_LABELS["visuospatialExecutiveFunction"],
            value=float(defaults["visuospatialExecutiveFunction"]),
            step=0.01, format="%.4f",
            help=FEATURE_DESCRIPTIONS["visuospatialExecutiveFunction"],
        )
        attn = st.number_input(
            FEATURE_LABELS["attention"],
            value=float(defaults["attention"]),
            step=0.01, format="%.4f",
            help=FEATURE_DESCRIPTIONS["attention"],
        )
        orient = st.number_input(
            FEATURE_LABELS["orientation"],
            value=float(defaults["orientation"]),
            step=0.01, format="%.4f",
            help=FEATURE_DESCRIPTIONS["orientation"],
        )

    with col_right:
        st.markdown("##### Saccade Velocity Parameters")
        mosv = st.number_input(
            FEATURE_LABELS["meanOverlapSaccadeVelocity"],
            value=float(defaults["meanOverlapSaccadeVelocity"]),
            step=0.01, format="%.4f",
            help=FEATURE_DESCRIPTIONS["meanOverlapSaccadeVelocity"],
        )
        masv = st.number_input(
            FEATURE_LABELS["meanAntiSaccadeVelocity"],
            value=float(defaults["meanAntiSaccadeVelocity"]),
            step=0.01, format="%.4f",
            help=FEATURE_DESCRIPTIONS["meanAntiSaccadeVelocity"],
        )

    submitted = st.button(
        "🚀 Run Prediction & SHAP Explanation",
        type="primary", use_container_width=True,
    )

    if submitted:
        raw_input = np.array([[vsef, attn, mosv, masv, orient]])
        scaled_input = scaler.transform(raw_input)
        sample_df = pd.DataFrame(scaled_input, columns=feature_names)

        pred_class = int(rf_model.predict(sample_df)[0])
        pred_prob = rf_model.predict_proba(sample_df)[0]

        col_res1, col_res2 = st.columns([1, 1.2], gap="large")

        with col_res1:
            if pred_class == 1:
                st.markdown("""
                <div class="risk-high">
                    ⚠️ High Risk (Case)<br>
                    <span style="font-size:0.9rem;font-weight:normal">
                        The model predicts this subject belongs to the Case group.
                    </span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="risk-low">
                    ✅ Low Risk (Control)<br>
                    <span style="font-size:0.9rem;font-weight:normal">
                        The model predicts this subject belongs to the Control group.
                    </span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.metric("P(Case)", f"{pred_prob[1]*100:.1f}%")
            c2.metric("P(Control)", f"{pred_prob[0]*100:.1f}%")

            fig_pb, ax = plt.subplots(figsize=(5, 1.6))
            labels = ["Control", "Case"]
            bars = ax.barh(labels, [pred_prob[0], pred_prob[1]],
                           color=["#10b981", "#ef4444"], height=0.5)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            ax.axvline(0.5, color="gray", lw=1, ls="--")
            for b, v in zip(bars, [pred_prob[0], pred_prob[1]]):
                ax.text(v + 0.01, b.get_y() + b.get_height() / 2,
                        f"{v*100:.1f}%", va="center", fontsize=10, fontweight="bold")
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_pb, use_container_width=True)
            plt.close(fig_pb)

        with col_res2:
            st.markdown("**📊 Feature Values (Standardized)**")
            feat_display = pd.DataFrame({
                "Feature": [FEATURE_LABELS.get(f, f) for f in feature_names],
                "Raw Value": raw_input[0],
                "Standardized": scaled_input[0],
            })
            st.dataframe(feat_display.round(4), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("🔍 SHAP Explanation")

        sv_s = get_shap_single(explainer, sample_df)
        dv_r = scaled_input[0]
        bv_r = round(base_val, 4)

        exp_obj = shap.Explanation(
            values=np.round(sv_s, 4),
            base_values=bv_r,
            data=np.round(dv_r, 4),
            feature_names=feature_names,
        )

        col_wf, col_fp = st.columns(2, gap="large")

        with col_wf:
            st.markdown("**💧 Waterfall Plot**")
            fig_wf, _ = plt.subplots(figsize=(8, 6))
            shap.plots.waterfall(exp_obj, show=False, max_display=10)
            plt.title("SHAP Waterfall (Your Subject)", fontsize=12, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_wf, use_container_width=True)
            plt.close(fig_wf)

        with col_fp:
            st.markdown("**⚡ Force Plot**")
            st.caption("Red = increases Case risk, Green = decreases Case risk")
            fig_fp = plot_force_plot_matplotlib(bv_r, sv_s, dv_r, feature_names)
            st.pyplot(fig_fp, use_container_width=True)
            plt.close(fig_fp)

        st.markdown("**📊 SHAP Feature Contributions**")
        shap_breakdown = []
        for f, v, s in zip(feature_names, dv_r, sv_s):
            shap_breakdown.append({
                "Feature": FEATURE_LABELS.get(f, f),
                "Value (std)": round(float(v), 4),
                "SHAP Value": round(float(s), 4),
                "Effect": "↑ Increases risk" if s >= 0 else "↓ Decreases risk",
            })
        shap_breakdown.sort(key=lambda x: abs(x["SHAP Value"]), reverse=True)
        df_bd = pd.DataFrame(shap_breakdown)

        def color_shap(val):
            if isinstance(val, float):
                return f"color: {'#dc2626' if val > 0 else '#059669'}; font-weight: bold"
            return ""

        st.dataframe(
            df_bd.style.map(color_shap, subset=["SHAP Value"]),
            use_container_width=True, hide_index=True,
        )


# ══════════════════════════════════════════
# PAGE 2 — Model Evaluation
# ══════════════════════════════════════════
elif page.startswith("📊"):
    st.subheader("📈 Model Performance on Validation Set")

    y_pred = rf_model.predict(X_val)
    y_prob = rf_model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_prob) if len(np.unique(y_val)) > 1 else 0.5
    acc = accuracy_score(y_val, y_pred) * 100
    rep = classification_report(y_val, y_pred, output_dict=True,
                                target_names=["Control", "Case"])

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("AUC-ROC", f"{auc:.3f}")
    m2.metric("Accuracy", f"{acc:.1f}%")
    m3.metric("Precision (Case)", f"{rep['Case']['precision']*100:.1f}%")
    m4.metric("Recall (Case)", f"{rep['Case']['recall']*100:.1f}%")
    m5.metric("F1 (Case)", f"{rep['Case']['f1-score']*100:.1f}%")

    st.divider()

    col_cm, col_roc = st.columns(2, gap="large")

    with col_cm:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_val, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Control", "Case"],
                    yticklabels=["Control", "Case"],
                    ax=ax, annot_kws={"size": 14})
        ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_roc:
        st.markdown("**ROC Curve**")
        fpr, tpr, _ = roc_curve(y_val, y_prob)
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.plot(fpr, tpr, color="darkorange", lw=2.5, label=f"AUC = {auc:.3f}")
        ax2.plot([0, 1], [0, 1], "navy", lw=1.5, ls="--", label="Random")
        ax2.fill_between(fpr, tpr, alpha=0.08, color="darkorange")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title("ROC Curve", fontsize=13, fontweight="bold")
        ax2.legend(loc="lower right")
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    st.divider()
    st.markdown("**Full Classification Report**")
    rep_df = pd.DataFrame(rep).transpose().round(3)
    st.dataframe(rep_df, use_container_width=True)


# ══════════════════════════════════════════
# PAGE 3 — Global SHAP
# ══════════════════════════════════════════
elif page.startswith("🌐"):
    st.subheader("🌐 Global SHAP Analysis")
    st.caption(f"SHAP values computed on the validation set ({len(X_val)} samples)")

    mean_abs = np.abs(shap_values_all).mean(0)
    imp_df = pd.DataFrame({
        "Feature": [FEATURE_LABELS.get(f, f) for f in feature_names],
        "Mean |SHAP|": np.round(mean_abs, 4),
    }).sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)
    imp_df.index += 1

    col_tbl, col_bar = st.columns([1, 1.4], gap="large")

    with col_tbl:
        st.markdown("**Feature Importance Ranking**")
        st.dataframe(imp_df, use_container_width=True)

    with col_bar:
        st.markdown("**SHAP Bar Plot**")
        fig, _ = plt.subplots(figsize=(8, 6))
        shap.summary_plot(shap_values_all, X_val, plot_type="bar",
                          show=False, color="#5865F2")
        plt.title("Feature Importance (SHAP)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.divider()

    col_bee, col_dep = st.columns(2, gap="large")

    with col_bee:
        st.markdown("**SHAP Beeswarm Plot**")
        fig, _ = plt.subplots(figsize=(8, 6))
        shap.summary_plot(shap_values_all, X_val, show=False)
        plt.title("Feature Impact Distribution", fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_dep:
        st.markdown("**SHAP Dependence Plot**")
        top_idx = int(np.argsort(mean_abs)[-1])
        top_feat = feature_names[top_idx]
        sel_feat = st.selectbox("Select feature:", feature_names, index=top_idx)
        fig, ax = plt.subplots(figsize=(7, 5))
        shap.dependence_plot(sel_feat, shap_values_all, X_val,
                             interaction_index="auto", show=False, ax=ax)
        ax.set_title(
            f"Dependence: {FEATURE_LABELS.get(sel_feat, sel_feat)}",
            fontsize=11, fontweight="bold",
        )
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ══════════════════════════════════════════
# PAGE 4 — Validation Samples
# ══════════════════════════════════════════
elif page.startswith("📋"):
    st.subheader("📋 Explore Validation Samples")
    st.caption("Browse any of the validation samples and view their SHAP explanations.")

    col_ctrl, _ = st.columns([1, 2])
    with col_ctrl:
        idx = st.slider(
            "Sample index",
            min_value=0, max_value=len(X_val) - 1, value=0, step=1,
        )
        st.caption(f"Showing sample **#{idx}**")

    sv_s = shap_values_all[idx]
    dv_r = X_val.iloc[idx].values
    pred = int(rf_model.predict(X_val.iloc[[idx]])[0])
    prob = rf_model.predict_proba(X_val.iloc[[idx]])[0]
    true_l = int(y_val.iloc[idx])
    correct = pred == true_l

    col_info, col_feat = st.columns([1, 1.2], gap="large")

    with col_info:
        label_str = "Case" if pred == 1 else "Control"
        true_str = "Case" if true_l == 1 else "Control"
        correct_str = "✅ Correct" if correct else "❌ Wrong"

        if pred == 1:
            st.markdown(f"""
            <div class="risk-high">
                ⚠️ Predicted: {label_str}<br>
                True label: {true_str} &nbsp; {correct_str}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-low">
                ✅ Predicted: {label_str}<br>
                True label: {true_str} &nbsp; {correct_str}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("P(Case)", f"{prob[1]*100:.1f}%")
        c2.metric("P(Control)", f"{prob[0]*100:.1f}%")

    with col_feat:
        st.markdown("**Feature values (standardized)**")
        feat_df = pd.DataFrame({
            "Feature": [FEATURE_LABELS.get(f, f) for f in feature_names],
            "Value": np.round(dv_r, 4),
            "SHAP": np.round(sv_s, 4),
        }).sort_values("SHAP", key=abs, ascending=False).reset_index(drop=True)
        feat_df.index += 1
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    st.divider()

    exp_obj = shap.Explanation(
        values=np.round(sv_s, 4),
        base_values=round(base_val, 4),
        data=np.round(dv_r, 4),
        feature_names=feature_names,
    )

    col_wf, col_fp = st.columns(2, gap="large")

    with col_wf:
        st.markdown(f"**💧 Waterfall Plot — Sample #{idx}**")
        fig, _ = plt.subplots(figsize=(8, 6))
        shap.plots.waterfall(exp_obj, show=False, max_display=10)
        plt.title(f"SHAP Waterfall (Sample #{idx})", fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_fp:
        st.markdown(f"**⚡ Force Plot — Sample #{idx}**")
        fig2 = plot_force_plot_matplotlib(base_val, sv_s, dv_r, feature_names)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)
