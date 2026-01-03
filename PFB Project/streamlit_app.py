import json
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st

from breast_cancer_classifier import BreastCancerClassifier, TrainConfig


st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { padding-top: 0rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧬 Breast Cancer Classification System")
st.markdown("### Wisconsin Breast Cancer Dataset Analysis")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Configuration")

    test_size = st.slider(
        "Test Set Size (%)",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
        help="Percentage of data to use for testing",
    ) / 100

    random_state = st.number_input(
        "Random State",
        min_value=0,
        max_value=10_000,
        value=42,
        help="For reproducibility",
    )

    st.subheader("Clinical-ish settings")

    cv_folds = st.slider(
        "Cross-validation folds",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
        help="Used for SVM tuning and probability calibration.",
    )

    calibrate_probabilities = st.toggle(
        "Calibrate probabilities (isotonic)",
        value=True,
        help="More realistic risk estimates; slightly slower.",
    )

    threshold_strategy = st.selectbox(
        "Decision threshold strategy",
        options=["target_sensitivity", "youden", "fixed"],
        index=0,
        help="Real-life screening often prefers high sensitivity.",
    )

    target_sensitivity = st.slider(
        "Target sensitivity (recall for malignant)",
        min_value=0.80,
        max_value=0.99,
        value=0.95,
        step=0.01,
        help="Only used when threshold strategy is target_sensitivity.",
    )

    fixed_threshold = st.slider(
        "Fixed threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help="Only used when threshold strategy is fixed.",
    )

    run_analysis = st.button("🚀 Run Analysis", use_container_width=True)


def _plot_dashboard(results: dict, classifier: BreastCancerClassifier):
    st.header("📊 Results Summary")

    info = results["dataset_info"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Samples", info.get("total_samples"))
    c2.metric("Training Samples", info.get("train_samples"))
    c3.metric("Test Samples", info.get("test_samples"))
    c4.metric("Features", info.get("num_features"))

    st.markdown("---")
    st.header("🎯 Model Performance Comparison")

    lr = results["models"]["Logistic Regression"]
    svm = results["models"]["SVM (Optimized)"]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Logistic Regression")
        a, b = st.columns(2)
        a.metric("Accuracy", f"{lr['accuracy']:.4f}")
        a.metric("Precision", f"{lr['precision']:.4f}")
        b.metric("Recall", f"{lr['recall']:.4f}")
        b.metric("F1-Score", f"{lr['f1_score']:.4f}")
        st.metric("ROC-AUC", f"{lr['roc_auc']:.4f}")
        st.metric("Avg Precision", f"{lr['avg_precision']:.4f}")
        st.metric("Specificity", f"{lr['specificity']:.4f}")
        st.metric("Brier score", f"{lr['brier_score']:.4f}")
        st.caption(f"Threshold used: {lr.get('threshold', 0.5):.3f}")

    with col2:
        st.subheader("SVM (Optimized)")
        a, b = st.columns(2)
        a.metric("Accuracy", f"{svm['accuracy']:.4f}")
        a.metric("Precision", f"{svm['precision']:.4f}")
        b.metric("Recall", f"{svm['recall']:.4f}")
        b.metric("F1-Score", f"{svm['f1_score']:.4f}")
        st.metric("ROC-AUC", f"{svm['roc_auc']:.4f}")
        st.metric("Avg Precision", f"{svm['avg_precision']:.4f}")
        st.metric("Specificity", f"{svm['specificity']:.4f}")
        st.metric("Brier score", f"{svm['brier_score']:.4f}")
        st.caption(f"Threshold used: {svm.get('threshold', 0.5):.3f}")

        if "best_params" in svm:
            st.caption(f"Best params: {svm['best_params']}")

    st.markdown("---")
    st.header("📈 Visualizations")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Breast Cancer Classification - Model Comparison", fontsize=16, fontweight="bold")

    # 1) Performance bar chart
    ax1 = axes[0, 0]
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    lr_values = [lr["accuracy"], lr["precision"], lr["recall"], lr["f1_score"]]
    svm_values = [svm["accuracy"], svm["precision"], svm["recall"], svm["f1_score"]]

    x = np.arange(len(metrics))
    width = 0.35
    ax1.bar(x - width / 2, lr_values, width, label="Logistic Regression", color="skyblue")
    ax1.bar(x + width / 2, svm_values, width, label="SVM (Optimized)", color="lightcoral")
    ax1.set_title("Performance Metrics Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=45, ha="right")
    ax1.set_ylim([0.85, 1.0])
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend()

    # 2) Confusion matrix LR
    ax2 = axes[0, 1]
    cm_lr = [
        [lr["confusion_matrix"]["true_negatives"], lr["confusion_matrix"]["false_positives"]],
        [lr["confusion_matrix"]["false_negatives"], lr["confusion_matrix"]["true_positives"]],
    ]
    sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues", ax=ax2, cbar=False)
    ax2.set_title("Confusion Matrix - Logistic Regression")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    ax2.set_xticklabels(["Benign", "Malignant"])
    ax2.set_yticklabels(["Benign", "Malignant"], rotation=0)

    # 3) Confusion matrix SVM
    ax3 = axes[1, 0]
    cm_svm = [
        [svm["confusion_matrix"]["true_negatives"], svm["confusion_matrix"]["false_positives"]],
        [svm["confusion_matrix"]["false_negatives"], svm["confusion_matrix"]["true_positives"]],
    ]
    sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Reds", ax=ax3, cbar=False)
    ax3.set_title("Confusion Matrix - SVM (Optimized)")
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")
    ax3.set_xticklabels(["Benign", "Malignant"])
    ax3.set_yticklabels(["Benign", "Malignant"], rotation=0)

    # 4) ROC + PR curves side-by-side (computed from trained models)
    ax4 = axes[1, 1]
    try:
        from sklearn.metrics import roc_curve, precision_recall_curve

        lr_score = classifier.predict_proba(classifier.lr_model, classifier.X_test)
        svm_score = classifier.predict_proba(classifier.svm_model, classifier.X_test)
        y_true = classifier.y_test.to_numpy()

        fpr_lr, tpr_lr, _ = roc_curve(y_true, lr_score)
        fpr_svm, tpr_svm, _ = roc_curve(y_true, svm_score)
        ax4.plot(fpr_lr, tpr_lr, label=f"LR (AUC={lr['roc_auc']:.3f})", linewidth=2)
        ax4.plot(fpr_svm, tpr_svm, label=f"SVM (AUC={svm['roc_auc']:.3f})", linewidth=2)
        ax4.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
        ax4.set_xlabel("False Positive Rate")
        ax4.set_ylabel("True Positive Rate")
        ax4.set_title("ROC Curves")
        ax4.grid(alpha=0.3)
        ax4.legend()
    except Exception:
        ax4.axis("off")
        ax4.text(0.5, 0.5, "ROC plot unavailable", ha="center", va="center")

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.header("📋 Detailed Results")

    st.subheader("Quick interpretability")
    top = results.get("interpretability", {}).get("lr_top_positive_coefficients", {})
    if top:
        st.caption("Top LR coefficients increasing malignant risk (directional; not causal).")
        st.dataframe(
            {
                "feature": list(top.keys()),
                "coefficient": list(top.values()),
            },
            use_container_width=True,
        )
    else:
        st.caption("No coefficients available.")

    with st.expander("📄 View Full Results JSON"):
        st.json(results)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            label="📥 Download Results (JSON)",
            data=json.dumps(results, indent=4),
            file_name="cancer_classification_results.json",
            mime="application/json",
        )

    with c2:
        img_bytes = io.BytesIO()
        fig.savefig(img_bytes, format="png", dpi=300, bbox_inches="tight")
        img_bytes.seek(0)
        st.download_button(
            label="📊 Download Visualization (PNG)",
            data=img_bytes,
            file_name="cancer_classification_visualization.png",
            mime="image/png",
        )


if run_analysis:
    with st.spinner("🔄 Running analysis... This may take a minute..."):
        try:
            data_path = Path(__file__).parent / "breast_cancer_data.csv"
            classifier = BreastCancerClassifier(data_path=data_path)
            config = TrainConfig(
                test_size=float(test_size),
                random_state=int(random_state),
                cv_folds=int(cv_folds),
                calibrate_probabilities=bool(calibrate_probabilities),
                threshold_strategy=str(threshold_strategy),
                target_sensitivity=float(target_sensitivity),
                fixed_threshold=float(fixed_threshold),
            )
            results = classifier.run(config=config)

            st.success("✅ Analysis completed successfully!")
            st.markdown("---")
            _plot_dashboard(results, classifier)

            st.markdown("---")
            st.header("🧪 Single-patient risk demo")
            st.warning(
                "This is a **demo only** and not medical advice. Real clinical systems require rigorous validation, "
                "governance, and prospective testing.",
                icon="⚠️",
            )

            model_choice = st.radio(
                "Model",
                options=["SVM (Optimized)", "Logistic Regression"],
                horizontal=True,
            )

            if classifier.X_train is not None:
                defaults = classifier.X_train.median(numeric_only=True).to_dict()
            else:
                defaults = {}

            # Show a smaller set of the most important features (LR coefficients) for usability.
            # Fall back to first 10 features.
            top_feats = list(results.get("interpretability", {}).get("lr_top_positive_coefficients", {}).keys())
            if not top_feats and classifier.feature_names:
                top_feats = classifier.feature_names[:10]

            with st.form("patient_form"):
                st.caption("Enter example feature values (defaults = median of training set).")
                patient_values = {}
                cols = st.columns(2)
                for i, feat in enumerate(top_feats):
                    col = cols[i % 2]
                    val = float(defaults.get(feat, 0.0))
                    patient_values[feat] = col.number_input(feat, value=val, format="%.6f")

                submit = st.form_submit_button("Predict risk")

            if submit:
                # Fill missing features with training medians to build a full 30-feature vector
                if classifier.feature_names:
                    full = {f: float(defaults.get(f, 0.0)) for f in classifier.feature_names}
                    full.update({k: float(v) for k, v in patient_values.items()})
                else:
                    full = patient_values

                pred = classifier.predict_patient(full, model_name=model_choice)
                st.metric("Predicted class", pred["predicted_class"].title())
                st.metric("Malignant probability", f"{pred['prob_malignant']:.3f}")
                st.caption(f"Threshold used by model: {pred['threshold']:.3f}")

        except Exception as e:
            st.error(f"❌ Error during analysis: {e}")
            st.info("If this is a dependency issue, install requirements.txt in your active environment.")
else:
    st.info("👈 Click **Run Analysis** in the sidebar to start.")
    st.markdown(
        """
        ## 📌 About
        This Streamlit demo trains and compares:
        1) **Logistic Regression** (baseline)
        2) **SVM (RBF)** tuned with GridSearchCV

        The ML logic lives in `breast_cancer_classifier.py` so the UI stays clean.
        """
    )
