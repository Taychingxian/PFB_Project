"""
Breast Cancer Classification: Interactive Streamlit Dashboard
==============================================================
Project: Bioinformatics Mini Project - Breast Cancer Diagnosis

Interactive web interface for comparing Logistic Regression vs SVM models
on the Wisconsin Breast Cancer dataset.

Author: TAY CHING XIAN
Run: streamlit run streamlit_app.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stAlert {
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and cache the breast cancer dataset."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y, data


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train Logistic Regression baseline model."""
    with st.spinner("Training Logistic Regression..."):
        start_time = time.time()
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                max_iter=1000,
                solver='lbfgs',
                random_state=42,
                class_weight='balanced'
            ))
        ])
        
        pipeline.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Cross-validation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        return {
            'model': pipeline,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'training_time': training_time,
            'cv_scores': cv_scores
        }


def train_svm(X_train, y_train, X_test, y_test, param_grid):
    """Train optimized SVM with GridSearchCV."""
    with st.spinner("Training SVM with GridSearchCV... This may take a moment."):
        start_time = time.time()
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel='rbf', random_state=42, class_weight='balanced'))
        ])
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        best_pipeline = grid_search.best_estimator_
        
        # Predictions
        y_pred = best_pipeline.predict(X_test)
        y_score = best_pipeline.decision_function(X_test)
        
        return {
            'model': best_pipeline,
            'y_pred': y_pred,
            'y_score': y_score,
            'training_time': training_time,
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'grid_search': grid_search
        }


def calculate_metrics(y_true, y_pred, y_score=None):
    """Calculate comprehensive evaluation metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    if y_score is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_score)
    
    return metrics


def display_metrics(model_name, metrics, training_time, col):
    """Display metrics in a formatted card."""
    with col:
        st.subheader(f"📊 {model_name}")
        
        # Metrics in columns
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        m2.metric("Precision", f"{metrics['precision']:.4f}")
        m3.metric("Recall", f"{metrics['recall']:.4f}")
        m4.metric("F1-Score", f"{metrics['f1']:.4f}")
        
        if 'roc_auc' in metrics:
            st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
        
        st.caption(f"⏱️ Training Time: {training_time:.2f}s")
        
        # Confusion Matrix
        st.markdown("**Confusion Matrix**")
        cm = metrics['confusion_matrix']
        cm_df = pd.DataFrame(
            cm,
            index=['Actual: Malignant', 'Actual: Benign'],
            columns=['Predicted: Malignant', 'Predicted: Benign']
        )
        st.dataframe(cm_df, use_container_width=True)
        
        # TN, FP, FN, TP
        tn, fp, fn, tp = cm.ravel()
        st.markdown(f"""
        - **True Negatives (TN)**: {tn}
        - **False Positives (FP)**: {fp}
        - **False Negatives (FN)**: {fn}
        - **True Positives (TP)**: {tp}
        """)


# Main App
st.title("🧬 Breast Cancer Classification System")
st.markdown("**Systematic Comparison: Logistic Regression vs Support Vector Machine**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Data Split")
    test_size = st.slider("Test Size (%)", 10, 40, 20, 5) / 100
    random_state = st.number_input("Random State", 0, 100, 42, 1)
    
    st.subheader("SVM Hyperparameters")
    st.markdown("**C parameter (Regularization)**")
    c_values = st.multiselect(
        "C values",
        [0.1, 0.5, 1, 2, 5, 10, 20],
        default=[0.1, 1, 5, 10]
    )
    
    st.markdown("**Gamma parameter**")
    gamma_values = st.multiselect(
        "Gamma values",
        ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        default=['scale', 0.01, 0.1]
    )
    
    run_training = st.button("🚀 Run Training", type="primary", use_container_width=True)

# Dataset Info
st.header("📋 Dataset Information")
X, y, data = load_data()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Samples", len(X))
col2.metric("Features", X.shape[1])
col3.metric("Malignant Cases", sum(y == 0))
col4.metric("Benign Cases", sum(y == 1))

with st.expander("📖 View Dataset Description"):
    st.write(data.DESCR)

# Training Section
if run_training:
    if not c_values or not gamma_values:
        st.error("⚠️ Please select at least one value for both C and Gamma parameters.")
    else:
        st.markdown("---")
        st.header("🔬 Model Training & Evaluation")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        info_col1, info_col2 = st.columns(2)
        info_col1.info(f"📚 Training samples: {len(X_train)}")
        info_col2.info(f"📝 Test samples: {len(X_test)}")
        
        # Train models
        progress_bar = st.progress(0, text="Training models...")
        
        # Logistic Regression
        progress_bar.progress(25, text="Training Logistic Regression...")
        lr_results = train_logistic_regression(X_train, y_train, X_test, y_test)
        lr_metrics = calculate_metrics(y_test, lr_results['y_pred'], lr_results['y_proba'])
        
        # SVM
        progress_bar.progress(50, text="Training SVM with GridSearchCV...")
        param_grid = {
            'classifier__C': c_values,
            'classifier__gamma': gamma_values
        }
        svm_results = train_svm(X_train, y_train, X_test, y_test, param_grid)
        svm_metrics = calculate_metrics(y_test, svm_results['y_pred'], svm_results['y_score'])
        
        progress_bar.progress(100, text="Training completed!")
        time.sleep(0.5)
        progress_bar.empty()
        
        st.success("✅ Training completed successfully!")
        
        # Display Results
        st.markdown("---")
        st.header("📈 Results Comparison")
        
        col1, col2 = st.columns(2)
        display_metrics("Logistic Regression (Baseline)", lr_metrics, lr_results['training_time'], col1)
        display_metrics("SVM (Optimized)", svm_metrics, svm_results['training_time'], col2)
        
        # SVM Best Parameters
        st.markdown("---")
        st.subheader("🎯 SVM Optimization Results")
        param_col1, param_col2, param_col3 = st.columns(3)
        param_col1.metric("Best C", svm_results['best_params']['classifier__C'])
        param_col2.metric("Best Gamma", str(svm_results['best_params']['classifier__gamma']))
        param_col3.metric("Best CV F1-Score", f"{svm_results['best_cv_score']:.4f}")
        
        # Cross-validation scores
        st.markdown("---")
        st.subheader("📊 Cross-Validation Scores (5-fold)")
        cv_col1, cv_col2 = st.columns(2)
        
        with cv_col1:
            st.markdown("**Logistic Regression**")
            lr_cv_df = pd.DataFrame({
                'Fold': range(1, 6),
                'F1-Score': lr_results['cv_scores']
            })
            st.dataframe(lr_cv_df, use_container_width=True)
            st.metric("Mean CV F1", f"{lr_results['cv_scores'].mean():.4f}")
            st.metric("Std CV F1", f"{lr_results['cv_scores'].std():.4f}")
        
        with cv_col2:
            st.markdown("**SVM**")
            st.info(f"Best CV F1-Score from GridSearch: {svm_results['best_cv_score']:.4f}")
            
            # Show grid search results summary
            grid_results = pd.DataFrame(svm_results['grid_search'].cv_results_)
            top_results = grid_results.nlargest(5, 'mean_test_score')[
                ['param_classifier__C', 'param_classifier__gamma', 'mean_test_score', 'std_test_score']
            ].round(4)
            top_results.columns = ['C', 'Gamma', 'Mean F1', 'Std F1']
            st.markdown("**Top 5 Parameter Combinations:**")
            st.dataframe(top_results, use_container_width=True)
        
        # Winner Summary
        st.markdown("---")
        st.header("🏆 Summary")
        
        winner_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Training Time'],
            'Logistic Regression': [
                f"{lr_metrics['accuracy']:.4f}",
                f"{lr_metrics['precision']:.4f}",
                f"{lr_metrics['recall']:.4f}",
                f"{lr_metrics['f1']:.4f}",
                f"{lr_metrics['roc_auc']:.4f}",
                f"{lr_results['training_time']:.2f}s"
            ],
            'SVM': [
                f"{svm_metrics['accuracy']:.4f}",
                f"{svm_metrics['precision']:.4f}",
                f"{svm_metrics['recall']:.4f}",
                f"{svm_metrics['f1']:.4f}",
                f"{svm_metrics['roc_auc']:.4f}",
                f"{svm_results['training_time']:.2f}s"
            ]
        }
        
        # Determine winners
        winners = []
        for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            lr_val = lr_metrics[metric_name]
            svm_val = svm_metrics[metric_name]
            if lr_val > svm_val:
                winners.append("✓ LR")
            elif svm_val > lr_val:
                winners.append("✓ SVM")
            else:
                winners.append("=")
        winners.append("✓ LR" if lr_results['training_time'] < svm_results['training_time'] else "✓ SVM")
        
        winner_data['Winner'] = winners
        
        winner_df = pd.DataFrame(winner_data)
        st.dataframe(winner_df, use_container_width=True, hide_index=True)

else:
    st.info("👈 Configure parameters in the sidebar and click **Run Training** to begin!")

# Footer
st.markdown("---")
st.markdown("**Author:** TAY CHING XIAN | **Project:** Bioinformatics Mini Project")
