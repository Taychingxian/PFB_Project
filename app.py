import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from app import BreastCancerClassifier

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        padding-top: 0rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🧬 Breast Cancer Classification System")
st.markdown("### Wisconsin Breast Cancer Dataset Analysis")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    st.write("Control the analysis parameters below")
    
    test_size = st.slider(
        "Test Set Size (%)",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
        help="Percentage of data to use for testing"
    ) / 100
    
    random_state = st.number_input(
        "Random State",
        min_value=0,
        max_value=1000,
        value=42,
        help="For reproducibility"
    )
    
    run_analysis = st.button("🚀 Run Analysis", key="run_button", use_container_width=True)

# Main content
if run_analysis:
    with st.spinner("🔄 Running analysis... This may take a few minutes..."):
        try:
            # Initialize classifier
            classifier = BreastCancerClassifier(data_path='breast_cancer_data.csv')
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Load data
            status_text.text("📊 Loading dataset...")
            progress_bar.progress(10)
            classifier.load_data()
            
            # Preprocess
            status_text.text("🔧 Preprocessing data...")
            progress_bar.progress(30)
            classifier.preprocess_data(test_size=test_size, random_state=int(random_state))
            
            # Train baseline
            status_text.text("🤖 Training Logistic Regression...")
            progress_bar.progress(50)
            classifier.train_baseline_model()
            
            # Train SVM
            status_text.text("⚡ Training SVM with GridSearchCV...")
            progress_bar.progress(75)
            classifier.train_svm_model()
            
            # Evaluate
            status_text.text("📈 Evaluating models...")
            progress_bar.progress(90)
            classifier.evaluate_models()
            
            # Save results
            status_text.text("💾 Saving results...")
            progress_bar.progress(95)
            classifier.save_results()
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            st.success("✅ Analysis completed successfully!")
            st.markdown("---")
            
            # Display Results
            st.header("📊 Results Summary")
            
            # Dataset Info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Samples", classifier.results['dataset_info']['total_samples'])
            with col2:
                st.metric("Training Samples", classifier.results['dataset_info']['train_samples'])
            with col3:
                st.metric("Test Samples", classifier.results['dataset_info']['test_samples'])
            with col4:
                st.metric("Features", classifier.results['dataset_info']['num_features'])
            
            st.markdown("---")
            
            # Model Comparison
            st.header("🎯 Model Performance Comparison")
            
            col1, col2 = st.columns(2)
            
            # Logistic Regression
            with col1:
                st.subheader("Logistic Regression")
                lr_metrics = classifier.results['models']['Logistic Regression']
                
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Accuracy", f"{lr_metrics['accuracy']:.4f}")
                    st.metric("Precision", f"{lr_metrics['precision']:.4f}")
                with metric_col2:
                    st.metric("Recall", f"{lr_metrics['recall']:.4f}")
                    st.metric("F1-Score", f"{lr_metrics['f1_score']:.4f}")
                
                st.metric("ROC-AUC", f"{lr_metrics['roc_auc']:.4f}")
                st.metric("Specificity", f"{lr_metrics['specificity']:.4f}")
            
            # SVM
            with col2:
                st.subheader("SVM (Optimized)")
                svm_metrics = classifier.results['models']['SVM (Optimized)']
                
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Accuracy", f"{svm_metrics['accuracy']:.4f}")
                    st.metric("Precision", f"{svm_metrics['precision']:.4f}")
                with metric_col2:
                    st.metric("Recall", f"{svm_metrics['recall']:.4f}")
                    st.metric("F1-Score", f"{svm_metrics['f1_score']:.4f}")
                
                st.metric("ROC-AUC", f"{svm_metrics['roc_auc']:.4f}")
                st.metric("Specificity", f"{svm_metrics['specificity']:.4f}")
            
            st.markdown("---")
            
            # Visualizations
            st.header("📈 Visualizations")
            
            # Create visualizations
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Breast Cancer Classification - Model Comparison', fontsize=16, fontweight='bold')
            
            models = {
                'Logistic Regression': classifier.lr_model,
                'SVM (Optimized)': classifier.svm_model
            }
            
            # 1. Performance Metrics Comparison
            ax1 = axes[0, 0]
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            lr_values = [
                classifier.results['models']['Logistic Regression']['accuracy'],
                classifier.results['models']['Logistic Regression']['precision'],
                classifier.results['models']['Logistic Regression']['recall'],
                classifier.results['models']['Logistic Regression']['f1_score']
            ]
            svm_values = [
                classifier.results['models']['SVM (Optimized)']['accuracy'],
                classifier.results['models']['SVM (Optimized)']['precision'],
                classifier.results['models']['SVM (Optimized)']['recall'],
                classifier.results['models']['SVM (Optimized)']['f1_score']
            ]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax1.bar(x - width/2, lr_values, width, label='Logistic Regression', color='skyblue')
            ax1.bar(x + width/2, svm_values, width, label='SVM (Optimized)', color='lightcoral')
            ax1.set_xlabel('Metrics')
            ax1.set_ylabel('Score')
            ax1.set_title('Performance Metrics Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(metrics, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)
            ax1.set_ylim([0.85, 1.0])
            
            # 2. Confusion Matrix - Logistic Regression
            ax2 = axes[0, 1]
            cm_lr = [
                [classifier.results['models']['Logistic Regression']['confusion_matrix']['true_negatives'],
                 classifier.results['models']['Logistic Regression']['confusion_matrix']['false_positives']],
                [classifier.results['models']['Logistic Regression']['confusion_matrix']['false_negatives'],
                 classifier.results['models']['Logistic Regression']['confusion_matrix']['true_positives']]
            ]
            sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar=False)
            ax2.set_title('Confusion Matrix - Logistic Regression')
            ax2.set_xlabel('Predicted')
            ax2.set_ylabel('Actual')
            ax2.set_xticklabels(['Benign', 'Malignant'])
            ax2.set_yticklabels(['Benign', 'Malignant'], rotation=0)
            
            # 3. Confusion Matrix - SVM
            ax3 = axes[1, 0]
            cm_svm = [
                [classifier.results['models']['SVM (Optimized)']['confusion_matrix']['true_negatives'],
                 classifier.results['models']['SVM (Optimized)']['confusion_matrix']['false_positives']],
                [classifier.results['models']['SVM (Optimized)']['confusion_matrix']['false_negatives'],
                 classifier.results['models']['SVM (Optimized)']['confusion_matrix']['true_positives']]
            ]
            sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Reds', ax=ax3, cbar=False)
            ax3.set_title('Confusion Matrix - SVM (Optimized)')
            ax3.set_xlabel('Predicted')
            ax3.set_ylabel('Actual')
            ax3.set_xticklabels(['Benign', 'Malignant'])
            ax3.set_yticklabels(['Benign', 'Malignant'], rotation=0)
            
            # 4. ROC Curves
            ax4 = axes[1, 1]
            for model_name, model in models.items():
                y_pred_proba = model.predict_proba(classifier.X_test_scaled)[:, 1]
                fpr, tpr, _ = roc_curve(classifier.y_test, y_pred_proba)
                auc = roc_auc_score(classifier.y_test, y_pred_proba)
                ax4.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.3f})', linewidth=2)
            
            ax4.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
            ax4.set_xlabel('False Positive Rate')
            ax4.set_ylabel('True Positive Rate')
            ax4.set_title('ROC Curves Comparison')
            ax4.legend()
            ax4.grid(alpha=0.3)
            
            plt.tight_layout()
            
            st.pyplot(fig)
            
            st.markdown("---")
            
            # Detailed Results
            st.header("📋 Detailed Results")
            
            with st.expander("📄 View Full Results JSON"):
                st.json(classifier.results)
            
            # Download results
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Download Results (JSON)",
                    data=json.dumps(classifier.results, indent=4),
                    file_name="cancer_classification_results.json",
                    mime="application/json"
                )
            
            with col2:
                # Save figure to bytes
                import io
                img_bytes = io.BytesIO()
                fig.savefig(img_bytes, format='png', dpi=300, bbox_inches='tight')
                img_bytes.seek(0)
                
                st.download_button(
                    label="📊 Download Visualization (PNG)",
                    data=img_bytes,
                    file_name="cancer_classification_visualization.png",
                    mime="image/png"
                )
            
            st.markdown("---")
            
            # Summary
            st.header("🎯 Summary & Recommendations")
            
            lr_acc = classifier.results['models']['Logistic Regression']['accuracy']
            svm_acc = classifier.results['models']['SVM (Optimized)']['accuracy']
            improvement = (svm_acc - lr_acc) * 100
            
            best_model = 'SVM (Optimized)' if svm_acc > lr_acc else 'Logistic Regression'
            
            st.info(f"""
            **Best Performing Model: {best_model}**
            
            - **Logistic Regression Accuracy**: {lr_acc:.4f} ({lr_acc*100:.2f}%)
            - **SVM (Optimized) Accuracy**: {svm_acc:.4f} ({svm_acc*100:.2f}%)
            - **Improvement**: {improvement:+.2f}%
            
            The {best_model} model shows superior performance on this dataset.
            """)
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.write("Please ensure all required packages are installed:")
            st.code("pip install -r requirements.txt", language="bash")

else:
    # Initial page content
    st.info("👈 Click the **Run Analysis** button in the sidebar to start the classification analysis.")
    
    st.markdown("""
    ## 📌 About This Application
    
    This application performs comprehensive machine learning analysis on the Wisconsin Breast Cancer dataset.
    
    ### Features:
    - 🔄 **Data Loading & Exploration**: Load and analyze the breast cancer dataset
    - 🔧 **Preprocessing**: Feature scaling and train-test splitting
    - 🤖 **Logistic Regression**: Baseline model for comparison
    - ⚡ **SVM with GridSearchCV**: Optimized Support Vector Machine
    - 📊 **Comprehensive Evaluation**: Multiple metrics and visualizations
    - 📥 **Results Export**: Download results and visualizations
    
    ### Dataset Information:
    - **Samples**: 569 breast cancer cases
    - **Features**: 30 numerical features
    - **Target**: Malignant (M) or Benign (B)
    
    ### Models Compared:
    1. **Logistic Regression** - Fast, interpretable baseline
    2. **Support Vector Machine (SVM)** - Optimized with GridSearchCV
    
    """)
