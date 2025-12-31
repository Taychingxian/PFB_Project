from __future__ import annotations

import time
from typing import Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
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
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class ModelMetrics:
    """Container for comprehensive model evaluation metrics."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: np.ndarray
    classification_report: str
    training_samples: int
    test_samples: int
    cv_scores: Dict[str, float] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to JSON-serializable dictionary."""
        return {
            "model_name": self.model_name,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "roc_auc": round(self.roc_auc, 4),
            "confusion_matrix": self.confusion_matrix.tolist(),
            "classification_report": self.classification_report,
            "training_samples": self.training_samples,
            "test_samples": self.test_samples,
            "cv_scores": {k: round(v, 4) for k, v in self.cv_scores.items()},
            "hyperparameters": self.hyperparameters,
            "training_time_seconds": round(self.training_time, 2),
        }


class BreastCancerClassifier:
    """
    Systematic breast cancer classification system.
    
    Implements baseline and optimized models with comprehensive evaluation.
    """
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize the classifier.
        
        Args:
            test_size: Proportion of dataset for testing (default: 0.2)
            random_state: Random seed for reproducibility (default: 42)
        """
        self.test_size = test_size
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        
        logger.info("Initializing Breast Cancer Classifier")
        logger.info(f"Configuration: test_size={test_size}, random_state={random_state}")
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load the Wisconsin Breast Cancer dataset.
        
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        logger.info("Loading Wisconsin Breast Cancer dataset...")
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")
        
        logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Class distribution: Malignant={sum(y==0)}, Benign={sum(y==1)}")
        
        return X, y
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Split data into training and test sets with stratification.
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        logger.info("Splitting data into train/test sets...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state, 
            stratify=y
        )
        
        logger.info(f"Training set: {len(self.X_train)} samples")
        logger.info(f"Test set: {len(self.X_test)} samples")
    
    def evaluate_model(
        self, 
        name: str, 
        pipeline: Pipeline, 
        y_true: pd.Series, 
        y_pred: np.ndarray,
        training_time: float = 0.0,
        cv_scores: Dict[str, float] = None,
        hyperparams: Dict[str, Any] = None
    ) -> ModelMetrics:
        """
        Comprehensive model evaluation with multiple metrics.
        
        Args:
            name: Model name
            pipeline: Trained pipeline
            y_true: True labels
            y_pred: Predicted labels
            training_time: Time taken to train (seconds)
            cv_scores: Cross-validation scores
            hyperparams: Model hyperparameters
            
        Returns:
            ModelMetrics object with all evaluation results
        """
        # Calculate probabilities for ROC-AUC (if available)
        if hasattr(pipeline, 'predict_proba'):
            y_proba = pipeline.predict_proba(self.X_test)[:, 1]
            roc_auc = roc_auc_score(y_true, y_proba)
        else:
            # For SVM without probability, use decision function
            y_score = pipeline.decision_function(self.X_test)
            roc_auc = roc_auc_score(y_true, y_score)
        
        metrics = ModelMetrics(
            model_name=name,
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, zero_division=0),
            recall=recall_score(y_true, y_pred, zero_division=0),
            f1_score=f1_score(y_true, y_pred, zero_division=0),
            roc_auc=roc_auc,
            confusion_matrix=confusion_matrix(y_true, y_pred),
            classification_report=classification_report(y_true, y_pred, target_names=['Malignant', 'Benign']),
            training_samples=len(self.X_train),
            test_samples=len(self.X_test),
            cv_scores=cv_scores or {},
            hyperparameters=hyperparams or {},
            training_time=training_time
        )
        
        return metrics
    
    def train_baseline_logistic_regression(self) -> ModelMetrics:
        """
        Train baseline Logistic Regression model with StandardScaler.
        
        Returns:
            ModelMetrics with evaluation results
        """
        logger.info("\n" + "="*60)
        logger.info("TRAINING BASELINE: Logistic Regression")
        logger.info("="*60)
        
        import time
        start_time = time.time()
        
        # Create pipeline with scaling + LR
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                max_iter=1000, 
                solver='lbfgs',
                random_state=self.random_state,
                class_weight='balanced'
            ))
        ])
        
        # Train model
        logger.info("Fitting Logistic Regression...")
        pipeline.fit(self.X_train, self.y_train)
        training_time = time.time() - start_time
        
        # Cross-validation
        logger.info("Performing 5-fold cross-validation...")
        cv_scores_acc = cross_val_score(pipeline, self.X_train, self.y_train, cv=5, scoring='accuracy')
        cv_scores_f1 = cross_val_score(pipeline, self.X_train, self.y_train, cv=5, scoring='f1')
        
        # Predictions
        y_pred = pipeline.predict(self.X_test)
        
        # Evaluate
        metrics = self.evaluate_model(
            name="Logistic Regression (Baseline)",
            pipeline=pipeline,
            y_true=self.y_test,
            y_pred=y_pred,
            training_time=training_time,
            cv_scores={
                "cv_accuracy_mean": cv_scores_acc.mean(),
                "cv_accuracy_std": cv_scores_acc.std(),
                "cv_f1_mean": cv_scores_f1.mean(),
                "cv_f1_std": cv_scores_f1.std(),
            },
            hyperparams={
                "max_iter": 1000,
                "solver": "lbfgs",
                "class_weight": "balanced"
            }
        )
        
        self.models["baseline_lr"] = pipeline
        self.results["Logistic Regression (Baseline)"] = metrics
        
        logger.info(f"Training completed in {training_time:.2f}s")
        logger.info(f"Test Accuracy: {metrics.accuracy:.4f}")
        logger.info(f"Test F1-Score: {metrics.f1_score:.4f}")
        
        return metrics
    
    def train_optimized_svm(self) -> ModelMetrics:
        """
        Train optimized SVM with RBF kernel using GridSearchCV.
        
        Returns:
            ModelMetrics with evaluation results
        """
        logger.info("\n" + "="*60)
        logger.info("TRAINING OPTIMIZED MODEL: Support Vector Machine")
        logger.info("="*60)
        
        import time
        start_time = time.time()
        
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel='rbf', random_state=self.random_state, class_weight='balanced'))
        ])
        
        # Define hyperparameter grid
        param_grid = {
            'classifier__C': [0.1, 0.5, 1, 2, 5, 10],
            'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        }
        
        logger.info(f"Performing GridSearchCV with {len(param_grid['classifier__C']) * len(param_grid['classifier__gamma'])} combinations...")
        logger.info(f"Parameter grid: C={param_grid['classifier__C']}, gamma={param_grid['classifier__gamma']}")
        
        # GridSearch with cross-validation
        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        
        # Train
        grid_search.fit(self.X_train, self.y_train)
        training_time = time.time() - start_time
        
        best_pipeline = grid_search.best_estimator_
        
        logger.info(f"Best parameters found: {grid_search.best_params_}")
        logger.info(f"Best CV F1-Score: {grid_search.best_score_:.4f}")
        
        # Predictions
        y_pred = best_pipeline.predict(self.X_test)
        
        # Evaluate
        metrics = self.evaluate_model(
            name="SVM (Optimized with GridSearch)",
            pipeline=best_pipeline,
            y_true=self.y_test,
            y_pred=y_pred,
            training_time=training_time,
            cv_scores={
                "best_cv_f1_score": grid_search.best_score_,
                "cv_mean_test_score": grid_search.cv_results_['mean_test_score'].mean(),
                "cv_std_test_score": grid_search.cv_results_['std_test_score'].mean(),
            },
            hyperparams=grid_search.best_params_
        )
        
        self.models["optimized_svm"] = best_pipeline
        self.results["SVM (Optimized with GridSearch)"] = metrics
        
        logger.info(f"Training completed in {training_time:.2f}s")
        logger.info(f"Test Accuracy: {metrics.accuracy:.4f}")
        logger.info(f"Test F1-Score: {metrics.f1_score:.4f}")
        
        return metrics



        logger.info(f"Training completed in {training_time:.2f}s")
        logger.info(f"Test Accuracy: {metrics.accuracy:.4f}")
        logger.info(f"Test F1-Score: {metrics.f1_score:.4f}")
        
        return metrics
    
    def print_comparative_report(self) -> None:
        """Print comprehensive comparative analysis of all models."""
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE MODEL COMPARISON REPORT")
        logger.info("="*80)
        
        # Create comparison table
        print("\n┌" + "─"*78 + "┐")
        print("│{:^78}│".format("PERFORMANCE METRICS"))
        print("├" + "─"*78 + "┤")
        print("│ {:<35} │ {:>18} │ {:>18} │".format("Metric", "Baseline LR", "Optimized SVM"))
        print("├" + "─"*78 + "┤")
        
        lr_metrics = self.results.get("Logistic Regression (Baseline)")
        svm_metrics = self.results.get("SVM (Optimized with GridSearch)")
        
        if lr_metrics and svm_metrics:
            metrics_to_compare = [
                ("Accuracy", "accuracy"),
                ("Precision", "precision"),
                ("Recall", "recall"),
                ("F1-Score", "f1_score"),
                ("ROC-AUC", "roc_auc"),
            ]
            
            for label, attr in metrics_to_compare:
                lr_val = getattr(lr_metrics, attr)
                svm_val = getattr(svm_metrics, attr)
                winner = "✓" if svm_val > lr_val else ("✓" if lr_val > svm_val else "=")
                
                print("│ {:<35} │ {:>18.4f} │ {:>17.4f}{} │".format(
                    label, lr_val, svm_val, winner
                ))
            
            print("├" + "─"*78 + "┤")
            print("│ {:<35} │ {:>18.2f}s │ {:>17.2f}s │".format(
                "Training Time", lr_metrics.training_time, svm_metrics.training_time
            ))
            print("└" + "─"*78 + "┘")
            
            # Confusion matrices
            print("\n" + "="*80)
            print("CONFUSION MATRICES")
            print("="*80)
            
            for name, metrics in self.results.items():
                print(f"\n{name}:")
                print("                Predicted")
                print("              Malignant  Benign")
                cm = metrics.confusion_matrix
                print(f"Actual  Malignant    {cm[0,0]:>3}      {cm[0,1]:>3}")
                print(f"        Benign       {cm[1,0]:>3}      {cm[1,1]:>3}")
                
                # Calculate specifics
                tn, fp, fn, tp = cm.ravel()
                print(f"\n  True Negatives (TN): {tn}  |  False Positives (FP): {fp}")
                print(f"  False Negatives (FN): {fn}  |  True Positives (TP): {tp}")
            
            # Classification reports
            print("\n" + "="*80)
            print("DETAILED CLASSIFICATION REPORTS")
            print("="*80)
            
            for name, metrics in self.results.items():
                print(f"\n{name}:")
                print(metrics.classification_report)
    
    def save_results(self, output_path: str = "results.json") -> None:
        """
        Save all results to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        output_data = {
            "experiment_metadata": {
                "timestamp": datetime.now().isoformat(),
                "dataset": "Wisconsin Breast Cancer (Diagnostic)",
                "test_size": self.test_size,
                "random_state": self.random_state,
                "total_samples": len(self.X_train) + len(self.X_test),
            },
            "models": {
                name: metrics.to_dict() 
                for name, metrics in self.results.items()
            }
        }
        
        output_file = Path(output_path)
        with output_file.open('w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"\n✓ Results saved to: {output_file.absolute()}")


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("{:^80}".format("BREAST CANCER CLASSIFICATION SYSTEM"))
    print("{:^80}".format("Systematic Comparison: Logistic Regression vs SVM"))
    print("="*80)
    
    # Initialize classifier
    classifier = BreastCancerClassifier(test_size=0.2, random_state=42)
    
    # Load and prepare data
    X, y = classifier.load_data()
    classifier.prepare_data(X, y)
    
    # Train baseline model
    lr_metrics = classifier.train_baseline_logistic_regression()
    
    # Train optimized model
    svm_metrics = classifier.train_optimized_svm()
    
    # Print comparative report
    classifier.print_comparative_report()
    
    # Save results
    classifier.save_results("results.json")
    
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
    logger.info("="*80)


if __name__ == "__main__":
    main()
