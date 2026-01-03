"""Breast cancer classification pipeline.

This module contains the core ML logic (no Streamlit dependency), so it can be
imported by both a CLI script and a Streamlit demo.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import json
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
)


def _specificity_from_cm(cm: np.ndarray) -> float:
    # cm is [[tn, fp], [fn, tp]]
    tn, fp = cm[0, 0], cm[0, 1]
    denom = tn + fp
    return float(tn / denom) if denom else 0.0


@dataclass
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5

    # Clinical-ish knobs
    positive_label: int = 1  # 1 = malignant
    target_sensitivity: float = 0.95  # aim to catch cancers
    calibrate_probabilities: bool = True
    threshold_strategy: str = "target_sensitivity"  # or "youden" or "fixed"
    fixed_threshold: float = 0.5


class BreastCancerClassifier:
    """Train and evaluate Logistic Regression vs tuned SVM."""

    def __init__(self, data_path: str | Path = "breast_cancer_data.csv"):
        self.data_path = Path(data_path)

        self.df: Optional[pd.DataFrame] = None
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None

        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None

        # Keep raw dataframes for patient-level predictions
        self.feature_names: Optional[List[str]] = None

        # Models are pipelines to avoid leakage.
        self.lr_model: Optional[Any] = None
        self.svm_model: Optional[Any] = None
        self.svm_grid: Optional[GridSearchCV] = None

        # Thresholds chosen on training data only
        self.lr_threshold_: float = 0.5
        self.svm_threshold_: float = 0.5

        self.results: Dict[str, Any] = {
            "run_info": {},
            "dataset_info": {},
            "models": {},
        }

    def load_data(self) -> pd.DataFrame:
        if not self.data_path.exists():
            data = load_breast_cancer()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df.insert(0, "id", np.arange(1, len(df) + 1))
            df.insert(1, "diagnosis", pd.Series(data.target).map({0: "M", 1: "B"}))
            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.data_path, index=False)
        else:
            df = pd.read_csv(self.data_path)
        self.df = df

        # Common Kaggle format: id, diagnosis, features...
        if "diagnosis" not in df.columns:
            raise ValueError("Expected a 'diagnosis' column in the dataset.")

        feature_cols = [c for c in df.columns if c not in {"id", "diagnosis"}]
        if not feature_cols:
            raise ValueError("No feature columns found (expected columns besides id/diagnosis).")

        self.X = df[feature_cols]
        self.feature_names = feature_cols
        # Map M/B to 1/0 for positive class = malignant
        self.y = df["diagnosis"].map({"B": 0, "M": 1})
        if self.y.isna().any():
            raise ValueError("Unexpected label values in 'diagnosis' (expected 'B'/'M').")

        self.results["run_info"] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "data_path": str(self.data_path),
        }
        self.results["dataset_info"] = {
            "total_samples": int(df.shape[0]),
            "num_features": int(self.X.shape[1]),
            "class_distribution": {
                "benign": int((self.y == 0).sum()),
                "malignant": int((self.y == 1).sum()),
            },
        }

        return df

    def preprocess_data(self, test_size: float = 0.2, random_state: int = 42) -> None:
        if self.X is None or self.y is None:
            raise RuntimeError("Call load_data() before preprocess_data().")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y,
        )

        self.results["dataset_info"].update(
            {
                "train_samples": int(self.X_train.shape[0]),
                "test_samples": int(self.X_test.shape[0]),
                "test_size": float(test_size),
                "random_state": int(random_state),
            }
        )

    def _build_lr_pipeline(self) -> Pipeline:
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000, solver="lbfgs")),
            ]
        )

    def _build_svm_pipeline(self) -> Pipeline:
        # probability=True gives predict_proba for ROC/AUC and calibration.
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", probability=True)),
            ]
        )

    def _maybe_calibrate(self, estimator: Any, config: TrainConfig) -> Any:
            # Calibrate on training folds only (wrapped CV). This is slow-ish but realistic.
            if not config.calibrate_probabilities:
                return estimator
            cv = StratifiedKFold(n_splits=int(config.cv_folds), shuffle=True, random_state=int(config.random_state))
            return CalibratedClassifierCV(estimator, method="isotonic", cv=cv)

    def _choose_threshold(self, y_true: np.ndarray, y_score: np.ndarray, config: TrainConfig) -> float:
        """Choose a decision threshold from training scores only.

        Strategies:
        - target_sensitivity: smallest threshold with recall >= target
        - youden: maximize TPR - FPR
        - fixed: config.fixed_threshold
        """
        if config.threshold_strategy == "fixed":
            return float(config.fixed_threshold)

        # guard
        if y_score.ndim != 1:
            y_score = y_score.ravel()

        if config.threshold_strategy == "youden":
            fpr, tpr, thr = roc_curve(y_true, y_score)
            j = tpr - fpr
            idx = int(np.nanargmax(j))
            return float(thr[idx])

        # default: target sensitivity
        precision, recall, thr = precision_recall_curve(y_true, y_score)
        # precision_recall_curve returns thresholds for points excluding last
        # We'll scan thresholds and pick the *highest* threshold that still meets sensitivity.
        if thr.size == 0:
            return 0.5

        # Compute recall at each threshold. PR curve returns recall values aligned with thresholds.
        # recall has length thr+1; use recall[:-1]
        recall_at_thr = recall[:-1]
        ok = np.where(recall_at_thr >= float(config.target_sensitivity))[0]
        if ok.size == 0:
            # Can't meet the target sensitivity; pick threshold that maximizes recall.
            idx = int(np.nanargmax(recall_at_thr))
            return float(thr[idx])
        return float(thr[ok[-1]])

    def train_baseline_model(self, config: TrainConfig = TrainConfig()) -> Any:
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Call preprocess_data() before training.")

        base = self._build_lr_pipeline()
        self.lr_model = self._maybe_calibrate(base, config)
        self.lr_model.fit(self.X_train, self.y_train)

        # pick threshold on training set
        y_score_train = self.predict_proba(self.lr_model, self.X_train)
        self.lr_threshold_ = self._choose_threshold(self.y_train.to_numpy(), y_score_train, config)
        return self.lr_model

    def train_svm_model(self, config: TrainConfig = TrainConfig()) -> Any:
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Call preprocess_data() before training.")

        # More realistic scoring than pure accuracy (imbalance-aware).
        # Keep grid modest for Streamlit responsiveness.
        param_grid = {
            "model__C": [0.5, 1, 5, 10, 50],
            "model__gamma": ["scale", 0.01, 0.1, 1],
        }

        base = self._build_svm_pipeline()
        cv = StratifiedKFold(n_splits=int(config.cv_folds), shuffle=True, random_state=int(config.random_state))
        self.svm_grid = GridSearchCV(
            base,
            param_grid,
            cv=cv,
            n_jobs=-1,
            scoring="roc_auc",
            refit=True,
        )
        self.svm_grid.fit(self.X_train, self.y_train)

        best = self.svm_grid.best_estimator_
        self.svm_model = self._maybe_calibrate(best, config)
        self.svm_model.fit(self.X_train, self.y_train)

        y_score_train = self.predict_proba(self.svm_model, self.X_train)
        self.svm_threshold_ = self._choose_threshold(self.y_train.to_numpy(), y_score_train, config)
        return self.svm_model

    @staticmethod
    def predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        if hasattr(model, "decision_function"):
            # fallback (shouldn't happen with our choices)
            scores = model.decision_function(X)
            # min-max to 0-1 just for display
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
            return scores
        raise TypeError("Model does not support probability or decision scores.")

    def _evaluate_one(self, model_name: str, model: Any, threshold: float) -> Dict[str, Any]:
        if self.X_test is None or self.y_test is None:
            raise RuntimeError("Call preprocess_data() before evaluation.")

        y_score = self.predict_proba(model, self.X_test)
        y_pred = (y_score >= float(threshold)).astype(int)
        cm = confusion_matrix(self.y_test, y_pred)

        metrics: Dict[str, Any] = {
            "accuracy": float(accuracy_score(self.y_test, y_pred)),
            "precision": float(precision_score(self.y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(self.y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(self.y_test, y_pred, zero_division=0)),
            "avg_precision": float(average_precision_score(self.y_test, y_score)),
            "specificity": float(_specificity_from_cm(cm)),
            "threshold": float(threshold),
            "confusion_matrix": {
                "true_negatives": int(cm[0, 0]),
                "false_positives": int(cm[0, 1]),
                "false_negatives": int(cm[1, 0]),
                "true_positives": int(cm[1, 1]),
            },
            "classification_report": classification_report(
                self.y_test,
                y_pred,
                target_names=["benign", "malignant"],
                zero_division=0,
                output_dict=True,
            ),
        }

        metrics["roc_auc"] = float(roc_auc_score(self.y_test, y_score))
        metrics["brier_score"] = float(brier_score_loss(self.y_test, y_score))

        return metrics

    def evaluate_models(self) -> Dict[str, Any]:
        if self.lr_model is None or self.svm_model is None:
            raise RuntimeError("Train both models before evaluate_models().")

        self.results["models"] = {
            "Logistic Regression": self._evaluate_one(
                "Logistic Regression", self.lr_model, threshold=self.lr_threshold_
            ),
            "SVM (Optimized)": self._evaluate_one(
                "SVM (Optimized)", self.svm_model, threshold=self.svm_threshold_
            ),
        }

        if self.svm_grid is not None:
            self.results["models"]["SVM (Optimized)"]["best_params"] = self.svm_grid.best_params_

        return self.results

    def top_features_lr(self, top_k: int = 10) -> Dict[str, float]:
        """Return top positive coefficients from LR pipeline for quick interpretability."""
        if self.lr_model is None or self.feature_names is None:
            return {}

        model = self.lr_model
        # unwrap calibration
        if hasattr(model, "base_estimator"):
            model = model.base_estimator

        if not hasattr(model, "named_steps"):
            return {}

        lr = model.named_steps.get("model")
        if lr is None or not hasattr(lr, "coef_"):
            return {}

        coefs = lr.coef_.ravel()
        pairs = list(zip(self.feature_names, coefs))
        # For malignant=1, positive coef increases risk
        pairs.sort(key=lambda x: x[1], reverse=True)
        return {k: float(v) for k, v in pairs[: int(top_k)]}

    def predict_patient(self, features: Dict[str, float], model_name: str = "SVM (Optimized)") -> Dict[str, Any]:
        """Predict malignancy probability for a single patient-like input."""
        if self.feature_names is None:
            raise RuntimeError("Call load_data() first to get feature schema.")

        missing = [f for f in self.feature_names if f not in features]
        if missing:
            raise ValueError(f"Missing feature(s): {missing[:5]}" + ("..." if len(missing) > 5 else ""))

        X_one = pd.DataFrame([{f: float(features[f]) for f in self.feature_names}])

        if model_name == "Logistic Regression":
            if self.lr_model is None:
                raise RuntimeError("Train the Logistic Regression model first.")
            p = float(self.predict_proba(self.lr_model, X_one)[0])
            thr = float(self.lr_threshold_)
        else:
            if self.svm_model is None:
                raise RuntimeError("Train the SVM model first.")
            p = float(self.predict_proba(self.svm_model, X_one)[0])
            thr = float(self.svm_threshold_)

        pred = int(p >= thr)
        return {
            "model": model_name,
            "prob_malignant": p,
            "threshold": thr,
            "predicted_label": pred,
            "predicted_class": "malignant" if pred == 1 else "benign",
        }

    def save_results(self, out_path: str | Path = "results.json") -> Path:
        out_path = Path(out_path)
        out_path.write_text(json.dumps(self.results, indent=4), encoding="utf-8")
        return out_path

    def run(self, config: TrainConfig = TrainConfig()) -> Dict[str, Any]:
        self.load_data()
        self.preprocess_data(test_size=config.test_size, random_state=config.random_state)
        self.train_baseline_model(config=config)
        self.train_svm_model(config=config)
        self.evaluate_models()
        self.results["training_config"] = {
            "test_size": float(config.test_size),
            "random_state": int(config.random_state),
            "cv_folds": int(config.cv_folds),
            "calibrate_probabilities": bool(config.calibrate_probabilities),
            "threshold_strategy": str(config.threshold_strategy),
            "target_sensitivity": float(config.target_sensitivity),
            "fixed_threshold": float(config.fixed_threshold),
        }

        self.results["interpretability"] = {
            "lr_top_positive_coefficients": self.top_features_lr(top_k=10)
        }
        return self.results
