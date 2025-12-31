# 🧬 Breast Cancer Diagnosis Mini Project

Predict malignant vs benign tumors on the **Breast Cancer Wisconsin (Diagnostic)** dataset using supervised machine learning (baseline Logistic Regression vs. tuned SVM), with clear steps for data prep, modeling, and evaluation.

---

## 1. Introduction
Breast cancer remains a leading cause of death among women. Early, accurate diagnosis can save lives. This mini project explores bioinformatics techniques with supervised ML on the Wisconsin Breast Cancer (Diagnostic) dataset, using geometric properties of cell nuclei (radius, texture, concavity, etc.) to classify tumors as **malignant** or **benign**.

## 1.1 Problem Background
Manual reading of Fine Needle Aspirate (FNA) images is labor-intensive and prone to human fatigue. Automated systems can process large amounts of data and uncover subtle patterns, offering clinicians a decision-support tool that complements traditional diagnostics.

## 1.2 Problem Statement
Build an ML model that predicts malignancy/benignity from 10 geometric features of digitized cell nuclei. Start with a Logistic Regression baseline, then close the accuracy gap using a tuned Support Vector Machine (SVM) with feature scaling and hyperparameter search (GridSearch).

## 1.3 Objectives
- Use **Logistic Regression** as a baseline for binary breast cancer classification.
- Train an **SVM** with feature scaling and **GridSearch** hyperparameter tuning.
- Compare models via **Accuracy, Precision, Recall, F1-score**, and discuss results.

## 1.4 Scope
- **Dataset:** Breast Cancer Wisconsin (Diagnostic) – 569 samples, 30 numeric features (Kaggle / scikit-learn built-in).
- **Focus:** Classification and identifying informative geometric features as biomarkers.
- **Tools:** Python, pandas, scikit-learn, matplotlib; runnable in Jupyter/Colab or locally.
- **Versioning:** Code/notebooks intended for GitHub with regular commits/branches.

## 1.5 Conclusion (Goal)
Demonstrate the impact of model optimization in bioinformatics by contrasting a simple baseline with a tuned model, aiming for reliable malignant/benign predictions that can assist clinicians.

---

## 2. Project Structure
```
PFB Project/
├─ app.py               # (Placeholder) main app/entry point
├─ README.md            # This document
└─ requirements.txt     # (Add if needed) Python dependencies
```
Add notebooks/scripts as you develop (e.g., `notebooks/eda.ipynb`, `src/train.py`).

---

## 3. Setup & Installation
```bash
# create/activate your environment (example)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# install dependencies (add to requirements.txt as you go)
pip install pandas scikit-learn matplotlib seaborn jupyter
```

If using the scikit-learn built-in dataset, no download is required. If using Kaggle, place the CSV under `data/` and adjust the load path in your notebook/script.

---

## 4. Data
- **Source:** Breast Cancer Wisconsin (Diagnostic). Available via scikit-learn (`load_breast_cancer()`) or Kaggle.
- **Target:** `diagnosis` / `target` (0 = malignant, 1 = benign in sklearn convention).
- **Features:** 30 numeric attributes; we may subset to 10 key geometric means for a simpler UI, or use all 30 for best accuracy.

---

## 5. Modeling Plan
1) **Baseline:** Logistic Regression (with standardization). Report metrics and confusion matrix.
2) **Tuned Model:** SVM with RBF kernel.
   - Apply **StandardScaler**.
   - Hyperparameters via **GridSearchCV** (e.g., `C`, `gamma`).
3) **Evaluation:**
   - Metrics: Accuracy, Precision, Recall, F1.
   - Confusion matrix visualization.
   - Optional: ROC-AUC and curves.
4) **Comparison:** Summarize which model performs best and why.

---

## 6. Suggested Workflow
- **EDA:** Inspect class balance, feature distributions, correlations.
- **Preprocess:** Handle scaling; consider train/validation split.
- **Train:** Baseline Logistic Regression, then SVM with grid search.
- **Evaluate:** Collect metrics on held-out test data; visualize confusion matrix.
- **Document:** Record results in the notebook/README.

---

## 7. How to Run (example notebook flow)
1. Open Jupyter/Colab and run the notebook cells in order.
2. Modify hyperparameter grids in the SVM section to experiment.
3. View printed metrics and plots (confusion matrix, optional ROC).

For a script-based flow, create `train.py` that loads data, trains both models, and prints metrics.

---

## 8. Results (to fill in as you train)
| Model | Accuracy | Precision | Recall | F1 | Notes |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | _TBD_ | _TBD_ | _TBD_ | _TBD_ | Baseline |
| SVM (tuned) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | After GridSearch |

---

## 9. Future Improvements
- Try tree ensembles (Random Forest, XGBoost) for feature importance.
- Add calibration for probability outputs.
- Add cross-validation with stratification.
- Build a small Streamlit UI for clinicians (input features → prediction).

---

## 10. References
- [Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- [scikit-learn `load_breast_cancer`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- [Support Vector Machines in scikit-learn](https://scikit-learn.org/stable/modules/svm.html)

---

## 11. Author
**TAY CHING XIAN** — Bioinformatics Mini Project.
