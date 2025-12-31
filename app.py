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
matplotlib.use('Agg')  # Use non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import seaborn as sns


class BreastCancerClassifier:
    """
    Main class for breast cancer classification using ML models
    """
    
    def __init__(self, data_path='breast_cancer_data.csv'):
        """
        Initialize the classifier
        
        Args:
            data_path (str): Path to the breast cancer dataset CSV
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.lr_model = None
        self.svm_model = None
        self.results = {}
        
    def load_data(self):
        """
        Load and perform initial exploration of the dataset
        """
        print("=" * 80)
        print("LOADING WISCONSIN BREAST CANCER DATASET")
        print("=" * 80)
        
        # Load dataset
        self.df = pd.read_csv(self.data_path)
        
        print(f"\n📊 Dataset Shape: {self.df.shape}")
        print(f"   - Samples: {self.df.shape[0]}")
        print(f"   - Features: {self.df.shape[1] - 2}")  # Excluding ID and diagnosis
        
        # Display basic info
        print("\n📋 Dataset Info:")
        print(f"   - Columns: {list(self.df.columns[:5])}... (showing first 5)")
        print(f"   - Data types: {self.df.dtypes.value_counts().to_dict()}")
        
        # Check for missing values
        missing = self.df.isnull().sum().sum()
        print(f"\n🔍 Missing Values: {missing}")
        
        # Target distribution
        diagnosis_counts = self.df['diagnosis'].value_counts()
        print(f"\n🎯 Target Distribution:")
        print(f"   - Malignant (M): {diagnosis_counts.get('M', 0)} ({diagnosis_counts.get('M', 0)/len(self.df)*100:.1f}%)")
        print(f"   - Benign (B): {diagnosis_counts.get('B', 0)} ({diagnosis_counts.get('B', 0)/len(self.df)*100:.1f}%)")
        
        return self.df
    
    def preprocess_data(self, test_size=0.2, random_state=42):
        """
        Preprocess the data: encode labels, split, and scale features
        
        Args:
            test_size (float): Proportion of test set
            random_state (int): Random seed for reproducibility
        """
        print("\n" + "=" * 80)
        print("PREPROCESSING DATA")
        print("=" * 80)
        
        # Separate features and target
        X = self.df.drop(['id', 'diagnosis'], axis=1)
        y = self.df['diagnosis'].map({'M': 1, 'B': 0})  # Malignant=1, Benign=0
        
        print(f"\n✂️ Splitting data: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"   - Training set: {self.X_train.shape[0]} samples")
        print(f"   - Test set: {self.X_test.shape[0]} samples")
        
        # Feature scaling (critical for SVM)
        print(f"\n📏 Applying StandardScaler for feature normalization")
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"   - Features scaled to mean=0, std=1")
        print(f"   - Original range example: [{self.X_train.iloc[:, 0].min():.2f}, {self.X_train.iloc[:, 0].max():.2f}]")
        print(f"   - Scaled range example: [{self.X_train_scaled[:, 0].min():.2f}, {self.X_train_scaled[:, 0].max():.2f}]")
        
    def train_baseline_model(self):
        """
        Train baseline Logistic Regression model
        """
        print("\n" + "=" * 80)
        print("TRAINING BASELINE MODEL: LOGISTIC REGRESSION")
        print("=" * 80)
        
        # Initialize and train
        self.lr_model = LogisticRegression(random_state=42, max_iter=10000)
        
        print("\n🔧 Model Configuration:")
        print(f"   - Algorithm: Logistic Regression")
        print(f"   - Solver: lbfgs (default)")
        print(f"   - Max iterations: 10000")
        
        print("\n⏳ Training model...")
        self.lr_model.fit(self.X_train_scaled, self.y_train)
        print("✅ Training complete!")
        
        # Cross-validation
        print("\n🔄 Performing 5-fold cross-validation...")
        cv_scores = cross_val_score(self.lr_model, self.X_train_scaled, self.y_train, cv=5)
        print(f"   - CV Scores: {[f'{score:.4f}' for score in cv_scores]}")
        print(f"   - Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
    def train_svm_model(self):
        """
        Train optimized SVM model with GridSearchCV
        """
        print("\n" + "=" * 80)
        print("TRAINING OPTIMIZED MODEL: SUPPORT VECTOR MACHINE")
        print("=" * 80)
        
        # Define parameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'linear', 'poly']
        }
        
        print("\n🔧 Hyperparameter Search Space:")
        for param, values in param_grid.items():
            print(f"   - {param}: {values}")
        print(f"\n   Total combinations: {np.prod([len(v) for v in param_grid.values()])}")
        
        # Initialize GridSearchCV
        print("\n⏳ Running GridSearchCV (5-fold CV)...")
        print("   This may take a few minutes...")
        
        grid_search = GridSearchCV(
            SVC(random_state=42, probability=True),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        print("✅ Grid search complete!")
        print(f"\n🎯 Best Parameters Found:")
        for param, value in grid_search.best_params_.items():
            print(f"   - {param}: {value}")
        print(f"\n   Best CV Score: {grid_search.best_score_:.4f}")
        
        # Store best model
        self.svm_model = grid_search.best_estimator_
        
    def evaluate_models(self):
        """
        Comprehensive evaluation of both models
        """
        print("\n" + "=" * 80)
        print("MODEL EVALUATION & COMPARISON")
        print("=" * 80)
        
        models = {
            'Logistic Regression': self.lr_model,
            'SVM (Optimized)': self.svm_model
        }
        
        self.results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_info': {
                'total_samples': len(self.df),
                'train_samples': len(self.X_train),
                'test_samples': len(self.X_test),
                'num_features': self.X_train.shape[1],
                'malignant_count': int(self.y_train.sum() + self.y_test.sum()),
                'benign_count': int(len(self.df) - (self.y_train.sum() + self.y_test.sum()))
            },
            'models': {}
        }
        
        for model_name, model in models.items():
            print(f"\n{'─' * 80}")
            print(f"📊 {model_name}")
            print('─' * 80)
            
            # Predictions
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            # Specificity
            specificity = tn / (tn + fp)
            
            # Print metrics
            print(f"\n✨ Performance Metrics:")
            print(f"   • Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"   • Precision:   {precision:.4f} ({precision*100:.2f}%)")
            print(f"   • Recall:      {recall:.4f} ({recall*100:.2f}%)")
            print(f"   • F1-Score:    {f1:.4f}")
            print(f"   • ROC-AUC:     {roc_auc:.4f}")
            print(f"   • Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
            
            print(f"\n📋 Confusion Matrix:")
            print(f"   {'':12} Predicted B    Predicted M")
            print(f"   Actual B    {tn:6d}         {fp:6d}")
            print(f"   Actual M    {fn:6d}         {tp:6d}")
            
            print(f"\n🔍 Clinical Interpretation:")
            print(f"   • True Negatives (TN):  {tn} - Correctly identified benign")
            print(f"   • True Positives (TP):  {tp} - Correctly identified malignant")
            print(f"   • False Positives (FP): {fp} - Benign classified as malignant")
            print(f"   • False Negatives (FN): {fn} - Malignant classified as benign ⚠️")
            
            # Store results
            self.results['models'][model_name] = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'roc_auc': float(roc_auc),
                'specificity': float(specificity),
                'confusion_matrix': {
                    'true_negatives': int(tn),
                    'false_positives': int(fp),
                    'false_negatives': int(fn),
                    'true_positives': int(tp)
                }
            }
        
        # Comparison
        print(f"\n{'=' * 80}")
        print("📊 MODEL COMPARISON SUMMARY")
        print('=' * 80)
        
        lr_acc = self.results['models']['Logistic Regression']['accuracy']
        svm_acc = self.results['models']['SVM (Optimized)']['accuracy']
        improvement = (svm_acc - lr_acc) * 100
        
        print(f"\n🎯 Accuracy Improvement:")
        print(f"   Logistic Regression: {lr_acc:.4f} ({lr_acc*100:.2f}%)")
        print(f"   SVM (Optimized):     {svm_acc:.4f} ({svm_acc*100:.2f}%)")
        print(f"   Improvement:         {improvement:+.2f}%")
        
        lr_f1 = self.results['models']['Logistic Regression']['f1_score']
        svm_f1 = self.results['models']['SVM (Optimized)']['f1_score']
        
        print(f"\n🎯 F1-Score Comparison:")
        print(f"   Logistic Regression: {lr_f1:.4f}")
        print(f"   SVM (Optimized):     {svm_f1:.4f}")
        print(f"   Improvement:         {(svm_f1 - lr_f1)*100:+.2f}%")
        
        # Winner
        winner = 'SVM (Optimized)' if svm_acc > lr_acc else 'Logistic Regression'
        print(f"\n🏆 Best Model: {winner}")
        
    def save_results(self, filename='results.json'):
        """
        Save results to JSON file
        
        Args:
            filename (str): Output filename
        """
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"\n💾 Results saved to: {filename}")
        
    def visualize_results(self):
        """
        Create comprehensive visualizations
        """
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Breast Cancer Classification - Model Comparison', fontsize=16, fontweight='bold')
        
        models = {
            'Logistic Regression': self.lr_model,
            'SVM (Optimized)': self.svm_model
        }
        
        # 1. Performance Metrics Comparison
        ax1 = axes[0, 0]
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        lr_values = [
            self.results['models']['Logistic Regression']['accuracy'],
            self.results['models']['Logistic Regression']['precision'],
            self.results['models']['Logistic Regression']['recall'],
            self.results['models']['Logistic Regression']['f1_score']
        ]
        svm_values = [
            self.results['models']['SVM (Optimized)']['accuracy'],
            self.results['models']['SVM (Optimized)']['precision'],
            self.results['models']['SVM (Optimized)']['recall'],
            self.results['models']['SVM (Optimized)']['f1_score']
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
            [self.results['models']['Logistic Regression']['confusion_matrix']['true_negatives'],
             self.results['models']['Logistic Regression']['confusion_matrix']['false_positives']],
            [self.results['models']['Logistic Regression']['confusion_matrix']['false_negatives'],
             self.results['models']['Logistic Regression']['confusion_matrix']['true_positives']]
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
            [self.results['models']['SVM (Optimized)']['confusion_matrix']['true_negatives'],
             self.results['models']['SVM (Optimized)']['confusion_matrix']['false_positives']],
            [self.results['models']['SVM (Optimized)']['confusion_matrix']['false_negatives'],
             self.results['models']['SVM (Optimized)']['confusion_matrix']['true_positives']]
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
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc = roc_auc_score(self.y_test, y_pred_proba)
            ax4.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.3f})', linewidth=2)
        
        ax4.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        ax4.set_xlabel('False Positive Rate')
        ax4.set_ylabel('True Positive Rate')
        ax4.set_title('ROC Curves Comparison')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("\n📊 Visualization saved to: model_comparison.png")
        
    def run_pipeline(self):
        """
        Execute the complete analysis pipeline
        """
        print("\n" + "🧬" * 40)
        print("BREAST CANCER CLASSIFICATION - BIOINFORMATICS PROJECT")
        print("Wisconsin Breast Cancer (Diagnostic) Dataset Analysis")
        print("🧬" * 40)
        
        # Execute pipeline
        self.load_data()
        self.preprocess_data()
        self.train_baseline_model()
        self.train_svm_model()
        self.evaluate_models()
        self.save_results()
        self.visualize_results()
        
        print("\n" + "=" * 80)
        print("✅ PIPELINE COMPLETE!")
        print("=" * 80)
        print("\n📁 Generated Files:")
        print("   • results.json - Detailed metrics and model comparison")
        print("   • model_comparison.png - Visualization of results")
        print("\n💡 Next Steps:")
        print("   • Review the results.json for detailed metrics")
        print("   • Examine model_comparison.png for visual insights")
        print("   • Run 'streamlit run streamlit_app.py' for interactive demo")
        print("\n" + "=" * 80 + "\n")


def main():
    """
    Main execution function
    """
    # Initialize classifier
    classifier = BreastCancerClassifier(data_path='breast_cancer_data.csv')
    
    # Run complete pipeline
    classifier.run_pipeline()


if __name__ == "__main__":
    main()
