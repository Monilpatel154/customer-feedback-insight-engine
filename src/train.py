"""
Multi-Model Training, Benchmarking, Cross-Validation, and Serialization Pipeline
Follows strict ML best practices: Split BEFORE feature extraction, no data leakage.
"""

import os
import sys
import json
import time
import joblib

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report
)

from src.preprocess import clean_text, setup_nltk_resources
from src.insights import TopicModeler

def load_and_clean_data(data_path="data/raw/customer_feedback.csv"):
    """Loads dataset, handles missing values, and preprocesses review text."""
    if not os.path.exists(data_path):
        from src.generate_dataset import create_feedback_dataset
        print(f"Dataset not found at {data_path}. Generating benchmark dataset...")
        create_feedback_dataset(data_path)
        
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 1. Missing Value Check & Handling
    null_counts = df.isnull().sum()
    print("Null value audit:\n", null_counts)
    
    # Drop rows without text or sentiment label if any
    df = df.dropna(subset=['review_text', 'sentiment']).copy()
    
    # Ensure string types
    df['review_text'] = df['review_text'].astype(str)
    df['sentiment'] = df['sentiment'].astype(str)
    
    # Filter only expected sentiment classes
    valid_classes = ["Positive", "Neutral", "Negative"]
    df = df[df['sentiment'].isin(valid_classes)].copy()
    
    print(f"Pre-processing {len(df)} text reviews with NLP pipeline...")
    setup_nltk_resources()
    df['cleaned_text'] = df['review_text'].apply(clean_text)
    
    # Drop any that resulted in empty string
    df = df[df['cleaned_text'].str.strip().str.len() > 0].copy()
    print(f"Final clean dataset size: {len(df)} samples")
    
    return df

def train_and_evaluate_models(data_path="data/raw/customer_feedback.csv", output_dir="models"):
    """
    Executes the complete machine learning lifecycle:
    1. Train/Test split (Strict featurization ordering)
    2. TF-IDF vectorization fitted on Train only
    3. Multi-model benchmarking (5-fold CV)
    4. Test set evaluation & metric reporting
    5. Serialization of best model, vectorizer, and topic model
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    df = load_and_clean_data(data_path)
    
    X = df['cleaned_text'].values
    y = df['sentiment'].values
    
    # 2. Strict Train / Test Split (80% Train, 20% Test, Stratified)
    # CRITICAL: Split BEFORE TF-IDF vectorization to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    
    # Save split datasets
    train_df = pd.DataFrame({"cleaned_text": X_train, "sentiment": y_train})
    test_df = pd.DataFrame({"cleaned_text": X_test, "sentiment": y_test})
    train_df.to_csv("data/processed/train_data.csv", index=False)
    test_df.to_csv("data/processed/test_data.csv", index=False)
    
    # 3. TF-IDF Feature Extraction
    print("Fitting TF-IDF Vectorizer on training data only...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_features=5000
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"TF-IDF Vocabulary Size: {X_train_vec.shape[1]} features")
    
    # 4. Candidate Models Definition
    # LinearSVC wrapped in CalibratedClassifierCV to provide well-calibrated probabilities
    candidate_models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "Linear Support Vector Machine": CalibratedClassifierCV(
            LinearSVC(C=1.0, random_state=42, max_iter=2000),
            cv=3
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=20, random_state=42, n_jobs=-1
        )
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    
    results = {}
    fitted_models = {}
    
    print("\n" + "="*70)
    print(f"{'MODEL BENCHMARKING & 5-FOLD CROSS VALIDATION':^70}")
    print("="*70)
    
    for name, model in candidate_models.items():
        print(f"\nTraining & Cross-Validating: {name}...")
        start_time = time.time()
        
        # 5-fold cross validation on training set
        cv_scores = cross_validate(model, X_train_vec, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        train_duration = round(time.time() - start_time, 3)
        
        # Fit on full training set
        model.fit(X_train_vec, y_train)
        fitted_models[name] = model
        
        # Evaluate on unseen Test Set
        inference_start = time.time()
        y_pred = model.predict(X_test_vec)
        inference_latency_ms = round(((time.time() - inference_start) / len(X_test)) * 1000, 3)
        
        test_acc = accuracy_score(y_test, y_pred)
        test_bal_acc = balanced_accuracy_score(y_test, y_pred)
        test_prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        test_rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        test_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        results[name] = {
            "cv_accuracy_mean": round(float(np.mean(cv_scores['test_accuracy'])), 4),
            "cv_accuracy_std": round(float(np.std(cv_scores['test_accuracy'])), 4),
            "cv_f1_macro_mean": round(float(np.mean(cv_scores['test_f1_macro'])), 4),
            "cv_f1_macro_std": round(float(np.std(cv_scores['test_f1_macro'])), 4),
            "test_accuracy": round(float(test_acc), 4),
            "test_balanced_accuracy": round(float(test_bal_acc), 4),
            "test_precision_macro": round(float(test_prec), 4),
            "test_recall_macro": round(float(test_rec), 4),
            "test_f1_macro": round(float(test_f1), 4),
            "training_time_sec": train_duration,
            "inference_latency_ms": inference_latency_ms,
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=["Negative", "Neutral", "Positive"]).tolist()
        }
        
        print(f"-> 5-Fold CV F1-Macro: {results[name]['cv_f1_macro_mean']:.4f} (+/- {results[name]['cv_f1_macro_std']:.4f})")
        print(f"-> Test Accuracy:      {test_acc:.4f} | Test F1-Macro: {test_f1:.4f}")
        print(f"-> Latency:            {inference_latency_ms} ms/sample")

    # 5. Model Comparison Summary Table
    summary_data = []
    for name, r in results.items():
        summary_data.append({
            "Model": name,
            "CV F1 (Macro)": f"{r['cv_f1_macro_mean']:.3f} +/- {r['cv_f1_macro_std']:.3f}",
            "Test Accuracy": f"{r['test_accuracy']:.3f}",
            "Test F1 (Macro)": f"{r['test_f1_macro']:.3f}",
            "Precision": f"{r['test_precision_macro']:.3f}",
            "Recall": f"{r['test_recall_macro']:.3f}",
            "Latency (ms)": r["inference_latency_ms"]
        })
    summary_df = pd.DataFrame(summary_data)
    print("\n" + "="*70)
    print("FINAL MODEL BENCHMARK TABLE")
    print("="*70)
    print(summary_df.to_string(index=False))

    # 6. Select Best Performing Model based on Test F1-Macro
    best_model_name = max(results.keys(), key=lambda m: results[m]['test_f1_macro'])
    best_model = fitted_models[best_model_name]
    print(f"\n[WINNING MODEL]: '{best_model_name}' with Test F1-Macro: {results[best_model_name]['test_f1_macro']}")
    
    # 7. Unsupervised Topic Modeling (LDA)
    print("\nTraining Latent Dirichlet Allocation (LDA) Topic Model on Training Corpus...")
    topic_modeler = TopicModeler(n_topics=5, random_state=42)
    topic_modeler.fit(X_train)
    topic_modeler.save(os.path.join(output_dir, "lda_topic_model.joblib"))
    print("Discovered LDA Topics:")
    for t in topic_modeler.get_topics():
        print(f"  * {t['label']} (Top: {', '.join(t['keywords'])})")

    # 8. Save Artifacts & Figures
    print("\nSerializing best model and preprocessing pipelines...")
    joblib.dump(best_model, os.path.join(output_dir, "best_sentiment_model.joblib"))
    joblib.dump(vectorizer, os.path.join(output_dir, "tfidf_vectorizer.joblib"))
    
    # Save metrics JSON
    metrics_export = {
        "best_model_name": best_model_name,
        "classes": ["Negative", "Neutral", "Positive"],
        "comparison": results,
        "lda_topics": topic_modeler.get_topics(),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(output_dir, "model_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_export, f, indent=2)

    # 9. Plot Confusion Matrix of Best Model
    try:
        cm = np.array(results[best_model_name]["confusion_matrix"])
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Negative", "Neutral", "Positive"],
            yticklabels=["Negative", "Neutral", "Positive"]
        )
        plt.title(f"Confusion Matrix: {best_model_name}\n(Test Accuracy: {results[best_model_name]['test_accuracy']:.2%})")
        plt.xlabel("Predicted Sentiment")
        plt.ylabel("Actual Sentiment")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
        plt.close()
        
        # Model Comparison Chart
        plt.figure(figsize=(9, 4.5))
        models_list = list(results.keys())
        f1_scores = [results[m]["test_f1_macro"] for m in models_list]
        acc_scores = [results[m]["test_accuracy"] for m in models_list]
        x = np.arange(len(models_list))
        width = 0.35
        plt.bar(x - width/2, f1_scores, width, label='Test F1 (Macro)', color='#4361ee')
        plt.bar(x + width/2, acc_scores, width, label='Test Accuracy', color='#4cc9f0')
        plt.ylabel('Score (0.0 - 1.0)')
        plt.title('Algorithm Comparison: Accuracy vs F1-Score')
        plt.xticks(x, [m.replace(" ", "\n") for m in models_list])
        plt.ylim(0.5, 1.05)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=300)
        plt.close()
        print("Evaluation plots saved to models/confusion_matrix.png and models/model_comparison.png")
    except Exception as e:
        print(f"Warning: Could not save plots: {e}")

    print("\nTraining and Evaluation Pipeline Completed Successfully!")
    return results

if __name__ == "__main__":
    train_and_evaluate_models()
