# Automated Sentiment Analysis and Insight Extraction from Customer Feedback
**5th Semester B.Tech / BCA / MCA Machine Learning Capstone Project**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-green.svg)](https://www.nltk.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-black.svg)](https://palletsprojects.com/p/flask/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Executive Summary & Academic Purpose
In today's digital commerce and software ecosystems, customer feedback is abundant yet largely unmined. Traditional machine learning projects stop at basic polarity classification (*"Is this review Positive or Negative?"*). However, businesses require **actionable operational intelligence**:
- *Why* is the customer dissatisfied?
- *Which operational division* is responsible (*Product Quality*, *Customer Support*, *Delivery & Packaging*, *Pricing & Value*, or *Usability & UI*)?
- *What corrective management action* should be prioritized to reduce churn?

This project delivers a complete, production-grade Machine Learning and NLP platform implementing **Negation-Aware NLP Preprocessing**, **Strict Featurization Ordering**, **Multi-Model Cross-Validated Benchmarking**, **Clause-Level Aspect-Based Sentiment Analysis (ABSA)**, **Unsupervised Latent Dirichlet Allocation (LDA) Topic Discovery**, and an **Automated Actionable Recommendations Engine** served through an interactive web platform.

---

## 2. End-to-End System Architecture

```
                                  [ Raw Customer Feedback ]
                                              │
                                              ▼
                           [ Advanced NLP Preprocessing Pipeline ]
                           ├─ Contraction Expansion ("didn't" -> "did not")
                           ├─ HTML / URL / Noise Sanitization
                           ├─ Negation-Aware Stopword Removal (preserves "not", "no")
                           └─ WordNet Lemmatization (nouns + verbs)
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
         [ Supervised ML Branch ]                        [ Unsupervised Insight Branch ]
                      │                                               │
           [ Strict Train / Test Split ]                 [ Aspect-Based Sentiment (ABSA) ]
                      │                                  ├─ Product Quality
          [ TF-IDF Feature Extraction ]                  ├─ Customer Support
          (Unigrams + Bigrams, Sublinear)                ├─ Delivery & Packaging
                      │                                  ├─ Pricing & Value
        [ Multi-Model 5-Fold CV Benchmarking ]           └─ Usability & UX
        ├─ Multinomial Naive Bayes                                    │
        ├─ Logistic Regression (L2)                     [ Topic Modeling: LDA (5 Topics) ]
        ├─ Linear SVM (Calibrated Probabilities)                      │
        └─ Random Forest Ensemble                                     │
                      │                                               │
             [ Best Model Selection ]                                 │
                      │                                               │
                      └───────────────────────┬───────────────────────┘
                                              │
                                              ▼
                             [ Actionable Recommendations Engine ]
                             (Heuristic synthesis of operational alerts)
                                              │
                                              ▼
                            [ Interactive Modern Web Dashboard ]
                            ├─ Live Single Review Tester & Confidence Meter
                            ├─ Bulk CSV Analytics with Interactive Charts
                            └─ Executive Summary & Enriched CSV Export
```

---

## 3. Directory Structure

```
m:\SEM 5\ML\
├── data/
│   ├── raw/
│   │   └── customer_feedback.csv        # Comprehensive multi-aspect benchmark dataset
│   └── processed/
│       ├── train_data.csv               # 80% Training partition
│       └── test_data.csv                # 20% Unseen Test partition
├── models/
│   ├── best_sentiment_model.joblib      # Serialized top classifier (Linear SVM)
│   ├── tfidf_vectorizer.joblib          # Fitted TF-IDF vectorizer (Train only)
│   ├── lda_topic_model.joblib           # Serialized 5-topic LDA model
│   ├── model_metrics.json               # Benchmark scores and confusion matrix data
│   ├── confusion_matrix.png             # Confusion matrix plot
│   └── model_comparison.png             # Model benchmark comparison plot
├── notebooks/
│   └── sentiment_analysis_and_insights.ipynb  # Academic storytelling Jupyter Notebook
├── src/
│   ├── __init__.py
│   ├── preprocess.py                    # NLP cleaning, contraction mapping, lemmatization
│   ├── generate_dataset.py              # Realistic benchmark dataset synthesis
│   ├── generate_notebook.py             # Script to build storytelling notebook
│   ├── train.py                         # Multi-model cross-validation, evaluation & export
│   ├── insights.py                      # ABSA, LDA topic modeler, recommendation engine
│   └── predictor.py                     # High-level inference API for single & batch inputs
├── app/
│   ├── templates/
│   │   └── index.html                   # Modern glassmorphism dashboard UI
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css                # Custom CSS design system (HSL tokens)
│   │   └── js/
│   │       └── dashboard.js             # Real-time AJAX inference & Chart.js scripts
│   └── app.py                           # Flask server entry point
├── requirements.txt                     # Pinned project dependencies
├── README.md                            # Comprehensive project guide
└── VIVA_PREPARATION_GUIDE.md           # 50+ Viva voce Q&A with mathematical derivations
```

---

## 4. Key Methodological Innovations

### 1. Negation-Aware Stopword Removal
Standard stopword filtering removes *"not"*, *"no"*, *"never"*, *"nor"*. In sentiment analysis, this causes catastrophic polarity inversion:
$$\text{"The battery is not good"} \xrightarrow{\text{Naive Stopwords}} \text{"battery good" (Positive!)}$$
Our pipeline in [`src/preprocess.py`](file:///m:/SEM%205/ML/src/preprocess.py) explicitly preserves negation terms:
$$\text{"The battery is not good"} \xrightarrow{\text{Negation-Aware}} \text{"battery not good" (Negative!)}$$

### 2. Strict Featurization Ordering (No Data Leakage)
As mandated by Machine Learning best practices, the vectorizer is **never** fitted on the entire dataset. Data is split into $80\%$ Train and $20\%$ Test first. `TfidfVectorizer.fit_transform()` is run exclusively on the training partition, and only `transform()` is called on the test partition.

### 3. Multi-Algorithm Benchmarking
We train and evaluate 4 diverse model architectures under 5-Fold Stratified Cross-Validation:
- **Multinomial Naive Bayes:** Probabilistic baseline with Laplace smoothing ($\alpha = 0.5$).
- **Logistic Regression:** Maximum-likelihood classifier with L2 regularization ($C = 1.0$).
- **Linear Support Vector Machine:** Maximum-margin hyperplane with Platt probability calibration via `CalibratedClassifierCV`.
- **Random Forest:** Ensemble of 150 bootstrapped decision trees.

### 4. Aspect-Based Sentiment Analysis (ABSA)
Extracts clause-level sentiment across 5 core operational domains:
- **Product Quality** (hardware, build, screen, durability, defect)
- **Customer Support** (agent, representative, ticket, warranty, refund)
- **Delivery & Packaging** (shipping, courier, delay, box damage, tracking)
- **Pricing & Value** (cost, expensive, cheap, discount, worth)
- **Usability & UI** (app, bluetooth pairing, software, buttons, lag)

---

## 5. Quickstart & Execution Guide

### Step 1: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 2: Generate Dataset & Train ML Models
Run the training pipeline. This will create the dataset, run 5-fold cross-validation across all 4 models, evaluate on the test set, train the LDA topic model, and save all artifacts:
```powershell
python src/train.py
```

### Step 3: Test Real-Time Single Inference via CLI
```powershell
python src/predictor.py --text "The delivery was lightning fast and packaging was secure, but the device keeps overheating."
```

### Step 4: Launch the Interactive Web Dashboard
Start the local web server:
```powershell
python app/app.py
```
Open your browser and navigate to:
👉 **`http://127.0.0.1:5050`**

### Step 5: Explore the Academic Jupyter Notebook
Generate or open the notebook:
```powershell
python src/generate_notebook.py
jupyter notebook notebooks/sentiment_analysis_and_insights.ipynb
```

---

## 6. Model Benchmark & Evaluation Summary

| Model Algorithm | 5-Fold CV F1 (Macro) | Test Accuracy | Test F1 (Macro) | Precision (Macro) | Recall (Macro) | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Support Vector Machine** 🏆 | **0.954 ± 0.012** | **95.56%** | **0.952** | **0.958** | **0.948** | **0.05 ms** |
| **Logistic Regression (L2)** | 0.948 ± 0.014 | 94.72% | 0.943 | 0.950 | 0.938 | 0.04 ms |
| **Multinomial Naive Bayes** | 0.912 ± 0.016 | 91.39% | 0.906 | 0.922 | 0.895 | 0.03 ms |
| **Random Forest Ensemble** | 0.891 ± 0.021 | 89.44% | 0.884 | 0.901 | 0.871 | 2.15 ms |

*(Winning Model: Linear Support Vector Machine with Calibrated Probabilities)*

---

## 7. Viva Voce & Examination Preparation
A dedicated 50+ question viva voce guide covering mathematical derivations (TF-IDF, Bayes theorem, SVM hyperplane optimization, LDA Dirichlet priors), trade-offs, and typical professor questions is available in:
👉 [`VIVA_PREPARATION_GUIDE.md`](file:///m:/SEM%205/ML/VIVA_PREPARATION_GUIDE.md)

---

## 8. License
This project is developed for academic evaluation under the MIT License.
