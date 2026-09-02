"""
Model Evaluation and Metrics Viewer
Reads serialized model metrics and prints a clean, detailed academic report.
"""

import os
import json
import pandas as pd

def show_evaluation_report(metrics_path="models/model_metrics.json"):
    if not os.path.exists(metrics_path):
        print(f"Error: Metrics file not found at '{metrics_path}'. Please run 'python src/train.py' first.")
        return

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    best_model = data.get("best_model_name", "Unknown")
    comparison = data.get("comparison", {})
    lda_topics = data.get("lda_topics", [])

    print("\n" + "=" * 78)
    print(f"{'MACHINE LEARNING MODEL BENCHMARK & EVALUATION REPORT':^78}")
    print("=" * 78)

    # 1. Comparison Table
    table_rows = []
    for name, m in comparison.items():
        table_rows.append({
            "Algorithm": name + (" (BEST)" if name == best_model else ""),
            "Test Accuracy": f"{m['test_accuracy'] * 100:.2f}%",
            "Macro F1": f"{m['test_f1_macro'] * 100:.2f}%",
            "Macro Prec": f"{m['test_precision_macro'] * 100:.2f}%",
            "Macro Rec": f"{m['test_recall_macro'] * 100:.2f}%",
            "5-Fold CV F1": f"{m['cv_f1_macro_mean']:.3f} +/- {m['cv_f1_macro_std']:.3f}",
            "Latency": f"{m['inference_latency_ms']} ms"
        })

    df_comp = pd.DataFrame(table_rows)
    print("\n[1] ALGORITHM COMPARISON TABLE (Evaluated on 360 Unseen Test Reviews):")
    print("-" * 78)
    print(df_comp.to_string(index=False))

    # 2. Detailed Breakdown for Best Model
    print("\n" + "=" * 78)
    print(f"WINNING MODEL DETAILS: {best_model}")
    print("=" * 78)

    best_details = comparison.get(best_model, {})
    rep = best_details.get("classification_report", {})

    class_rows = []
    for cls_name in ["Positive", "Neutral", "Negative"]:
        if cls_name in rep:
            c = rep[cls_name]
            class_rows.append({
                "Sentiment Class": cls_name,
                "Precision": f"{c['precision'] * 100:.2f}%",
                "Recall": f"{c['recall'] * 100:.2f}%",
                "F1-Score": f"{c['f1-score'] * 100:.2f}%",
                "Test Support (Samples)": int(c['support'])
            })
    
    df_class = pd.DataFrame(class_rows)
    print("\n[2] CLASS-WISE PRECISION, RECALL & F1-SCORE:")
    print("-" * 78)
    print(df_class.to_string(index=False))

    # 3. Confusion Matrix
    cm = best_details.get("confusion_matrix", [])
    if cm:
        print("\n[3] CONFUSION MATRIX (Rows: Actual, Columns: Predicted [Negative, Neutral, Positive]):")
        print("-" * 78)
        print(f"  Actual Negative ->  [TN={cm[0][0]:3d},  Neu={cm[0][1]:2d},  Pos={cm[0][2]:2d}]")
        print(f"  Actual Neutral  ->  [Neg={cm[1][0]:3d},  Neu={cm[1][1]:2d},  Pos={cm[1][2]:2d}]")
        print(f"  Actual Positive ->  [Neg={cm[2][0]:3d},  Neu={cm[2][1]:2d},  TP={cm[2][2]:3d}]")

    # 4. Discovered LDA Topics
    if lda_topics:
        print("\n" + "=" * 78)
        print("[4] DISCOVERED LDA TOPIC CLUSTERS (Unsupervised Insight Mining):")
        print("-" * 78)
        for t in lda_topics:
            print(f"  * {t['label']}")
            print(f"    Top Weighted Terms: {', '.join(t['keywords'])}")

    print("\n" + "=" * 78)
    print("VISUAL EVALUATION ASSETS AVAILABLE AT:")
    print("  - Confusion Matrix Plot: models/confusion_matrix.png")
    print("  - Benchmark Comparison:  models/model_comparison.png")
    print("  - Interactive Web View:  http://127.0.0.1:5050/#models")
    print("=" * 78 + "\n")

if __name__ == "__main__":
    show_evaluation_report()
