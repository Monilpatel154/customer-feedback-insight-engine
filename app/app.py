"""
Flask Application Entrypoint for Customer Feedback Sentiment & Insight Platform
"""

import os
import sys
import json
import io
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory

# Ensure project root is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.predictor import SentimentInsightPipeline
from src.generate_dataset import create_feedback_dataset

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = SentimentInsightPipeline(models_dir=os.path.join(BASE_DIR, "models"))
    return pipeline

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Empty text supplied."}), 400

    pipe = get_pipeline()
    try:
        result = pipe.predict_single(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/batch_upload", methods=["POST"])
def api_batch_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files are supported."}), 400

    try:
        df = pd.read_csv(file)
        # Check text column
        candidate_cols = ["review_text", "feedback", "review", "text", "comment", "feedback_text"]
        text_col = None
        for col in candidate_cols:
            if col in df.columns:
                text_col = col
                break
        if text_col is None:
            # Fall back to first string column
            str_cols = df.select_dtypes(include=['object']).columns
            if len(str_cols) > 0:
                text_col = str_cols[0]
            else:
                return jsonify({"error": "No text column found in CSV."}), 400

        pipe = get_pipeline()
        enriched_df, analytics = pipe.predict_batch(df, text_column=text_col)

        # Return top 250 rows for frontend rendering
        reviews_sample = enriched_df.head(250).to_dict(orient="records")
        return jsonify({
            "analytics": analytics,
            "reviews": reviews_sample
        })
    except Exception as e:
        return jsonify({"error": f"Failed to process CSV: {str(e)}"}), 500

@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    metrics_path = os.path.join(BASE_DIR, "models", "model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    else:
        return jsonify({"error": "Metrics not found. Please train models first."}), 404

@app.route("/api/download_sample", methods=["GET"])
def api_download_sample():
    sample_path = os.path.join(BASE_DIR, "data", "raw", "customer_feedback.csv")
    if not os.path.exists(sample_path):
        create_feedback_dataset(sample_path)
    return send_file(sample_path, as_attachment=True, download_name="customer_feedback_sample.csv")

@app.route("/static/img/<path:filename>")
def serve_model_images(filename):
    models_dir = os.path.join(BASE_DIR, "models")
    return send_from_directory(models_dir, filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"Starting InsightPulse ML Server at http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
