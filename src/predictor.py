"""
High-Level Inference and Insight Extraction Pipeline
Serves both real-time single review inference and bulk CSV batch processing.
"""

import os
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import pandas as pd
import numpy as np

from src.preprocess import clean_text
from src.insights import AspectSentimentAnalyzer, TopicModeler, generate_actionable_recommendations

class SentimentInsightPipeline:
    """Unified engine for sentiment classification, aspect mining, and insight generation."""
    
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.model = None
        self.vectorizer = None
        self.topic_modeler = None
        self.aspect_analyzer = AspectSentimentAnalyzer()
        self._load_resources()

    def _load_resources(self):
        """Loads serialized model, vectorizer, and topic model."""
        model_path = os.path.join(self.models_dir, "best_sentiment_model.joblib")
        vec_path = os.path.join(self.models_dir, "tfidf_vectorizer.joblib")
        lda_path = os.path.join(self.models_dir, "lda_topic_model.joblib")

        if os.path.exists(model_path) and os.path.exists(vec_path):
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vec_path)
        else:
            self.model = None
            self.vectorizer = None

        if os.path.exists(lda_path):
            try:
                self.topic_modeler = TopicModeler.load(lda_path)
            except Exception:
                self.topic_modeler = None
        else:
            self.topic_modeler = None

    def is_ready(self):
        return self.model is not None and self.vectorizer is not None

    def predict_single(self, raw_text: str):
        """
        Executes complete inference on a single customer review.
        Returns:
            - cleaned_text
            - sentiment (Positive, Neutral, Negative)
            - confidence
            - probabilities dictionary
            - aspects (detected aspects + clause sentiment)
            - topic (dominant LDA topic)
        """
        if not self.is_ready():
            self._load_resources()
            if not self.is_ready():
                raise RuntimeError("Models are not trained yet. Please run 'python src/train.py' first.")

        cleaned = clean_text(raw_text)
        if not cleaned:
            return {
                "raw_text": raw_text,
                "cleaned_text": "",
                "sentiment": "Neutral",
                "confidence": 0.50,
                "probabilities": {"Negative": 0.33, "Neutral": 0.34, "Positive": 0.33},
                "aspects": {},
                "topic": None
            }

        vec = self.vectorizer.transform([cleaned])
        pred_label = self.model.predict(vec)[0]

        # Calculate class probabilities
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(vec)[0]
            classes = list(self.model.classes_)
            prob_dict = {cls: round(float(prob), 4) for cls, prob in zip(classes, probs)}
            confidence = round(float(max(probs)), 4)
        else:
            prob_dict = {pred_label: 0.85}
            confidence = 0.85

        # Aspect extraction
        aspects = self.aspect_analyzer.analyze_aspect_sentiments(
            raw_text, sentiment_classifier=self.model, vectorizer=self.vectorizer
        )

        # Topic prediction
        topic = None
        if self.topic_modeler:
            topic = self.topic_modeler.predict_topic(cleaned)

        return {
            "raw_text": raw_text,
            "cleaned_text": cleaned,
            "sentiment": pred_label,
            "confidence": confidence,
            "probabilities": prob_dict,
            "aspects": aspects,
            "topic": topic
        }

    def predict_batch(self, df: pd.DataFrame, text_column: str = "review_text"):
        """
        Processes a batch of feedbacks from a DataFrame.
        Enriches DataFrame with predictions and calculates aggregate analytics & recommendations.
        """
        if not self.is_ready():
            self._load_resources()
            if not self.is_ready():
                raise RuntimeError("Models are not trained yet. Please run 'python src/train.py' first.")

        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in dataframe.")

        df = df.copy()
        raw_texts = df[text_column].astype(str).tolist()
        cleaned_texts = [clean_text(t) for t in raw_texts]
        df['cleaned_text'] = cleaned_texts

        # Vectorize and predict
        vecs = self.vectorizer.transform(cleaned_texts)
        pred_labels = self.model.predict(vecs)
        df['predicted_sentiment'] = pred_labels

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(vecs)
            classes = list(self.model.classes_)
            df['confidence'] = [round(float(max(p)), 3) for p in probs]
            for i, cls_name in enumerate(classes):
                df[f'prob_{cls_name}'] = [round(float(p[i]), 3) for p in probs]
        else:
            df['confidence'] = 0.85

        # Aspect and Topic processing
        all_aspects = []
        dominant_topics = []
        aspect_counts = {k: 0 for k in self.aspect_analyzer.aspect_lexicon.keys()}
        aspect_negative = {k: 0 for k in self.aspect_analyzer.aspect_lexicon.keys()}
        aspect_positive = {k: 0 for k in self.aspect_analyzer.aspect_lexicon.keys()}

        for raw_t, clean_t, pred_s in zip(raw_texts, cleaned_texts, pred_labels):
            aspect_res = self.aspect_analyzer.analyze_aspect_sentiments(
                raw_t, sentiment_classifier=self.model, vectorizer=self.vectorizer
            )
            all_aspects.append(list(aspect_res.keys()))

            for asp, details in aspect_res.items():
                aspect_counts[asp] = aspect_counts.get(asp, 0) + 1
                if details['sentiment'] == "Negative":
                    aspect_negative[asp] = aspect_negative.get(asp, 0) + 1
                elif details['sentiment'] == "Positive":
                    aspect_positive[asp] = aspect_positive.get(asp, 0) + 1

            if self.topic_modeler and clean_t:
                top = self.topic_modeler.predict_topic(clean_t)
                dominant_topics.append(top['topic_label'] if top else "General")
            else:
                dominant_topics.append("General")

        df['detected_aspects'] = all_aspects
        df['dominant_topic'] = dominant_topics

        # Aggregate Statistics
        sentiment_counts = df['predicted_sentiment'].value_counts().to_dict()
        total_samples = len(df)
        
        analytics = {
            "total_samples": total_samples,
            "sentiment_counts": sentiment_counts,
            "sentiment_percentages": {
                k: round((v / total_samples) * 100, 1) for k, v in sentiment_counts.items()
            },
            "aspect_counts": aspect_counts,
            "aspect_negative": aspect_negative,
            "aspect_positive": aspect_positive,
            "avg_confidence": round(float(df['confidence'].mean()), 3),
            "top_topics": df['dominant_topic'].value_counts().to_dict()
        }

        # Generate Actionable Business Recommendations
        analytics["recommendations"] = generate_actionable_recommendations(analytics)

        return df, analytics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test Sentiment Inference")
    parser.add_argument("--text", type=str, default="The delivery was quick and packaging was neat, but the app crashed constantly.", help="Feedback text")
    args = parser.parse_args()

    pipeline = SentimentInsightPipeline()
    res = pipeline.predict_single(args.text)
    print("\n--- Inference Result ---")
    print(f"Raw Text:    {res['raw_text']}")
    print(f"Sentiment:   {res['sentiment']} (Confidence: {res['confidence']:.2%})")
    print(f"Aspects:     {res['aspects']}")
    if res['topic']:
        print(f"Topic:       {res['topic']['topic_label']}")
