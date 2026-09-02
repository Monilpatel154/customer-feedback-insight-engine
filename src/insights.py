"""
Aspect-Based Sentiment Mining, Topic Modeling (LDA), and Actionable Insights Engine
"""

import re
import joblib
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

ASPECT_LEXICON = {
    "Product Quality": [
        "quality", "build", "material", "durability", "durable", "sturdy",
        "battery", "screen", "display", "hardware", "performance", "defect",
        "defective", "broke", "broken", "lag", "crash", "overheat", "heat",
        "finish", "casing", "hinge", "pixel", "microphone", "sound"
    ],
    "Customer Support": [
        "support", "service", "agent", "representative", "rep", "call", "chat",
        "helpdesk", "warranty", "claim", "replacement", "refund", "exchange",
        "ticket", "courteous", "polite", "rude", "unresponsive", "hold", "email"
    ],
    "Delivery & Packaging": [
        "delivery", "shipping", "courier", "package", "packaging", "box",
        "wrap", "transit", "ship", "dispatch", "arrive", "arrival", "late",
        "delay", "delayed", "lost", "crushed", "doorstep", "tracking"
    ],
    "Pricing & Value": [
        "price", "pricing", "cost", "expensive", "cheap", "value", "money",
        "budget", "affordable", "discount", "deal", "overpriced", "worth",
        "penny", "rip-off", "investment", "economic", "gouging"
    ],
    "Usability & UI": [
        "app", "software", "ui", "interface", "setup", "pair", "pairing",
        "bluetooth", "navigate", "navigation", "button", "control", "ergonomic",
        "ergonomics", "bug", "glitch", "freeze", "display", "setting", "manual"
    ]
}

class AspectSentimentAnalyzer:
    """Extracts aspect mentions and calculates clause-level sentiment."""
    
    def __init__(self, aspect_lexicon=None):
        self.aspect_lexicon = aspect_lexicon or ASPECT_LEXICON
        
    def detect_aspects(self, text: str):
        """Identifies which predefined business aspects are referenced in the review."""
        text_lower = text.lower()
        found_aspects = {}
        
        for aspect, keywords in self.aspect_lexicon.items():
            matches = [kw for kw in keywords if re.search(r'\b' + re.escape(kw), text_lower)]
            if matches:
                found_aspects[aspect] = list(set(matches))
                
        return found_aspects

    def analyze_aspect_sentiments(self, review_text: str, sentiment_classifier=None, vectorizer=None):
        """
        Splits review into clauses/sentences and attributes sentiment to specific aspects.
        Returns dictionary of aspect -> {sentiment, confidence, matched_text}.
        """
        detected = self.detect_aspects(review_text)
        if not detected:
            return {}
            
        # Split on sentence boundaries and contrastive conjunctions (, but / , however / ;)
        clauses = re.split(r'(?:[.!?;]|\s*,\s*(?:but|however|although|though|whereas|while)\s+)', review_text, flags=re.IGNORECASE)
        clauses = [c.strip() for c in clauses if c.strip()]
        aspect_results = {}
        
        for aspect, keywords in detected.items():
            relevant_clauses = []
            for c in clauses:
                c_lower = c.lower()
                if any(re.search(r'\b' + re.escape(kw), c_lower) for kw in keywords):
                    relevant_clauses.append(c.strip())
            
            aspect_context = ". ".join(relevant_clauses) if relevant_clauses else review_text
            
            # Predict sentiment on this aspect snippet if classifier provided
            if sentiment_classifier and vectorizer and aspect_context.strip():
                try:
                    from src.preprocess import clean_text
                    cleaned_ctx = clean_text(aspect_context)
                    vec = vectorizer.transform([cleaned_ctx])
                    pred_sentiment = sentiment_classifier.predict(vec)[0]
                    # Check if predict_proba is available
                    if hasattr(sentiment_classifier, "predict_proba"):
                        probs = sentiment_classifier.predict_proba(vec)[0]
                        confidence = float(max(probs))
                    else:
                        confidence = 0.85
                except Exception:
                    pred_sentiment = "Neutral"
                    confidence = 0.5
            else:
                pred_sentiment = "Detected"
                confidence = 1.0
                
            aspect_results[aspect] = {
                "keywords": keywords,
                "snippet": aspect_context,
                "sentiment": pred_sentiment,
                "confidence": round(confidence, 3)
            }
            
        return aspect_results


class TopicModeler:
    """Unsupervised Latent Dirichlet Allocation (LDA) for topic discovery."""
    
    def __init__(self, n_topics=5, max_features=1000, random_state=42):
        self.n_topics = n_topics
        self.count_vectorizer = CountVectorizer(
            max_df=0.90, min_df=2, max_features=max_features, stop_words='english'
        )
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics, random_state=random_state, learning_method='online', max_iter=25
        )
        self.is_fitted = False
        self.topic_names = []

    def fit(self, preprocessed_texts):
        """Fit count vectorizer and LDA on preprocessed texts."""
        dtm = self.count_vectorizer.fit_transform(preprocessed_texts)
        self.lda_model.fit(dtm)
        self.is_fitted = True
        self._generate_topic_labels()
        return self

    def _generate_topic_labels(self):
        """Generates interpretable topic descriptors from top weighted terms."""
        feature_names = self.count_vectorizer.get_feature_names_out()
        self.topic_names = []
        for idx, topic in enumerate(self.lda_model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
            label = f"Topic {idx + 1}: {' / '.join(top_words[:3])}"
            self.topic_names.append({"id": idx, "label": label, "keywords": top_words})

    def get_topics(self):
        return self.topic_names

    def predict_topic(self, preprocessed_text):
        """Predicts dominant topic for a given preprocessed text."""
        if not self.is_fitted:
            return None
        dtm = self.count_vectorizer.transform([preprocessed_text])
        topic_probs = self.lda_model.transform(dtm)[0]
        dominant_idx = int(topic_probs.argmax())
        return {
            "topic_id": dominant_idx,
            "topic_label": self.topic_names[dominant_idx]["label"],
            "confidence": round(float(topic_probs[dominant_idx]), 3),
            "distribution": [round(float(p), 3) for p in topic_probs]
        }

    def save(self, filepath="models/lda_topic_model.joblib"):
        joblib.dump({
            "vectorizer": self.count_vectorizer,
            "lda_model": self.lda_model,
            "topic_names": self.topic_names,
            "n_topics": self.n_topics
        }, filepath)

    @classmethod
    def load(cls, filepath="models/lda_topic_model.joblib"):
        data = joblib.load(filepath)
        instance = cls(n_topics=data["n_topics"])
        instance.count_vectorizer = data["vectorizer"]
        instance.lda_model = data["lda_model"]
        instance.topic_names = data["topic_names"]
        instance.is_fitted = True
        return instance


def generate_actionable_recommendations(batch_analytics: dict):
    """
    Synthesizes aggregate statistics into prioritized executive business recommendations.
    """
    recommendations = []
    
    aspect_counts = batch_analytics.get("aspect_counts", {})
    aspect_negative = batch_analytics.get("aspect_negative", {})
    sentiment_counts = batch_analytics.get("sentiment_counts", {})
    total_reviews = sum(sentiment_counts.values()) or 1
    
    neg_pct = (sentiment_counts.get("Negative", 0) / total_reviews) * 100
    
    # 1. Overall Health Alert
    if neg_pct > 30:
        recommendations.append({
            "severity": "CRITICAL",
            "category": "Customer Satisfaction Alert",
            "finding": f"High overall negative sentiment ({neg_pct:.1f}% of total reviews).",
            "action": "Immediate executive review required. Initiate immediate outreach to dissatisfied customers.",
            "impact": "Mitigates churn rate and prevents brand equity erosion."
        })
    elif neg_pct < 15:
        recommendations.append({
            "severity": "POSITIVE",
            "category": "High Loyalty Trend",
            "finding": f"Strong positive reception with only {neg_pct:.1f}% negative feedback.",
            "action": "Leverage satisfied customer base for referral incentives, testimonials, and case studies.",
            "impact": "Accelerates organic customer acquisition."
        })

    # 2. Aspect-Specific Insights
    for aspect, total_mentions in aspect_counts.items():
        if total_mentions == 0:
            continue
        neg_mentions = aspect_negative.get(aspect, 0)
        neg_ratio = (neg_mentions / total_mentions) * 100
        
        if aspect == "Delivery & Packaging" and neg_ratio >= 25:
            recommendations.append({
                "severity": "HIGH",
                "category": "Logistics & Fulfillment",
                "finding": f"{neg_ratio:.1f}% negative sentiment in Delivery & Packaging ({neg_mentions} complaints).",
                "action": "Review courier SLAs in affected zones. Enhance packaging cushioning to eliminate transit damage.",
                "impact": "Decreases return costs and transit-related refund requests by up to 25%."
            })
        elif aspect == "Product Quality" and neg_ratio >= 25:
            recommendations.append({
                "severity": "CRITICAL",
                "category": "Quality Assurance",
                "finding": f"{neg_ratio:.1f}% negative sentiment in Product Quality ({neg_mentions} reports of defects/durability issues).",
                "action": "Conduct immediate batch inspection with manufacturing QA. Flag recurring hardware failure modes.",
                "impact": "Prevents product recalls and improves lifetime value (LTV)."
            })
        elif aspect == "Customer Support" and neg_ratio >= 20:
            recommendations.append({
                "severity": "HIGH",
                "category": "Support Operations",
                "finding": f"{neg_ratio:.1f}% negative sentiment regarding Customer Service.",
                "action": "Reduce wait times with smart triage routing and train front-line agents on empathetic de-escalation.",
                "impact": "Increases first-contact resolution (FCR) rate and CSAT score."
            })
        elif aspect == "Pricing & Value" and neg_ratio >= 25:
            recommendations.append({
                "severity": "MEDIUM",
                "category": "Pricing & Packaging",
                "finding": f"{neg_ratio:.1f}% of price-conscious reviews perceive poor value for money.",
                "action": "Introduce entry-tier bundles, seasonal discounts, or clearly articulate premium feature ROI in marketing.",
                "impact": "Improves conversion rate for price-sensitive buyers."
            })
        elif aspect == "Usability & UI" and neg_ratio >= 25:
            recommendations.append({
                "severity": "MEDIUM",
                "category": "Software & UX",
                "finding": f"{neg_ratio:.1f}% of UX feedback reported pairing glitches, bugs, or confusing navigation.",
                "action": "Prioritize onboarding walkthrough in companion app and release bug fix for Bluetooth sync latency.",
                "impact": "Improves day-1 retention and app store star ratings."
            })

    if not recommendations:
        recommendations.append({
            "severity": "LOW",
            "category": "Steady State Operations",
            "finding": "Sentiment is well-balanced across all monitored operational aspects.",
            "action": "Continue baseline monitoring and maintain standard operating quality procedures.",
            "impact": "Sustains ongoing customer satisfaction."
        })
        
    return recommendations
