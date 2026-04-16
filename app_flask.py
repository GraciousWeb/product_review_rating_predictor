from flask import Flask, request, jsonify
import pickle
import json
import os
import numpy as np
from datetime import datetime, timedelta
from collections import Counter

app = Flask(__name__)

with open("predictor.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    vectorizer = pickle.load(f)


class PredictionLogger:
    def __init__(self, log_file="logs/predictions.jsonl"):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log(self, review_text, predicted_rating):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "review_text": review_text[:200],
            "review_length": len(review_text),
            "word_count": len(review_text.split()),
            "predicted_rating": int(predicted_rating),
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_logs(self, hours=24):
        if not os.path.exists(self.log_file):
            return []
        cutoff = datetime.now() - timedelta(hours=hours)
        logs = []
        with open(self.log_file) as f:
            for line in f:
                entry = json.loads(line.strip())
                if datetime.fromisoformat(entry["timestamp"]) > cutoff:
                    logs.append(entry)
        return logs

    def get_stats(self, hours=24):
        logs = self.get_logs(hours)
        if not logs:
            return {"status": "no_data", "period_hours": hours}
        ratings = [l["predicted_rating"] for l in logs]
        lengths = [l["review_length"] for l in logs]
        return {
            "period_hours": hours,
            "total_predictions": len(logs),
            "avg_predicted_rating": round(float(np.mean(ratings)), 2),
            "rating_distribution": dict(Counter(ratings)),
            "avg_review_length": round(float(np.mean(lengths)), 0),
        }

    def check_alerts(self, hours=24):
        logs = self.get_logs(hours)
        if len(logs) < 10:
            return {"status": "insufficient_data", "count": len(logs)}
        ratings = [l["predicted_rating"] for l in logs]
        total = len(ratings)
        alerts = []

        for rating, count in Counter(ratings).items():
            if count / total > 0.80:
                alerts.append(f"Rating {rating} dominates at {count/total:.0%}")

        avg = float(np.mean(ratings))
        if avg < 2.0:
            alerts.append(f"Average rating very low: {avg:.2f}")
        elif avg > 4.5:
            alerts.append(f"Average rating suspiciously high: {avg:.2f}")

        return {
            "status": "alerts_found" if alerts else "healthy",
            "total_predictions": total,
            "avg_rating": round(avg, 2),
            "alerts": alerts,
        }

logger = PredictionLogger()


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "name": "Review Star Rating Predictor",
        "version": "1.0",
        "endpoints": {
            "GET /": "API info",
            "GET /health": "Health check",
            "POST /predict": "Predict rating",
            "POST /batch": "Batch predictions",
            "GET /monitoring/stats": "Prediction stats",
            "GET /monitoring/alerts": "Check for anomalies",
        }
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if "review" not in data:
        return jsonify({"error": "Missing 'review' field"}), 400

    review = data["review"]
    if not review.strip():
        return jsonify({"error": "Review text is empty"}), 400

    review_tfidf = vectorizer.transform([review])
    prediction = model.predict(review_tfidf)[0]

    # Log the prediction
    logger.log(review, prediction)

    return jsonify({
        "review": review,
        "predicted_rating": int(prediction),
        "stars": "⭐" * int(prediction)
    })

@app.route("/batch", methods=["POST"])
def batch_predict():
    data = request.get_json()

    if "reviews" not in data:
        return jsonify({"error": "Missing 'reviews' field"}), 400

    reviews = data["reviews"]
    reviews_tfidf = vectorizer.transform(reviews)
    predictions = model.predict(reviews_tfidf)

    # Log each prediction
    for rev, pred in zip(reviews, predictions):
        logger.log(rev, pred)

    results = [
        {"review": rev, "predicted_rating": int(pred), "stars": "⭐" * int(pred)}
        for rev, pred in zip(reviews, predictions)
    ]
    return jsonify({"predictions": results, "count": len(results)})


@app.route("/monitoring/stats", methods=["GET"])
def get_stats():
    """Prediction stats for the last N hours."""
    hours = request.args.get("hours", 24, type=int)
    return jsonify(logger.get_stats(hours))

@app.route("/monitoring/alerts", methods=["GET"])
def get_alerts():
    """Check for anomalies in recent predictions."""
    hours = request.args.get("hours", 24, type=int)
    return jsonify(logger.check_alerts(hours))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000, debug=True)
