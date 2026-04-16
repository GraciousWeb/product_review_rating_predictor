from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import json
import os
import numpy as np
from datetime import datetime, timedelta
from collections import Counter

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
            "avg_predicted_rating": round(np.mean(ratings), 2),
            "rating_distribution": dict(Counter(ratings)),
            "avg_review_length": round(np.mean(lengths), 0),
        }

    def check_alerts(self, hours=24):
        logs = self.get_logs(hours)
        if len(logs) < 10:
            return {"status": "insufficient_data", "count": len(logs)}
        ratings = [l["predicted_rating"] for l in logs]
        total = len(ratings)
        alerts = []

        # One rating dominates (>80%)
        for rating, count in Counter(ratings).items():
            if count / total > 0.80:
                alerts.append(f"Rating {rating} dominates at {count/total:.0%}")

        # Average rating too low or too high
        avg = np.mean(ratings)
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
class ReviewRequest(BaseModel):
    review: str = Field(..., description="Product review text", min_length=1,
                        examples=["This product is absolutely amazing!"])

class PredictionResponse(BaseModel):
    review: str
    predicted_rating: int
    stars: str

class BatchRequest(BaseModel):
    reviews: list[str] = Field(..., description="List of review texts", min_length=1)

class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int


app = FastAPI(title="Review Star Rating Predictor", version="1.0.0")

@app.get("/")
def home():
    return {"name": "Review Star Rating Predictor", "version": "1.0.0", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: ReviewRequest):
    review_tfidf = vectorizer.transform([request.review])
    prediction = model.predict(review_tfidf)[0]

    # Log the prediction
    logger.log(request.review, prediction)

    return PredictionResponse(
        review=request.review,
        predicted_rating=int(prediction),
        stars="⭐" * int(prediction)
    )

@app.post("/batch", response_model=BatchResponse)
def batch_predict(request: BatchRequest):
    reviews_tfidf = vectorizer.transform(request.reviews)
    predictions = model.predict(reviews_tfidf)

    # Log each prediction
    for rev, pred in zip(request.reviews, predictions):
        logger.log(rev, pred)

    results = [
        PredictionResponse(review=rev, predicted_rating=int(pred), stars="⭐" * int(pred))
        for rev, pred in zip(request.reviews, predictions)
    ]
    return BatchResponse(predictions=results, count=len(results))

@app.get("/monitoring/stats")
def get_stats(hours: int = 24):
    """Prediction stats for the last N hours."""
    return logger.get_stats(hours)

@app.get("/monitoring/alerts")
def get_alerts(hours: int = 24):
    """Check for anomalies in recent predictions."""
    return logger.check_alerts(hours)