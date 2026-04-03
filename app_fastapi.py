from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle

with open("predictor.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    vectorizer = pickle.load(f)


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
    return {
        "name": "Review Star Rating Predictor",
        "version": "1.0.0",
        "docs": "/docs"
    }
@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: ReviewRequest):
    review_tfidf = vectorizer.transform([request.review])
    prediction = model.predict(review_tfidf)[0]

    return PredictionResponse(
        review=request.review,
        predicted_rating=int(prediction),
        stars="⭐" * int(prediction)
    )

@app.post("/batch", response_model=BatchResponse)
def batch_predict(request: BatchRequest):
    reviews_tfidf = vectorizer.transform(request.reviews)
    predictions = model.predict(reviews_tfidf)

    results = [
        PredictionResponse(
            review=rev,
            predicted_rating=int(pred),
            stars="⭐" * int(pred)
        )
        for rev, pred in zip(request.reviews, predictions)
    ]

    return BatchResponse(predictions=results, count=len(results))

