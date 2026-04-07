from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)


with open("predictor.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "name": "Review Star Rating Predictor",
        "version": "1.0",
        "endpoints": {
            "GET /": "API info",
            "GET /health": "Health check",
            "POST /predict": "Predict rating from review text",
            "POST /batch": "Predict ratings for multiple reviews"
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

    return jsonify({
        "review": review,
        "predicted_rating": int(prediction),
        "stars": " " * int(prediction)
    })

@app.route("/batch", methods=["POST"])
def batch_predict():
    data = request.get_json()

    if "reviews" not in data:
        return jsonify({"error": "Missing 'reviews' field"}), 400

    reviews = data["reviews"]
    reviews_tfidf = vectorizer.transform(reviews)
    predictions = model.predict(reviews_tfidf)

    results = [
        {"review": rev, "predicted_rating": int(pred), "stars": "⭐" * int(pred)}
        for rev, pred in zip(reviews, predictions)
    ]

    return jsonify({"predictions": results, "count": len(results)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000, debug=True)