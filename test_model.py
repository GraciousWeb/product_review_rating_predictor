# test_model.py
import pytest
import pickle
import os
import json

def test_model_files_exist():
    assert os.path.exists("predictor.pkl")
    assert os.path.exists("tfidf.pkl")

def test_model_loads():
    with open("predictor.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    assert model is not None
    assert vectorizer is not None

def test_model_predicts():
    with open("predictor.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    reviews = ["This product is amazing!", "Terrible, waste of money", "Its okay"]
    tfidf = vectorizer.transform(reviews)
    predictions = model.predict(tfidf)

    for pred in predictions:
        assert pred in [1, 2, 3, 4, 5]

def test_extreme_reviews():
    with open("predictor.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    pos = vectorizer.transform(["best product ever love it amazing wonderful perfect"])
    assert model.predict(pos)[0] >= 4

    neg = vectorizer.transform(["worst product ever hate it terrible awful broken garbage"])
    assert model.predict(neg)[0] <= 2

def test_accuracy_threshold():
    if not os.path.exists("metrics/scores.json"):
        pytest.skip("metrics not found")
    with open("metrics/scores.json") as f:
        metrics = json.load(f)
    assert metrics["accuracy"] >= 0.50

def test_batch_shape():
    with open("predictor.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    reviews = ["good", "bad", "okay", "great", "terrible"]
    preds = model.predict(vectorizer.transform(reviews))
    assert len(preds) == 5