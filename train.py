# train.py
import pandas as pd
import re
import pickle
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

def train():
    df = pd.read_csv("cleaned_amazon_reviews.csv")

    X = df["clean_review"]
    y = df["rating"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LinearSVC(class_weight="balanced", max_iter=2000, random_state=42)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    with open("predictor.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("tfidf.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/scores.json", "w") as f:
        json.dump({"accuracy": round(acc, 4), "weighted_f1": round(report["weighted avg"]["f1-score"], 4)}, f, indent=2)

    print("Model saved!")
    return acc

if __name__ == "__main__":
    train()