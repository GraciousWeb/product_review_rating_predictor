

# Amazon Review Rating Prediction (NLP with SVM)

## Overview

This project builds a Natural Language Processing (NLP) model to predict Amazon product review ratings (1–5 stars) based solely on review text. The pipeline includes text cleaning, TF-IDF feature extraction, and multi-class classification using a Linear Support Vector Machine (SVM).

The trained model and vectorizer are serialized for reuse in downstream applications.

## Objective

To automatically classify customer reviews into star ratings (1 to 5) using machine learning, enabling automated sentiment-based rating prediction and large-scale review analysis.

## Dataset

The dataset contains:

* **Base_Review** – Raw review text
* **Stars** – Rating in text format (e.g., “5.0 out of 5 stars”)

### Preprocessing Steps

* Removed irrelevant columns
* Dropped missing values
* Converted text to lowercase
* Removed punctuation and special characters using regex
* Removed noise entries (e.g., “Translate review to English”)
* Removed duplicates
* Extracted numeric rating from star text
* Final target: `rating` (integer 1–5)


## Modeling Approach

### Train-Test Split

* 80% training
* 20% testing
* Random state = 42

### Text Vectorization

Used **TF-IDF Vectorizer** with:

* English stop words removal
* Maximum 5000 features
* N-grams (1,2) to capture word pairs
* Sparse matrix representation

### Classification Model

**LinearSVC (Support Vector Machine)** with:

* `class_weight='balanced'` (handles class imbalance)
* `max_iter=2000`
* `random_state=42`

## Model Performance

**Overall Accuracy:** `56%`

### Classification Summary:

* Strong performance on 1-star and 5-star reviews
* Lower performance on middle ratings (2–4 stars)
* Weighted F1-score: 0.56

The model performs better on extreme sentiment (very negative and very positive reviews), which is common in rating classification problems.

## Model Serialization

The trained artifacts are saved for reuse:

* `predictor.pkl` → Trained LinearSVC model
* `tfidf.pkl` → Fitted TF-IDF vectorizer

This allows the model to be loaded into a production API or web application without retraining.

## Tech Stack

* Python
* Pandas
* Scikit-learn
* TF-IDF (TfidfVectorizer)
* LinearSVC
* Pickle (model persistence)


## What This Project Demonstrates

* Practical NLP preprocessing pipeline
* Feature engineering with TF-IDF + n-grams
* Multi-class classification
* Handling class imbalance
* Model evaluation and interpretation
* Model serialization for deployment

## Potential Improvements

* Hyperparameter tuning (GridSearchCV)
* Compare with Logistic Regression / XGBoost
* Use pre-trained embeddings (Word2Vec, GloVe)
* Fine-tune a transformer model (BERT)
* Deploy as REST API (FastAPI)
