import streamlit as st
import pickle

# Load saved model and vectorizer
with open("predictor.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.title("⭐ Review Star Rating Predictor")

st.write("Enter a product review and get the predicted star rating.")

review = st.text_area("Write your review here:")

if st.button("Predict Rating"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        review_tfidf = vectorizer.transform([review])
        prediction = model.predict(review_tfidf)

        st.success(f"Predicted Rating: {prediction[0]} ⭐")
