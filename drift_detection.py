import pandas as pd
import numpy as np
import re
import os
import pickle
import json
from sklearn.model_selection import train_test_split
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

df = pd.read_csv("dataset.csv")
df["rating"] = df["Stars"].str.extract(r'(\d+)').astype(float)
df = df.dropna(subset=["rating"])
df["rating"] = df["rating"].astype(int)
df["clean_review"] = df["Base_Review"].apply(clean_text)
df = df.dropna(subset=["clean_review"])
df = df[df["clean_review"].str.len() > 0]
df = df.drop_duplicates(subset=["clean_review"])

print(f"Total cleaned reviews: {len(df)}")


df["review_length"] = df["clean_review"].str.len()


df["word_count"] = df["clean_review"].str.split().str.len()


df["avg_word_length"] = df["clean_review"].apply(
    lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
)

df["exclamation_count"] = df["Base_Review"].apply(
    lambda x: str(x).count("!")
)

df["question_count"] = df["Base_Review"].apply(
    lambda x: str(x).count("?")
)


df["uppercase_ratio"] = df["Base_Review"].apply(
    lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
)

monitor_columns = [
    "rating", "review_length", "word_count",
    "avg_word_length", "exclamation_count",
    "question_count", "uppercase_ratio"
]
df_monitor = df[monitor_columns]

print(f"Monitor features: {monitor_columns}")


reference_data, current_data = train_test_split(
    df_monitor, test_size=0.3, random_state=42
)

print(f"Reference data (training): {len(reference_data)} reviews")
print(f"Current data (production): {len(current_data)} reviews")

os.makedirs("reports", exist_ok=True)


print("\n" + "=" * 50)
print("SCENARIO 1: No Drift")
print("=" * 50)

report_no_drift = Report(metrics=[DataDriftPreset()])
report_no_drift.run(reference_data=reference_data, current_data=current_data)
report_no_drift.save_html("reports/no_drift_report.html")

result = report_no_drift.as_dict()
drift_detected = result["metrics"][0]["result"]["dataset_drift"]
drift_share = result["metrics"][0]["result"]["drift_share"]

print(f"Dataset drift detected: {drift_detected}")
print(f"Share of drifted features: {drift_share:.0%}")
print(f"Report saved → reports/no_drift_report.html")


print("\n" + "=" * 50)
print("SCENARIO 2: Reviews Get Shorter (Data Drift)")
print("=" * 50)

drifted_short = current_data.copy()
drifted_short["review_length"] = drifted_short["review_length"] * 0.3

drifted_short["word_count"] = drifted_short["word_count"] * 0.4


report_short = Report(metrics=[DataDriftPreset()])
report_short.run(reference_data=reference_data, current_data=drifted_short)
report_short.save_html("reports/drift_shorter_reviews.html")

result_short = report_short.as_dict()
print(f"Dataset drift detected: {result_short['metrics'][0]['result']['dataset_drift']}")
print(f"Drifted features: {result_short['metrics'][0]['result']['drift_share']:.0%}")
print(f"Report saved → reports/drift_shorter_reviews.html")


print("\n" + "=" * 50)
print("SCENARIO 3: Ratings Shift Down (Concept Drift)")
print("=" * 50)

drifted_ratings = current_data.copy()
drifted_ratings["rating"] = drifted_ratings["rating"].apply(lambda x: max(1, x - 2))

drifted_ratings["exclamation_count"] = drifted_ratings["exclamation_count"] * 3

drifted_ratings["uppercase_ratio"] = drifted_ratings["uppercase_ratio"] * 2


report_ratings = Report(metrics=[DataDriftPreset()])
report_ratings.run(reference_data=reference_data, current_data=drifted_ratings)
report_ratings.save_html("reports/drift_rating_shift.html")

result_ratings = report_ratings.as_dict()
print(f"Dataset drift detected: {result_ratings['metrics'][0]['result']['dataset_drift']}")
print(f"Drifted features: {result_ratings['metrics'][0]['result']['drift_share']:.0%}")
print(f"Report saved → reports/drift_rating_shift.html")


print("\n" + "=" * 50)
print("SCENARIO 4: Mixed Drift (Data + Concept)")
print("=" * 50)