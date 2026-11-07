
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load dataset
data = pd.read_csv("data/news_cleaned.csv")

# Clean data
data = data.dropna(subset=["text", "label"]).copy()
data["text"] = data["text"].astype(str).str.strip()
data = data[data["text"].str.split().str.len() >= 10]
data = data.drop_duplicates(subset=["text"])

# Convert labels
data["label"] = (
    data["label"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "fake": 0,
        "0": 0,
        "real": 1,
        "1": 1
    })
)

data = data.dropna(subset=["label"])
data["label"] = data["label"].astype(int)

print("Dataset distribution:")
print(data["label"].value_counts().sort_index())
print("0 = FAKE, 1 = REAL")
print("Total articles:", len(data))

# Features and labels
X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Build model
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=100000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=2000,
        C=1.0,
        class_weight="balanced",
        random_state=42
    ))
])

# Train
print("\nTraining model...")
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["FAKE", "REAL"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Save model
os.makedirs("outputs", exist_ok=True)

joblib.dump(model, "outputs/fake_news_model_improved.pkl")

print("\nModel saved successfully!")

# Test both classes using actual dataset examples
print("\nTesting actual REAL and FAKE articles:")

for label, name in [(0, "FAKE"), (1, "REAL")]:
    sample = data[data["label"] == label].sample(
        1, random_state=42
    )["text"].iloc[0]

    probabilities = model.predict_proba([sample])[0]
    predicted = model.predict([sample])[0]

    print("\nActual:", name)
    print("Predicted:", "REAL" if predicted == 1 else "FAKE")
    print("FAKE probability:", round(probabilities[0] * 100, 2), "%")
    print("REAL probability:", round(probabilities[1] * 100, 2), "%")
