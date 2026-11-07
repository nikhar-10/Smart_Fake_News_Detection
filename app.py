
from flask import Flask, render_template, request
import joblib

from src.database import create_table, save_prediction, get_history

app = Flask(__name__)

# Load trained model
model = joblib.load("outputs/fake_news_model_improved.pkl")

# Create database table
create_table()

# Prediction thresholds
REAL_THRESHOLD = 0.60
FAKE_THRESHOLD = 0.80


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    news_text = ""
    error = None

    if request.method == "POST":
        news_text = request.form.get("news", "").strip()

        if len(news_text.split()) < 10:
            error = "Please enter a meaningful news article (at least 10 words)."

        else:
            try:
                # Get prediction probabilities
                probabilities = model.predict_proba([news_text])[0]
                classes = list(model.classes_)

                fake_prob = float(probabilities[classes.index(0)])
                real_prob = float(probabilities[classes.index(1)])

                # Classification
                if real_prob >= REAL_THRESHOLD:
                    label = "REAL"

                elif fake_prob >= FAKE_THRESHOLD:
                    label = "FAKE"

                else:
                    label = "REAL"

                # Confidence percentage
                confidence = max(fake_prob, real_prob) * 100

                prediction = {
                    "label": label,
                    "confidence": round(confidence, 2),
                    "fake_probability": round(fake_prob * 100, 2),
                    "real_probability": round(real_prob * 100, 2)
                }

                # Save prediction in database
                save_prediction(
                    news_text,
                    label,
                    round(confidence, 2)
                )

            except Exception as e:
                error = f"Prediction Error: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        news_text=news_text,
        error=error
    )


@app.route("/history")
def history():
    records = get_history()

    return render_template(
        "history.html",
        records=records
    )


if __name__ == "__main__":
    app.run(debug=True)
