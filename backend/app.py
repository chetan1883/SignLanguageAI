from flask import Flask, request, jsonify
from flask_cors import CORS

import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

model_path = os.path.join(
    BASE_DIR,
    "models",
    "model.pkl"
)

with open(model_path, "rb") as f:
    model = pickle.load(f)

print("Model Loaded Successfully")


@app.route("/")
def home():
    return "Backend Running"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json["landmarks"]

    prediction = model.predict(
        [np.array(data)]
    )[0]

    probabilities = model.predict_proba(
        [np.array(data)]
    )[0]

    confidence = round(
        np.max(probabilities) * 100,
        2
    )

    return jsonify({
        "prediction": prediction,
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(debug=True)