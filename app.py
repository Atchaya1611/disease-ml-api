import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

model = joblib.load("disease_prediction_model.pkl")
encoder = joblib.load("label_encoder.pkl")

@app.route("/")
def home():
    return "ML Disease Prediction API Running"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    ecg = data["ecg"]
    pulse = data["pulse"]
    temp = data["temperature"]

    input_data = np.array([[ecg, pulse, temp]])

    prediction = model.predict(input_data)
    disease = encoder.inverse_transform(prediction)

    return jsonify({"disease": disease[0]})

if __name__ == "__main__":
    app.run()