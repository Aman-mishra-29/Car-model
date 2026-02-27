from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load your trained model
model = joblib.load("car_price_prediction.joblib")

@app.route('/')
def landing_page():
    return render_template("home.html")

@app.route('/predict')
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print("🔍 Received input:", data)

        # Extract and convert input values to match training feature order
        features = [
            float(data.get("wheelbase", 0)),
            float(data.get("curbweight", 0)),
            float(data.get("enginesize", 0)),
            float(data.get("boreratio", 0)),
            float(data.get("horsepower", 0)),
            float(data.get("citympg", 0)),
            float(data.get("highwaympg", 0)),
            float(data.get("carlength", 0)),
            float(data.get("carwidth", 0)),

            # One-hot encoded categorical values (as 0 or 1)
            int(data.get("fueltype_gas", 0)),
            int(data.get("aspiration_turbo", 0)),
            int(data.get("carbody_hardtop", 0)),
            int(data.get("carbody_hatchback", 0)),
            int(data.get("carbody_sedan", 0)),
            int(data.get("carbody_wagon", 0)),
            int(data.get("drivewheel_fwd", 0)),
            int(data.get("drivewheel_rwd", 0)),
            int(data.get("enginetype_dohcv", 0)),
            int(data.get("enginetype_l", 0)),
            int(data.get("enginetype_ohc", 0)),
            int(data.get("enginetype_ohcf", 0)),
            int(data.get("enginetype_ohcv", 0)),
            int(data.get("enginetype_rotor", 0)),
            int(data.get("cylindernumber_five", 0)),
            int(data.get("cylindernumber_four", 0)),
            int(data.get("cylindernumber_six", 0)),
            int(data.get("cylindernumber_three", 0)),
            int(data.get("cylindernumber_twelve", 0)),
            int(data.get("cylindernumber_two", 0))
        ]

        prediction = model.predict([features])[0]
        print("Prediction:", prediction)

        return jsonify({"price": round(prediction, 2)})
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)

