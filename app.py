#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# app.py

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# ✅ Load the trained churn model
try:
    with open("xgboost_model.pkl", "rb") as file:
        model = pickle.load(file)  # Ensure the file exists in the project directory
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Handle cases where model is missing

@app.route("/", methods=["GET", "POST", "HEAD"])
def home():
    if request.method == "HEAD":
        return "", 200  # Send an empty response for HEAD requests
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return "Error: Model not loaded."

        # ✅ Get input values from form (assuming 5 inputs)
        features = [float(request.form.get(f'feature{i+1}', 0)) for i in range(5)]
        features = np.array([features])

        # ✅ Predict churn
        prediction = model.predict(features)[0]
        result = "Will Churn" if prediction == 1 else "Will Not Churn"

        return render_template("result.html", prediction=result)
    except Exception as e:
        return f"Error: {e}"

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Use Render’s dynamic port
    app.run(host="0.0.0.0", port=port)

