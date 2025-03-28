#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# app.py
#!/usr/bin/env python
# coding: utf-8

from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# ✅ Print current directory files to check if "scaler.pkl" exists
print("Current directory files:", os.listdir("."))

# ✅ Load the trained model with error handling
MODEL_FILE = "scaler.pkl"
model = None  # Initialize model as None

try:
    if os.path.exists(MODEL_FILE):  # Check if the file exists
        with open(MODEL_FILE, "rb") as file:
            model = pickle.load(file)
        print("Model loaded successfully!")
    else:
        print(f"Error: Model file '{MODEL_FILE}' not found.")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route("/", methods=["GET", "POST", "HEAD"])
def home():
    if request.method == "HEAD":
        return "", 200  # Send an empty response for HEAD requests
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return "Error: Model not loaded. Please check logs."

        # ✅ Get input values from form (assuming 5 inputs)
        features = [float(request.form.get(f'feature{i+1}', 0)) for i in range(5)]
        features = np.array([features])

        # ✅ Predict churn
        prediction = model.predict(features)[0]
        result = "Will Churn" if prediction == 1 else "Will Not Churn"

        return render_template("result.html", prediction=result)
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)


