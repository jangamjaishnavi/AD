#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# app.py

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained churn model
from flask import send_file

@app.route("/download")
def download_file():
    return send_file("churn_model.pkl", as_attachment=True)

model = pickle.load(open('churn_model.pkl', 'wb'))

@app.route("/", methods=["GET", "POST", "HEAD"])  # Added HEAD method
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from form
        features = [float(request.form.get(f'feature{i+1}')) for i in range(5)]
        features = np.array([features])

        # Predict churn
        prediction = model.predict(features)[0]
        result = "Will Churn" if prediction == 1 else "Will Not Churn"

        return render_template("result.html", prediction=result)
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
