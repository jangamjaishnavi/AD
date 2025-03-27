#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# app.py

from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = joblib.load('iris_model.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    try:
        features = [float(request.form[f'feature{i}']) for i in range(1, 5)]
    except ValueError:
        return render_template('result.html', prediction="Invalid input. Please enter numeric values.")

    # Make prediction
    prediction = model.predict([features])[0]

    # Map prediction to class name
    class_names = ['Setosa', 'Versicolor', 'Virginica']
    result = class_names[prediction]

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)

