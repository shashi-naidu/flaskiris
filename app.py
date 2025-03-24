#!/usr/bin/env python
# coding: utf-8

from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained model with error handling
try:
    model = joblib.load('iris_model.pkl')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Avoid crashes if the model fails to load

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('result.html', prediction="Model not available. Please contact support.")

    try:
        # Get data from form safely
        features = [float(request.form.get(f'feature{i}', 0)) for i in range(1, 5)]
    except ValueError:
        return render_template('result.html', prediction="Invalid input. Please enter numeric values.")

    # Ensure valid input length
    if len(features) != 4:
        return render_template('result.html', prediction="Incomplete input. Please provide all 4 features.")

    # Make prediction
    prediction = model.predict([features])[0]

    # Map prediction to class name
    class_names = ['Setosa', 'Versicolor', 'Virginica']
    result = class_names[int(prediction)] if 0 <= int(prediction) < len(class_names) else "Unknown"

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
