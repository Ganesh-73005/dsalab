# app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load('model.joblib')

# Preprocessing function (optional if needed)
def preprocess(data):
    df = pd.DataFrame([data])
    # Map categorical values if needed
    df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    df['Geography'] = df['Geography'].map({'France': 0, 'Spain': 1, 'Germany': 2})
    return df

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        processed_data = preprocess(input_data)
        prediction = model.predict(processed_data)[0]
        prediction_proba = model.predict_proba(processed_data)[0]

        response = {
            'prediction': int(prediction),
            'probabilities': prediction_proba.tolist()
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
