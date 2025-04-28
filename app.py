# app.py
from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model (for .pkl file)
with open('best_random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Preprocessing function
def preprocess(data):
    df = pd.DataFrame([data])
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
