from flask import Flask, request, jsonify
import numpy as np
from models import train_and_predict

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    start_year = data.get("start_year")
    end_year = data.get("end_year")
    accuracy, mae, r2, future_cases_male, future_cases_female, num_pred_male, num_pred_female = train_and_predict(start_year, end_year)
    
    result = {
        'Akurasi Model': accuracy,
        'Mean Absolute Error (MAE)': mae,
        'R² Score': r2,
        'Prediksi Probabilitas Gender (male)': future_cases_male[0],
        'Jumlah Prediksi (male)': num_pred_male,
        'Prediksi Probabilitas Gender (female)': future_cases_female[0],
        'Jumlah Prediksi (female)': num_pred_female
    }
    return jsonify(result)

@app.route('/', methods=['GET'])
def home():
    return "API for predicting stunting status. Use /predict endpoint with POST method."

if __name__ == '__main__':
    app.run(debug=True, port=5011)
