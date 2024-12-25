from flask import Flask, request, jsonify
import numpy as np
from datetime import datetime
from models import train_and_predict, fetch_joined_data

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    start_year = request.args.get('start_year', type=int)
    end_year = request.args.get('end_year', type=int)
    
    if not start_year or not end_year:
        return jsonify({'error': 'Harap berikan start_year dan end_year sebagai parameter kueri'}), 400

    fetch_joined_data(start_year, end_year)
    accuracy, mae, r2, future_cases_male, future_cases_female, num_pred_male, num_pred_female = train_and_predict(start_year, end_year)
    
    result = {
        'Akurasi Model': accuracy * 100,
        'Mean Absolute Error (MAE)': format(mae, ".5f"),
        'R² Score': r2 * 100,
        'Prediksi Probabilitas Gender (male)': future_cases_male[0] * 100,
        'Jumlah Prediksi (male)': num_pred_male,
        'Prediksi Probabilitas Gender (female)': future_cases_female[0] * 100,
        'Jumlah Prediksi (female)': num_pred_female
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5011)
