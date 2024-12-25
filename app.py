from flask import Flask, request, jsonify
from models import train_and_predict

# Initialize Flask app
app = Flask(__name__)

# Define prediction route
@app.route('/predict', methods=['GET'])
def predict():
    start_year = request.args.get('start_year', type=int)
    end_year = request.args.get('end_year', type=int)
    
    if not start_year or not end_year:
        return jsonify({'code': 400, 'message': 'Please provide both start_year and end_year as query parameters', 'data': {}}), 400

    result = train_and_predict(start_year, end_year)
    if result:
        data = {
            "prediksi": [
                {"Tahun Prediksi": end_year + 1},
                {"name": "Laki - Laki", "data": [pred['num_pred_male'] for pred in result['monthly_predictions']]},
                {"name": "Perempuan", "data": [pred['num_pred_female'] for pred in result['monthly_predictions']]}
            ],
            "categories": [
                "Januari", "Februari", "Maret", "April", "Mei", "Juni",
                "Juli", "Agustus", "September", "Oktober", "November", "Desember"
            ],
            "total_predictions": {
                "total_pred_male": result['total_predictions']['total_pred_male'],
                "total_pred_female": result['total_predictions']['total_pred_female']
            },
            "average_probabilities": {
                "avg_prob_male": result['average_probabilities']['avg_prob_male'],
                "avg_prob_female": result['average_probabilities']['avg_prob_female']
            },
            "model_evaluation": {
                "accuracy": f"{result['accuracy'] * 100:.2f}%",
                "mae": f"{result['mae']:.5f}",
                "r2": f"{result['r2']:.5f}"
            }
        }
        response = {
            "code": 200,
            "message": "Data berhasil dikembalikan",
            "data": data
        }
        return jsonify(response)
    else:
        return jsonify({'code': 400, 'message': 'No stunting data available for the specified years', 'data': {}}), 400

if __name__ == '__main__':
    app.run(debug=True)
