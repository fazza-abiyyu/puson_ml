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
        return jsonify({'error': 'Please provide both start_year and end_year as query parameters'}), 400

    result = train_and_predict(start_year, end_year)
    if result:
        return jsonify(result)
    else:
        return jsonify({'error': 'No stunting data available for the specified years'}), 400

if __name__ == '__main__':
    app.run(debug=True)
