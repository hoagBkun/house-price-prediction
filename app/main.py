import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'final_model.joblib')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports', 'figures')

try:
    model = joblib.load(MODEL_PATH)
except:
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'status': 'error', 'message': 'Model not found'}), 500
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        # Force conversion for numeric fields
        numeric_cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt', 'FullBath']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col])
        prediction = model.predict(df)[0]
        return jsonify({'status': 'success', 'price': round(float(prediction), 2)})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

# Endpoint to serve evaluation plots to the UI
@app.route('/reports/<filename>')
def serve_report(filename):
    return send_from_directory(REPORTS_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5009, debug=True)