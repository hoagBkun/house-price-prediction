import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Initialize Flask app
app = Flask (__name__)

# Production path configuration
BASE_DIR = os.path.dirname (os.path.dirname (os.path.abspath (__file__)))
MODEL_PATH = os.path.join (BASE_DIR, 'models', 'final_model.joblib')

# Load the trained model globally with error handling
model = None
if os.path.exists (MODEL_PATH):
	try:
		model = joblib.load (MODEL_PATH)
		print ("Production model loaded successfully.")
	except Exception as e:
		print (f"Error loading model: {str (e)}")
else:
	print (f"Warning: Model file not found at {MODEL_PATH}. Ensure the model is trained and pushed.")


@app.route ('/')
def index ():
	# Render the main UI
	return render_template ('index.html')


@app.route ('/predict', methods=['POST'])
def predict ():
	# Prediction endpoint for web requests
	if not model:
		return jsonify ({'status': 'error', 'message': 'Model not available on server'}), 500

	try:
		data = request.get_json ()
		df = pd.DataFrame ([data])

		# Enforce numeric data types for specific columns
		numeric_cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt', 'FullBath']
		for col in numeric_cols:
			if col in df.columns:
				df [col] = pd.to_numeric (df [col])

		# Execute prediction
		prediction = model.predict (df) [0]

		return jsonify ({
			'status': 'success',
			'price': round (float (prediction), 2),
			'currency': 'USD'
		})
	except Exception as e:
		return jsonify ({'status': 'error', 'message': f"Prediction failed: {str (e)}"}), 400


if __name__ == '__main__':
	# Use environment PORT for cloud deployment (Render, Heroku, etc.)
	port = int (os.environ.get ("PORT", 5000))
	app.run (host='0.0.0.0', port=port)