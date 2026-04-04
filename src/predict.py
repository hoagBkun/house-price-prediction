import pandas as pd
import os
import joblib

# Import feature definitions
from src.features import get_all_features

# Directory configuration
BASE_DIR = os.path.dirname (os.path.dirname (os.path.abspath (__file__)))
TEST_DATA_PATH = os.path.join (BASE_DIR, 'data', 'raw', 'test.csv')
MODEL_PATH = os.path.join (BASE_DIR, 'models', 'final_model.joblib')
SUBMISSION_DIR = os.path.join (BASE_DIR, 'data', 'processed')


def load_data (file_path):
	try:
		return pd.read_csv (file_path)
	except FileNotFoundError:
		print (f"Error: File {file_path} not found.")
		exit (1)


def generate_predictions ():
	print ("Loading test data...")
	df_test = load_data (TEST_DATA_PATH)

	print ("Loading trained model...")
	try:
		model = joblib.load (MODEL_PATH)
	except FileNotFoundError:
		print (f"Error: Model not found at {MODEL_PATH}. Run train.py first.")
		exit (1)

	features = get_all_features ()

	print ("Generating predictions...")
	X_test = df_test [features]
	test_predictions = model.predict (X_test)

	submission = pd.DataFrame ({
		'Id': df_test ['Id'],
		'SalePrice': test_predictions
	})

	os.makedirs (SUBMISSION_DIR, exist_ok=True)
	submission_path = os.path.join (SUBMISSION_DIR, 'submission.csv')
	submission.to_csv (submission_path, index=False)
	print (f"Submission file successfully saved to: {submission_path}")


if __name__ == '__main__':
	generate_predictions ()