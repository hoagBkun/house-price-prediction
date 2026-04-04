import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

from src.preprocess import build_preprocessor
from src.features import get_all_features, get_numeric_features, get_categorical_features
from src.utils import setup_logger, load_config, load_data, save_model
from src.visualization import plot_actual_vs_predicted, plot_residuals, plot_feature_importance

# Configuration
logger = setup_logger ('logs/train.log')
BASE_DIR = os.path.dirname (os.path.dirname (os.path.abspath (__file__)))
config = load_config (os.path.join (BASE_DIR, 'config/config.yaml'))


def train ():
	logger.info ("Starting professional training pipeline")

	# Load and Split Data
	train_path = os.path.join (BASE_DIR, config ['data'] ['train_path'])
	df = load_data (train_path, logger)
	features = get_all_features ()

	X = df [features]
	y = df ['SalePrice']

	X_train, X_val, y_train, y_val = train_test_split (
		X, y, test_size=config ['training'] ['test_size'], random_state=config ['training'] ['random_state']
	)

	# Build and Train Model
	preprocessor = build_preprocessor ()
	rf_pipeline = Pipeline (steps=[
		('preprocessor', preprocessor),
		('model', RandomForestRegressor (random_state=config ['training'] ['random_state']))
	])

	param_grid = {
		'model__n_estimators': [200, 300],
		'model__max_depth': [15, 20],
		'model__max_features': [1.0, 'sqrt']
	}

	logger.info ("Optimizing model with GridSearchCV")
	grid_search = GridSearchCV (rf_pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
	grid_search.fit (X_train, y_train)

	best_model = grid_search.best_estimator_
	y_pred = best_model.predict (X_val)

	# Metrics
	r2 = r2_score (y_val, y_pred)
	rmse = np.sqrt (mean_squared_error (y_val, y_pred))
	logger.info (f"Performance: R2={r2:.4f}, RMSE={rmse:.2f}")

	# Export Reports & Figures
	reports_dir = os.path.join (BASE_DIR, 'reports')
	figures_dir = os.path.join (reports_dir, 'figures')
	os.makedirs (figures_dir, exist_ok=True)

	# Save Metrics
	with open (os.path.join (reports_dir, 'metrics.json'), 'w') as f:
		json.dump ({'r2': r2, 'rmse': rmse, 'best_params': grid_search.best_params_}, f, indent=4)

	# Visualization
	logger.info ("Generating evaluation plots")
	plot_actual_vs_predicted (y_val, y_pred, os.path.join (figures_dir, 'actual_vs_pred.png'))
	plot_residuals (y_val, y_pred, os.path.join (figures_dir, 'residuals.png'))

	cat_encoder = best_model.named_steps ['preprocessor'].named_transformers_ ['cat'].named_steps ['onehot']
	feature_names = get_numeric_features () + list (cat_encoder.get_feature_names_out (get_categorical_features ()))
	plot_feature_importance (best_model, feature_names, os.path.join (figures_dir, 'feature_importance.png'))

	# Finalize
	save_model (best_model, os.path.join (BASE_DIR, config ['model'] ['save_dir']), config ['model'] ['model_name'],
	            logger)
	logger.info ("Training pipeline finished")


if __name__ == '__main__':
	train ()