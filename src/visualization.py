import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


def plot_actual_vs_predicted (y_true, y_pred, save_path):
	# Plot Actual vs Predicted values
	plt.figure (figsize=(10, 6))
	plt.scatter (y_true, y_pred, alpha=0.5, color='#3b82f6')
	plt.plot ([y_true.min (), y_true.max ()], [y_true.min (), y_true.max ()], 'r--', lw=2)
	plt.xlabel ('Actual Price')
	plt.ylabel ('Predicted Price')
	plt.title ('Actual vs Predicted House Prices')
	plt.tight_layout ()
	plt.savefig (save_path)
	plt.close ()


def plot_residuals (y_true, y_pred, save_path):
	# Plot residuals to check for homoscedasticity
	residuals = y_true - y_pred
	plt.figure (figsize=(10, 6))
	plt.scatter (y_pred, residuals, alpha=0.5, color='#f59e0b')
	plt.axhline (y=0, color='r', linestyle='--')
	plt.xlabel ('Predicted Price')
	plt.ylabel ('Residuals')
	plt.title ('Residual Plot (Error Distribution)')
	plt.tight_layout ()
	plt.savefig (save_path)
	plt.close ()


def plot_feature_importance (model, feature_names, save_path):
	# Plot top 10 most important features
	importances = model.named_steps ['model'].feature_importances_
	importance_df = pd.DataFrame ({'Feature': feature_names, 'Importance': importances})
	importance_df = importance_df.sort_values (by='Importance', ascending=True).tail (10)

	plt.figure (figsize=(10, 6))
	plt.barh (importance_df ['Feature'], importance_df ['Importance'], color='#10b981')
	plt.title ('Top 10 Feature Importances')
	plt.xlabel ('Relative Importance')
	plt.tight_layout ()
	plt.savefig (save_path)
	plt.close ()