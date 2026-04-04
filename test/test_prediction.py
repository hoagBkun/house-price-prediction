import pytest
import pandas as pd
from src.features import get_all_features


@pytest.fixture
def sample_input_data ():
	"""Fixture to generate dummy input data for testing"""
	data = {
		'OverallQual': [7],
		'GrLivArea': [1500],
		'GarageCars': [2],
		'TotalBsmtSF': [850],
		'YearBuilt': [2003],
		'FullBath': [2],
		'Neighborhood': ['CollgCr'],
		'ExterQual': ['Gd'],
		'KitchenQual': ['Gd'],
		'BldgType': ['1Fam']
	}
	return pd.DataFrame (data)


def test_feature_list_length ():
	"""Ensure feature definitions return the correct number of expected variables"""
	features = get_all_features ()
	# Expecting 6 numeric + 4 categorical = 10 features
	assert len (features) == 10, "Should have 10 features defined"


def test_data_schema (sample_input_data):
	"""Test if the generated sample data matches required model features"""
	features = get_all_features ()

	missing_cols = [feat for feat in features if feat not in sample_input_data.columns]
	assert len (missing_cols) == 0, f"Sample data is missing columns: {missing_cols}"