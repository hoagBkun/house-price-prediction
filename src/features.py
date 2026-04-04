# Feature definitions and selection

def get_categorical_features():
    # Return list of categorical columns
    return ['Neighborhood', 'ExterQual', 'KitchenQual', 'BldgType']

def get_numeric_features():
    # Return list of numerical columns
    return ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt', 'FullBath']

def get_all_features():
    # Combine numeric and categorical features
    return get_numeric_features() + get_categorical_features()