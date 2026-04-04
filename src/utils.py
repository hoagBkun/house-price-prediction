import os
import yaml
import joblib
import logging
import pandas as pd

def setup_logger(log_file):
    # Configure logging for the project
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def load_config(path):
    # Load YAML configuration
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_data(path, logger=None):
    # Load dataset from path
    if logger: logger.info(f"Loading data from {path}")
    return pd.read_csv(path)

def save_model(model, directory, filename, logger=None):
    # Save model to disk
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    joblib.dump(model, path)
    if logger: logger.info(f"Model saved to {path}")