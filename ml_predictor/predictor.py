# ml_predictor/predictor.py

import joblib
import pandas as pd
from typing import Union

def load_model(model_path: str):
    """
    Load a serialized machine learning model.
    
    Args:
        model_path (str): Path to the saved model file (.joblib).
    
    Returns:
        object: Loaded model.
    """
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

def predict(model, data: Union[pd.DataFrame, dict]):
    """
    Make predictions using the given model and data.
    
    Args:
        model: A trained ML model.
        data (pd.DataFrame or dict): Input data for prediction.
    
    Returns:
        np.ndarray or list: Predictions.
    """
    if isinstance(data, dict):
        data = pd.DataFrame([data])  # Convert single sample dict to DataFrame

    try:
        predictions = model.predict(data)
        return predictions
    except Exception as e:
        raise RuntimeError(f"Error making prediction: {e}")
