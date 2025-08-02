from typing import Dict, Any  
from ml_config import load_config  
from ml_validator import validate_schema 
import joblib
import pandas as pd
import numpy as np
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
        validate_model(model)
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")


def predict(model, data: Union[pd.DataFrame, dict]) -> np.ndarray:
    """
    Make predictions using the given model and data.
    
    Args:
        model: A trained ML model.
        data (pd.DataFrame or dict): Input data for prediction.
            For dicts: values should be scalars or arrays, not nested lists
    
    Returns:
        np.ndarray: Predictions.
    """
    if isinstance(data, dict):
        # Convert dict to DataFrame, ensuring no nested sequences
        data = pd.DataFrame({k: [v] if not isinstance(v, (list, np.ndarray)) else v 
                     for k, v in data.items()})
    
    try:
        return model.predict(data)
    except Exception as e:
        raise RuntimeError(
            f"Prediction failed. Ensure input data matches model requirements.\n"
            f"Original error: {str(e)}"
        )


def predict_with_validation(
    model_path: str, 
    data: Union[pd.DataFrame, Dict[str, Any]], 
    config_path: str = "config/prediction.yaml"):

    """
    End-to-end prediction with schema validation.
    
    Args:
        model_path: Path to serialized model
        data: Input data (dict or DataFrame)
        config_path: Path to validation config
    
    Returns:
        Predictions with guaranteed input validation
    """
    model = load_model(model_path)
    config = load_config(config_path)
    validate_schema(pd.DataFrame([data] if isinstance(data, dict) else data), 
                   config["input_schema"])
    return predict(model, data)



def predict_batch(
    model, 
    data: pd.DataFrame, 
    chunk_size: int = 1000,
    logger=None) -> np.ndarray:

    """
    Predict on large datasets in chunks.
    
    Args:
        chunk_size: Rows per batch (default: 1000)
        logger: Optional logger from ml_logger
    
    Returns:
        Concatenated predictions
    """
    if logger:
        logger.info(f"Predicting on {len(data)} rows in {np.ceil(len(data)/chunk_size)} chunks")
    
    return np.concatenate([
        model.predict(data[i:i+chunk_size]) 
        for i in range(0, len(data), chunk_size)
    ])


def validate_model(model):
    """
    Verify model has required methods.
    Raises ValueError if invalid.
    """
    if not (hasattr(model, 'predict') or hasattr(model, 'predict_proba')):
        raise ValueError(
            "Model must implement predict() or predict_proba()\n"
            f"Actual methods: {dir(model)}"
        )


