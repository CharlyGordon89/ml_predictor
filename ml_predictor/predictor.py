import pandas as pd
import numpy as np
import joblib
from typing import Union, Dict, Any
from ml_config import load_config
from ml_validator import SchemaValidator

class Predictor:

    def __init__(self, config_path: str = "config/prediction.yaml"):
        """
        Initialize the predictor with config and model loading.
        """
        self.config = load_config(config_path)
        self.input_schema = self.config["input_schema"]
        self.model = self._load_model(self.config["model_path"])
    
    def _load_model(self, model_path: str):
        """
        Load serialized ML model and validate it has prediction methods.
        """
        try:
            model = joblib.load(model_path)
            self._validate_model(model)
            return model
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

    def _validate_model(self, model):
        """
        Verify that the model implements predict() or predict_proba().
        """
        if not (hasattr(model, 'predict') or hasattr(model, 'predict_proba')):
            raise ValueError(
                "Model must implement predict() or predict_proba(). "
                f"Got: {dir(model)}"
            )

    def _prepare_dataframe(self, data: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
        """
        Convert input to DataFrame if needed and validate schema.
        """
        df = pd.DataFrame([data] if isinstance(data, dict) else data)
        validator = SchemaValidator(schema=self.input_schema)
        validator.validate(df)
        return df

    def predict(self, data: Union[Dict[str, Any], pd.DataFrame]) -> np.ndarray:
        """
        Predict using the loaded model and validated data.
        Returns np.ndarray.
        """
        df = self._prepare_dataframe(data)
        try:
            return self.model.predict(df)
        except Exception as e:
            raise RuntimeError(
                f"Prediction failed. Ensure input data matches model expectations.\n"
                f"Original error: {e}"
            )

    def predict_batch(
        self,
        data: pd.DataFrame,
        chunk_size: int = 1000,
        logger=None
    ) -> np.ndarray:
        """
        Batch prediction for large DataFrames. Splits into chunks.
        """
        df = self._prepare_dataframe(data)

        if logger:
            logger.info(f"Predicting on {len(df)} rows in chunks of {chunk_size}")

        predictions = [
            self.model.predict(df[i:i + chunk_size])
            for i in range(0, len(df), chunk_size)
        ]
        return np.concatenate(predictions)



