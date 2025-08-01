# tests/test_predictor.py

import os
import pytest
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from ml_predictor.predictor import load_model, predict

# === SETUP ===
@pytest.fixture(scope="module")
def dummy_model(tmp_path_factory):
    """Create and save a dummy model to test load_model and predict."""
    # Create fake data
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    model = LogisticRegression()
    model.fit(X, y)

    # Save model
    model_path = tmp_path_factory.mktemp("models") / "dummy_model.joblib"
    joblib.dump(model, model_path)

    return model, model_path, pd.DataFram_
