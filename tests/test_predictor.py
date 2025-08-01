import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from ml_predictor.predictor import load_model, predict

def test_predict_dummy_model(tmp_path):
    # Create dummy model using sklearn
    model = LogisticRegression()
    X_train = np.array([[0, 0], [1, 1]])
    y_train = np.array([0, 1])
    model.fit(X_train, y_train)

    # Save model to temporary path
    model_path = tmp_path / "dummy_model.joblib"
    joblib.dump(model, model_path)

    # Test load_model
    loaded_model = load_model(str(model_path))
    assert hasattr(loaded_model, "predict")

    # Test predict
    inputs = np.array([[0, 0], [1, 1]])
    preds = predict(loaded_model, inputs)

    assert isinstance(preds, (list, np.ndarray))  # Accept both
    assert np.array_equal(preds, [0, 1])
