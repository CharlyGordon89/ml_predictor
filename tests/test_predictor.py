import pytest
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.linear_model import LogisticRegression
from ml_predictor.predictor import Predictor


DUMMY_INPUT_PATH = "tests/resources/dummy_input.json"


# ---------------------- Fixtures ----------------------

@pytest.fixture
def dummy_input():
    with open(DUMMY_INPUT_PATH, "r") as f:
        return json.load(f)


@pytest.fixture
def dummy_batch(dummy_input):
    return [dummy_input, dummy_input]


@pytest.fixture
def sample_model():
    model = LogisticRegression()
    X = np.array([[25, 50000], [45, 120000]])
    y = np.array([0, 1])
    model.fit(X, y)
    return model


@pytest.fixture
def test_config_path(tmp_path, sample_model):
    # Save the model
    model_path = tmp_path / "model.joblib"
    joblib.dump(sample_model, model_path)

    # Create YAML config
    config_text = f"""
input_schema:
  age: int64
  income: float64
model_path: "{model_path}"
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text)

    return str(config_path)


@pytest.fixture
def predictor(test_config_path):
    return Predictor(config_path=test_config_path)


# ---------------------- Tests ----------------------

def test_predict_single_dict(predictor, dummy_input):
    preds = predictor.predict(dummy_input)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (1,)


def test_predict_dataframe(predictor, dummy_batch):
    df = pd.DataFrame(dummy_batch)
    preds = predictor.predict(df)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (2,)


def test_predict_batch_chunking(predictor, dummy_batch):
    df = pd.DataFrame(dummy_batch * 3)  # 6 rows
    preds = predictor.predict_batch(df, chunk_size=2)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (6,)


def test_missing_column_raises(predictor, dummy_input):
    bad_input = dummy_input.copy()
    bad_input.pop("age", None)
    with pytest.raises(ValueError) as exc:
        predictor.predict(bad_input)
    assert "Missing columns" in str(exc.value)



def test_wrong_dtype_raises(predictor, dummy_input):
    bad_input = dummy_input.copy()
    bad_input["income"] = "not_a_number"
    with pytest.raises(ValueError):
        predictor.predict(bad_input)


def test_invalid_model_interface(tmp_path, monkeypatch):
    class BadModel:
        def transform(self, X):  # Doesn't implement predict()
            return X

    monkeypatch.setattr("joblib.load", lambda path: BadModel())

    dummy_model_path = tmp_path / "dummy_model.joblib"
    dummy_model_path.write_text("placeholder")  # Create dummy file

    dummy_config = tmp_path / "bad_config.yaml"
    dummy_config.write_text(f"""
input_schema:
  age: int64
  income: float64
model_path: "{dummy_model_path}"
""")

    with pytest.raises(RuntimeError) as exc_info:
        _ = Predictor(config_path=str(dummy_config))
    assert "Model must implement predict" in str(exc_info.value)
