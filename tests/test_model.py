import pytest
from app.predictor import model


@pytest.mark.model
def test_model_has_predict_method():
    assert hasattr(model, "predict")


@pytest.mark.model
def test_model_has_predict_proba_method():
    assert hasattr(model, "predict_proba")