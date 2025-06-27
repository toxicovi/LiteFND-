import pytest
import numpy as np
from litefnd.models import LiteFND

@pytest.fixture
def trained_model(sample_data):
    X, y = sample_data
    model = LiteFND()
    model.fit(X, y)
    return model

def test_model_prediction(trained_model, sample_text):
    pred = trained_model.predict(sample_text)
    assert pred in [0, 1]  # 0=fake, 1=real

def test_explanation(trained_model, sample_text):
    explanation = trained_model.explain(sample_text)
    assert hasattr(explanation, 'as_list')
    assert len(explanation.as_list()) > 0
