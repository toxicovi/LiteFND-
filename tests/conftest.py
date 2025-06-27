import pytest
import pandas as pd
from sklearn.datasets import make_classification

@pytest.fixture
def sample_data():
    X, y = make_classification(n_samples=100, n_features=20)
    return X, y

@pytest.fixture
def sample_text():
    return "This is a sample news article for testing purposes"
