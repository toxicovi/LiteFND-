import pytest
from litefnd.preprocessing import NERPreprocessor, TextNormalizer

@pytest.fixture
def preprocessor():
    return NERPreprocessor()

def test_ner_joining(preprocessor):
    text = "Barack Obama visited Washington"
    processed = preprocessor(text)
    assert "Barack_Obama" in processed

def test_normalization():
    normalizer = TextNormalizer()
    text = "The event occurred on 01/01/2023 in New York"
    processed = normalizer(text)
    assert "[DATE]" in processed
    assert "[LOC]" in processed
