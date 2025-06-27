# LiteFND++: Lightweight Interpretable Fake News Detection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15752844.svg)](https://doi.org/10.5281/zenodo.15752844)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/toxicovi/LiteFND-/actions/workflows/tests.yml/badge.svg)](https://github.com/toxicovi/LiteFND-/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/toxicovi/LiteFND-/branch/main/graph/badge.svg)](https://codecov.io/gh/toxicovi/LiteFND-)

## 🔍 Overview

LiteFND++ is an open-source Python package that implements a **novel ensemble architecture** for fake news detection, combining:

- **Advanced Feature Engineering**: Semantic-aware TF-IDF with cognitive load metrics
- **Dynamic Ensemble Learning**: Confidence-weighted LR + NB combination
- **Explainable AI**: Context-aware LIME explanations

**Key Advantages**:
- 🚀 **20x faster** than BERT on CPUs (12ms inference)
- 🔍 **Human-interpretable** decision explanations
- 🛡️ **Adversarially robust** through novel training augmentation
- 📱 **Edge-compatible** (<350MB memory)

## 🎯 Key Features

### 1. Cognitive Load Analysis
```python
analyzer = CognitiveLoadAnalyzer()
scores = analyzer.analyze_text("Claim: Vaccines cause autism")
# Returns {'emotional': 0.72, 'logical_gaps': 0.65, ...}
