---
title: 'LiteFND++: A Lightweight and Interpretable Model for Fake News Detection'
tags:
  - Python
  - Natural Language Processing
  - Fake News Detection
  - Interpretable Machine Learning
  - Ensemble Learning
authors:
  - name: Ovi Pal
    orcid: 0009-0002-6849-162X
    affiliation: "1"
affiliations:
  - name: Independent Researcher
    index: 1
date: 20 June 2024
bibliography: paper.bib
---

# Summary

**LiteFND++** is an open-source Python package that provides a computationally efficient and interpretable solution for fake news detection. It implements a novel dual-path ensemble architecture combining:

1. **Logistic Regression** with L2 regularization (`C=1.8`)
2. **Multinomial Naive Bayes** with Laplace smoothing (`α=0.03`)

Key technical features include:

- Advanced text preprocessing pipeline (NER-aware token joining, semantic normalization)
- Hybrid TF-IDF vectorization (word n-grams 1–4 and character n-grams 3–6)
- Weighted soft-voting ensemble (65% Logistic Regression, 35% Naive Bayes)
- Integrated LIME explanations for local model interpretability
- CPU-optimized implementation requiring <16GB RAM

LiteFND++ achieves state-of-the-art performance (F1 = 0.991) while being approximately **18–20× faster than transformer-based models like BERT** on CPU-only systems.

# Statement of Need

Despite recent advances, fake news detection systems continue to face key challenges:

1. **High computational cost**: Transformer models are resource-intensive and often require GPU acceleration.
2. **Limited interpretability**: Many high-performing models operate as black boxes.
3. **Difficult deployment**: Existing models are rarely suitable for low-resource or real-time environments.

LiteFND++ addresses these issues by providing:

- A pure Python solution with minimal dependencies
- Transparent, human-readable explanations via LIME
- Compatibility with consumer-grade hardware (e.g., laptops, Raspberry Pi)
- Open-source licensing for academic and commercial use

This enables new applications such as:

- Browser-based real-time verification tools
- Lightweight mobile fact-checking apps
- Offline detection in remote or low-bandwidth areas
- Interactive educational tools to combat misinformation

# Key Features

## Core Functionality

```python
from litefnd import LiteFND

# Initialize and train the model
model = LiteFND()
model.fit(X_train, y_train)

# Predict and explain
prediction = model.predict(news_text)
explanation = model.explain(news_text)
