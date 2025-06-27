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

LiteFND++ is an open-source Python package that provides a computationally efficient and interpretable solution for fake news detection. The software implements a novel dual-path ensemble architecture combining:

1. Logistic Regression with L2 regularization (C=1.8)
2. Multinomial Naive Bayes with Laplace smoothing (α=0.03)

Key technical features include:

- Advanced text preprocessing pipeline (NER-aware token joining, semantic normalization)
- Hybrid TF-IDF vectorization (word n-grams 1-4 + character n-grams 3-6)
- Weighted soft-voting ensemble (65% LR, 35% NB)
- Integrated LIME explanations for model interpretability
- CPU-optimized implementation requiring <16GB RAM

The software achieves state-of-the-art performance (F1=0.991) while being 18-20× faster than transformer models like BERT on CPU-only systems.

# Statement of Need

Current fake news detection systems face three critical challenges:

1. **Computational Intensity**: Transformer models require GPU acceleration
2. **Interpretability Limitations**: Black-box decisions hinder trust
3. **Deployment Barriers**: High resource requirements exclude edge devices

LiteFND++ addresses these needs through:

- Pure Python implementation with minimal dependencies
- Human-readable explanations via LIME
- Efficient execution on consumer hardware
- MIT-licensed open source code

The software enables new applications in:
- Browser-based real-time verification
- Mobile fact-checking apps
- Resource-constrained environments
- Educational tools for media literacy

# Key Features

## Core Functionality

```python
from litefnd import LiteFND

# Initialize and train model
model = LiteFND()
model.fit(X_train, y_train)

# Generate predictions with explanations
prediction = model.predict(news_text)
explanation = model.explain(news_text)

## Archive

This software has been archived with Zenodo:

[![DOI](https://zenodo.org/badge/984619477.svg)](https://doi.org/10.5281/zenodo.15752843)
