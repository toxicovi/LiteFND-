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

LiteFND++ is an open-source Python package that implements an innovative ensemble architecture for fake news detection, combining:

- Logistic Regression with L2 regularization (C=1.8)
- Multinomial Naive Bayes with Laplace smoothing (α=0.03)
- Novel confidence-based dynamic weighting

**Key Advantages**:
- 🚀 18-20× faster than BERT on CPUs (12ms inference)
- 🔍 Built-in explainability with enhanced LIME
- 📱 Runs on edge devices (<350MB memory)

```python
# Core dynamic weighting implementation
def calculate_weights(y_pred_proba):
    confidence = np.abs(y_pred_proba - 0.5) * 2
    return 0.5 + (confidence - 0.5) * 0.3
