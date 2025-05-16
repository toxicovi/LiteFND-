# LiteFND++

**LiteFND++** is a lightweight, interpretable, and high-performing fake news detection system based on TF-IDF, Logistic Regression, and Naive Bayes. It offers real-time classification with LIME-based explanations.

## 📌 Features
- Dual-path ensemble (Logistic Regression + Naive Bayes)
- TF-IDF with custom token preprocessing
- LIME interpretability
- <50ms inference time on CPU
- Accuracy: 0.991, F1-score: 0.991

## 🚀 Quick Start

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset
Download the dataset from [Kaggle: Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) and place `True.csv` and `Fake.csv` inside the `data/` directory.

### 3. Train Model
```bash
python train_model.py
```

### 4. Run Interactive Inference
```bash
python interactive_demo.py
```

## 📊 Sample Output
```
Prediction: Fake
Confidence: 92.3%
Top Predictive Features:
- anonymous (Indicates Fake)
- conspiracy (Indicates Fake)
```

## 📄 License
To be decided.

## 🔍 Paper
The paper is currently in preparation and will be shared upon publication.
