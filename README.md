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
MIT License

Copyright (c) 2025 Ovi Pal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## 🔍 Paper
The paper is currently in preparation and will be shared upon publication.
