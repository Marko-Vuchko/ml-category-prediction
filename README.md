# Product Category Prediction

Multi-class product category classifier for an e-commerce catalog. Predicts product categories from titles and view counts using TF-IDF text features and a Linear SVM pipeline.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

---

## Overview

This project builds an automated category-assignment model for marketplace listings. Given a product title and number of views, the model predicts the correct category label — supporting catalog cleanup, search relevance, and listing automation.

| Task | Business question | Method |
|------|-------------------|--------|
| **1** | Can we auto-classify products from title text? | TF-IDF + Linear SVC |
| **2** | Does view count improve predictions? | Numeric feature scaling |
| **3** | Which categories are hardest to classify? | Per-class precision/recall |
| **4** | Is the model production-ready? | Saved sklearn Pipeline (joblib) |

---

## Model performance

Hold-out test set (~6,952 samples). Best model: **Linear SVC** with TF-IDF on product titles and MinMax-scaled view counts.

| Model | Test accuracy |
|-------|---------------|
| Linear SVC (selected) | **96.0%** |
| Random Forest | 93.1% |
| Decision Tree | 94.0% |
| Logistic Regression | 95.9% |
| Naive Bayes | 97.1% |

The production pipeline uses Linear SVC for strong accuracy with efficient inference on text features.

---

## Repository structure

```
.
├── data/
│   └── products.csv              # Training dataset
├── model/
│   └── category_prediction_model.pkl
├── notebook/
│   └── ML_Category_Prediction.ipynb
├── src/
│   ├── train_model.py            # Train and save pipeline
│   └── predict_category.py       # Interactive CLI inference
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/Marko-Vuchko/ml-category-prediction.git
cd ml-category-prediction
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Train the model

```bash
python src/train_model.py
```

### 3. Run predictions (interactive CLI)

```bash
python src/predict_category.py
```

Example:

```
Enter product title: iPhone 7 32GB Gold
Predicted category: Mobile Phones
```

---

## Methodology

1. Load `products.csv` and drop rows with missing values.
2. Normalize category labels (lowercase, canonical spelling).
3. Features: `product_title` (TF-IDF) + `number_of_views` (MinMaxScaler).
4. Compare classifiers in the notebook; deploy Linear SVC via sklearn Pipeline.
5. Serialize the full pipeline with joblib for consistent preprocessing at inference time.

---

## Tech stack

**Python:** pandas, scikit-learn, joblib, Jupyter

---

## Author

**Marko Vučković** — Data Analyst & Developer  
[GitHub](https://github.com/Marko-Vuchko) · [Email](mailto:markovucko12@gmail.com)

---

## License

This project is released under the [MIT License](LICENSE).
