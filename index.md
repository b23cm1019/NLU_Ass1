# Problem 4: Sports or Politics Classifier

This project builds a binary text classifier that predicts whether a news article is **Sport** or **Politics** using classical NLP and machine learning.

## Project Objective
Design and evaluate a document classifier with:
- Feature representations: **Bag of Words**, **N-grams**, **TF-IDF**
- Multiple ML techniques: **MultinomialNB**, **LogisticRegression**, **LinearSVC**, **SGDClassifier**
- Quantitative comparison and analysis

## Dataset
- Source: Scraped from **The Guardian** URLs provided in category-wise CSV files
- Final balanced dataset: **2180** samples
- Class distribution:
  - Politics: **1090**
  - Sport: **1090**

## Pipeline
1. **Data Collection**: `scrape_guardian.py`
2. **Recovery / Filtering Utility**: `pending.py`
3. **Preprocessing + Feature Extraction**: `prepare_features.py`
4. **Training + Evaluation**: `train_models.py`
5. **Prediction on custom text/document**: `predict_text.py`

## Feature Representations
- `bow`: unigram count vectors
- `ngrams_2_3`: bigram + trigram count vectors
- `tfidf`: unigram TF-IDF vectors

## Models Compared
- Multinomial Naive Bayes
- Logistic Regression
- Linear SVC
- SGD Classifier

## Best Result
From `results/model_comparison.csv`:
- **Feature**: `bow`
- **Model**: `LogisticRegression`
- **Test Accuracy**: `0.9969`
- **Test Macro F1**: `0.9969`
- **Confusion Matrix**: `[[163, 1], [0, 163]]`

## Reproducibility
```bash
python prepare_features.py --input data/raw/guardian_articles_balanced.csv
python train_models.py
python predict_text.py --feature tfidf --model LogisticRegression --text "The team dominated the tournament final."
