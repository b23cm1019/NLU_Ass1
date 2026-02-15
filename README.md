# Sports vs Politics Text Classification (Problem 4)

## Overview
This repository contains a complete NLP classification pipeline for predicting whether a news article is **Sport** or **Politics** using classical machine learning.

The assignment requirements covered:
- Data collection and dataset analysis
- Feature representation with BoW, N-grams, TF-IDF
- Comparison of at least 3 ML techniques
- Quantitative evaluation and system limitations
- GitHub page with all details

## Project Structure
```text
.
├── scrape_guardian.py
├── pending.py
├── prepare_features.py
├── train_models.py
├── predict_text.py
├── requirements.txt
├── report.pdf
├── index.md
├── guardian_politics_links.csv
├── guardian_sports_links.csv
├── data
│   ├── raw
│   │   └── guardian_articles_balanced.csv
│   └── processed
│       └── clean_dataset.csv
├── models
│   ├── bow__*.joblib
│   ├── ngrams_2_3__*.joblib
│   └── tfidf__*.joblib
└── results
    ├── model_comparison.csv
    └── observations.md
