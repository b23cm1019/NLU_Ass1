# Observations

## Dataset
- Total cleaned samples: 2180
- Class counts: {'politics': 1090, 'sport': 1090}
- Split sizes: {'train': 1526, 'val': 327, 'test': 327}

## Best Overall Configuration
- Feature: bow
- Model: LogisticRegression
- Model artifact: models\bow__LogisticRegression.joblib
- Test Accuracy: 0.9969
- Test Macro F1: 0.9969
- Test Confusion Matrix [[TN, FP], [FN, TP]]: [[163, 1], [0, 163]]

## Best F1 by Feature Representation
- bow: 0.9969
- tfidf: 0.9969
- ngrams_2_3: 0.9908

## Best F1 by ML Model
- LogisticRegression: 0.9969
- SGDClassifier: 0.9969
- LinearSVC: 0.9939
- MultinomialNB: 0.9878
