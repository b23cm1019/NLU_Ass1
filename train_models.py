import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


MODEL_NAMES = ["MultinomialNB", "LogisticRegression", "LinearSVC", "SGDClassifier"]
FEATURE_SETS = ["bow", "ngrams_2_3", "tfidf"]


def build_model(model_name: str):
    if model_name == "MultinomialNB":
        return MultinomialNB(alpha=0.5)
    if model_name == "LogisticRegression":
        return LogisticRegression(max_iter=5000)
    if model_name == "LinearSVC":
        return LinearSVC(C=1.0)
    if model_name == "SGDClassifier":
        return SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=3000, random_state=42)
    raise ValueError(f"Unsupported model: {model_name}")


def evaluate_predictions(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_sport": f1_score(y_true, y_pred, pos_label=1),
        "f1_politics": f1_score(y_true, y_pred, pos_label=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train, compare, and save text classifiers.")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--output", default="results/model_comparison.csv")
    parser.add_argument("--summary_output", default="results/observations.md")
    parser.add_argument("--model_dir", default="models")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.output)
    summary_path = Path(args.summary_output)
    model_dir = Path(args.model_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    y_train = np.load(data_dir / "y_train.npy")
    y_val = np.load(data_dir / "y_val.npy")
    y_test = np.load(data_dir / "y_test.npy")
    meta = joblib.load(data_dir / "meta.joblib")

    rows = []
    for feat in FEATURE_SETS:
        X_train = sparse.load_npz(data_dir / f"X_train_{feat}.npz")
        X_val = sparse.load_npz(data_dir / f"X_val_{feat}.npz")
        X_test = sparse.load_npz(data_dir / f"X_test_{feat}.npz")

        for model_name in MODEL_NAMES:
            model = build_model(model_name)
            model.fit(X_train, y_train)

            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)
            val_metrics = evaluate_predictions(y_val, val_pred)
            test_metrics = evaluate_predictions(y_test, test_pred)

            model_path = model_dir / f"{feat}__{model_name}.joblib"
            joblib.dump(model, model_path)

            row = {
                "feature": feat,
                "model": model_name,
                "model_path": str(model_path),
                "val_accuracy": val_metrics["accuracy"],
                "val_precision_macro": val_metrics["precision_macro"],
                "val_recall_macro": val_metrics["recall_macro"],
                "val_f1_macro": val_metrics["f1_macro"],
                "test_accuracy": test_metrics["accuracy"],
                "test_precision_macro": test_metrics["precision_macro"],
                "test_recall_macro": test_metrics["recall_macro"],
                "test_f1_macro": test_metrics["f1_macro"],
                "test_f1_sport": test_metrics["f1_sport"],
                "test_f1_politics": test_metrics["f1_politics"],
                "test_confusion_matrix": test_metrics["confusion_matrix"],
            }
            rows.append(row)
            print(feat, model_name, {"test_f1_macro": row["test_f1_macro"], "test_accuracy": row["test_accuracy"]})

    result_df = pd.DataFrame(rows).sort_values(
        ["test_f1_macro", "test_accuracy", "val_f1_macro"], ascending=False
    )
    result_df.to_csv(out_path, index=False)

    best_row = result_df.iloc[0]
    by_feature = result_df.groupby("feature")["test_f1_macro"].max().sort_values(ascending=False)
    by_model = result_df.groupby("model")["test_f1_macro"].max().sort_values(ascending=False)

    summary_lines = [
        "# Observations",
        "",
        "## Dataset",
        f"- Total cleaned samples: {meta['n_samples']}",
        f"- Class counts: {meta['class_counts']}",
        f"- Split sizes: {meta['split_sizes']}",
        "",
        "## Best Overall Configuration",
        f"- Feature: {best_row['feature']}",
        f"- Model: {best_row['model']}",
        f"- Model artifact: {best_row['model_path']}",
        f"- Test Accuracy: {best_row['test_accuracy']:.4f}",
        f"- Test Macro F1: {best_row['test_f1_macro']:.4f}",
        f"- Test Confusion Matrix [[TN, FP], [FN, TP]]: {best_row['test_confusion_matrix']}",
        "",
        "## Best F1 by Feature Representation",
    ]
    for feature_name, score in by_feature.items():
        summary_lines.append(f"- {feature_name}: {score:.4f}")

    summary_lines.extend(["", "## Best F1 by ML Model"])
    for model_name, score in by_model.items():
        summary_lines.append(f"- {model_name}: {score:.4f}")

    summary_lines.extend(
        [
            "",
            "## Notes",
            "- Compare both test_accuracy and test_f1_macro in your report.",
            "- Macro F1 is important if class counts are not perfectly balanced.",
            "- Use confusion matrices to explain common misclassifications.",
            "- For inference, choose a feature and matching model artifact from model_path.",
        ]
    )
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print("\nSaved:", out_path)
    print("Saved:", summary_path)
    print("Saved model artifacts in:", model_dir)
    print(result_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()