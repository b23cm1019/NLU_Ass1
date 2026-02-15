import argparse
import re
from pathlib import Path

import joblib
import pandas as pd


FEATURE_CHOICES = ["bow", "ngrams_2_3", "tfidf"]
MODEL_CHOICES = ["MultinomialNB", "LogisticRegression", "LinearSVC", "SGDClassifier"]
ID_TO_LABEL = {0: "politics", 1: "sport"}


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_artifacts(data_dir: Path, model_dir: Path, feature: str, model: str):
    vectorizers = joblib.load(data_dir / "vectorizers.joblib")
    if feature not in vectorizers:
        raise ValueError(f"Feature '{feature}' not found. Available: {list(vectorizers.keys())}")

    model_path = model_dir / f"{feature}__{model}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}. Run train_models.py first."
        )

    clf = joblib.load(model_path)
    vectorizer = vectorizers[feature]
    return vectorizer, clf, model_path


def predict_one(raw_text: str, vectorizer, clf) -> dict:
    cleaned = clean_text(raw_text)
    X = vectorizer.transform([cleaned])
    pred = int(clf.predict(X)[0])

    result = {
        "pred_label_id": pred,
        "pred_label": ID_TO_LABEL[pred],
        "clean_text_preview": cleaned[:220],
    }

    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)[0]
        result["prob_politics"] = float(proba[0])
        result["prob_sport"] = float(proba[1])

    return result


def classify_text_or_file(args):
    vectorizer, clf, model_path = load_artifacts(Path(args.data_dir), Path(args.model_dir), args.feature, args.model)

    if args.text:
        result = predict_one(args.text, vectorizer, clf)
        print("Feature:", args.feature)
        print("Model:", args.model)
        print("Model artifact:", model_path)
        print("Prediction:", result["pred_label"], f"({result['pred_label_id']})")
        if "prob_politics" in result:
            print(
                "Probabilities:",
                f"politics={result['prob_politics']:.4f}",
                f"sport={result['prob_sport']:.4f}",
            )
        print("Clean preview:", result["clean_text_preview"])
        return

    if args.file:
        in_path = Path(args.file)
        if not in_path.exists():
            raise FileNotFoundError(f"Input file not found: {in_path}")

        if in_path.suffix.lower() == ".csv":
            df = pd.read_csv(in_path)
            if "text" not in df.columns:
                raise ValueError("CSV input must contain a 'text' column")

            clean_series = df["text"].astype(str).map(clean_text)
            X = vectorizer.transform(clean_series.tolist())
            pred_ids = clf.predict(X)

            out = df.copy()
            out["pred_label_id"] = pred_ids.astype(int)
            out["pred_label"] = out["pred_label_id"].map(ID_TO_LABEL)

            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X)
                out["prob_politics"] = probs[:, 0]
                out["prob_sport"] = probs[:, 1]

            out_path = Path(args.output_csv or "results/predictions.csv")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(out_path, index=False)
            print("Saved predictions:", out_path)
            print("Counts:")
            print(out["pred_label"].value_counts())
            return

        raw_text = in_path.read_text(encoding="utf-8", errors="ignore")
        result = predict_one(raw_text, vectorizer, clf)
        print("Feature:", args.feature)
        print("Model:", args.model)
        print("Model artifact:", model_path)
        print("Prediction:", result["pred_label"], f"({result['pred_label_id']})")
        if "prob_politics" in result:
            print(
                "Probabilities:",
                f"politics={result['prob_politics']:.4f}",
                f"sport={result['prob_sport']:.4f}",
            )
        print("Clean preview:", result["clean_text_preview"])
        return

    raise ValueError("Provide either --text or --file")


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify input text/document as politics or sport.")
    parser.add_argument("--feature", required=True, choices=FEATURE_CHOICES)
    parser.add_argument("--model", required=True, choices=MODEL_CHOICES)
    parser.add_argument("--text", default="")
    parser.add_argument("--file", default="")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--output_csv", default="")
    args = parser.parse_args()

    if bool(args.text) == bool(args.file):
        raise ValueError("Use exactly one of --text or --file")

    classify_text_or_file(args)


if __name__ == "__main__":
    main()