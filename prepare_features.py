import argparse
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_label(label: str) -> int:
    val = str(label).strip().lower()
    if val in {"politics", "politic", "0"}:
        return 0
    if val in {"sport", "sports", "1"}:
        return 1
    raise ValueError(f"Unexpected label: {label}")


def save_sparse(path: Path, matrix) -> None:
    sparse.save_npz(path, matrix)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess text and build BoW/TF-IDF/n-gram features.")
    parser.add_argument("--input", default="data/raw/guardian_articles.csv")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--min_df", type=int, default=2)
    parser.add_argument("--max_df", type=float, default=0.95)
    parser.add_argument("--min_tokens", type=int, default=30)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    required = {"label", "text"}
    if not required.issubset(df.columns):
        raise ValueError(f"Input CSV must contain columns: {required}")

    df = df.dropna(subset=["label", "text"]).copy()
    if "url" in df.columns:
        df = df.drop_duplicates(subset=["url"]).copy()
    df = df.drop_duplicates(subset=["text"]).copy()
    df["clean_text"] = df["text"].apply(clean_text)
    df["token_count"] = df["clean_text"].str.split().str.len()
    df = df[df["token_count"] >= args.min_tokens].copy()
    df = df[df["clean_text"].str.len() > 0].copy()

    df["label_id"] = df["label"].apply(normalize_label).astype(int)
    label_map = {"politics": 0, "sport": 1}

    X = df["clean_text"].values.astype(str)
    y = df["label_id"].astype(int).values

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=args.random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=args.random_state, stratify=y_tmp
    )

    vectorizers = {
        "bow": CountVectorizer(
            ngram_range=(1, 1), min_df=args.min_df, max_df=args.max_df, stop_words="english"
        ),
        "ngrams_2_3": CountVectorizer(
            ngram_range=(2, 3), min_df=args.min_df, max_df=args.max_df, stop_words="english"
        ),
        "tfidf": TfidfVectorizer(
            ngram_range=(1, 1),
            min_df=args.min_df,
            max_df=args.max_df,
            stop_words="english",
            sublinear_tf=True,
        ),
    }

    feature_shapes = {}
    vocab_sizes = {}
    for name, vec in vectorizers.items():
        X_train_vec = vec.fit_transform(X_train)
        X_val_vec = vec.transform(X_val)
        X_test_vec = vec.transform(X_test)

        save_sparse(out_dir / f"X_train_{name}.npz", X_train_vec)
        save_sparse(out_dir / f"X_val_{name}.npz", X_val_vec)
        save_sparse(out_dir / f"X_test_{name}.npz", X_test_vec)

        feature_shapes[name] = {
            "train_shape": X_train_vec.shape,
            "val_shape": X_val_vec.shape,
            "test_shape": X_test_vec.shape,
        }
        vocab_sizes[name] = len(vec.get_feature_names_out())

    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "y_val.npy", y_val)
    np.save(out_dir / "y_test.npy", y_test)

    np.save(out_dir / "X_train_text.npy", X_train)
    np.save(out_dir / "X_val_text.npy", X_val)
    np.save(out_dir / "X_test_text.npy", X_test)

    pd.DataFrame(
        {
            "text": df["text"],
            "clean_text": df["clean_text"],
            "label": df["label"],
            "label_id": df["label_id"],
            "token_count": df["token_count"],
        }
    ).to_csv(out_dir / "clean_dataset.csv", index=False)

    meta = {
        "label_map": label_map,
        "feature_shapes": feature_shapes,
        "vocab_sizes": vocab_sizes,
        "n_samples": len(df),
        "class_counts": df["label"].value_counts().to_dict(),
        "split_sizes": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "min_tokens": args.min_tokens,
    }

    joblib.dump(vectorizers, out_dir / "vectorizers.joblib")
    joblib.dump(meta, out_dir / "meta.joblib")

    print("Saved processed outputs to", out_dir)
    print("Class counts:", meta["class_counts"])
    print("Feature shapes:")
    for k, v in feature_shapes.items():
        print(k, v)


if __name__ == "__main__":
    main()
