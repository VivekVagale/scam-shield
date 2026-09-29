"""Baseline: TF-IDF features + logistic regression.

Run:  python -m src.baseline

Why this baseline: it trains in seconds on a CPU, every weight can be
inspected, and any fancier model later has to beat this number to be worth it.
"""

import argparse
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .data import ROOT, load_split, load_train
from .metrics import report

MODEL_PATH = ROOT / "models" / "baseline.joblib"


def build() -> Pipeline:
    features = FeatureUnion([
        # whole words and word pairs: "verify your", "account blocked"
        ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        # 3-5 letter pieces inside words: catch "bit.ly", "http", "v3rify", odd spellings
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
    ])
    # class_weight="balanced": smishing is only ~10% of the data, so without this
    # the model could score well by mostly guessing "ham"
    clf = LogisticRegression(max_iter=2000, C=10.0, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


def main(augment: bool = False) -> None:
    name = "baseline_aug" if augment else "baseline"
    model_path = ROOT / "models" / f"{name}.joblib"
    results_path = ROOT / "results" / f"{name}.json"
    train, test = load_train(augment), load_split("test")
    model = build().fit(train["text"], train["label"])

    pred = model.predict(test["text"])
    results = report(test["label"], pred, title=f"{name}: TF-IDF + logistic regression (test set)")

    model_path.parent.mkdir(exist_ok=True)
    results_path.parent.mkdir(exist_ok=True)
    joblib.dump(model, model_path)
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {model_path.relative_to(ROOT)} and {results_path.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--augment", action="store_true", help="add data/augment/india_train.csv to training")
    main(ap.parse_args().augment)
