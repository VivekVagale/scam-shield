"""Baseline: TF-IDF features + logistic regression.

Run:  python -m src.baseline

Why this baseline: it trains in seconds on a CPU, every weight can be
inspected, and any fancier model later has to beat this number to be worth it.
"""

import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .data import ROOT, load_split
from .metrics import report

MODEL_PATH = ROOT / "models" / "baseline.joblib"
RESULTS_PATH = ROOT / "results" / "baseline.json"


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


def main() -> None:
    train, test = load_split("train"), load_split("test")
    model = build().fit(train["text"], train["label"])

    pred = model.predict(test["text"])
    results = report(test["label"], pred, title="Baseline: TF-IDF + logistic regression (test set)")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {MODEL_PATH.relative_to(ROOT)} and {RESULTS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
