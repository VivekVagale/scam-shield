"""Download, clean and split the SMS phishing dataset.

Run:  python -m src.data

Source: Mishra & Soni (2022), "SMS Phishing Dataset for Machine Learning and
Pattern Recognition", Mendeley Data, DOI 10.17632/f45bkkt8pr.1
"""

import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
SPLIT_DIR = ROOT / "data" / "splits"
RAW_CSV = RAW_DIR / "Dataset_5971.csv"
ZIP_URL = (
    "https://data.mendeley.com/public-files/datasets/f45bkkt8pr/files/"
    "edb361de-918d-469f-9106-e84823830665/file_downloaded"
)

LABELS = ["ham", "spam", "smishing"]
SEED = 42  # fixed so every model is scored on the exact same test messages


def download() -> None:
    """Fetch the zip (245 KB) once and unpack the CSV."""
    if RAW_CSV.exists():
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "Dataset_5971.zip"
    print(f"downloading {ZIP_URL}")
    urllib.request.urlretrieve(ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAW_DIR)


def load_clean() -> pd.DataFrame:
    """Return one row per unique message with a lowercase label.

    Cleaning steps, each printed so the numbers can be quoted later:
      1. lowercase labels   ("Spam" and "spam" are the same class)
      2. drop messages that appear with two different labels (ambiguous)
      3. drop exact duplicates (a copy in train and test would leak the answer)
    """
    download()
    df = pd.read_csv(RAW_CSV, encoding="utf-8")[["LABEL", "TEXT"]]
    df.columns = ["label", "text"]
    df["label"] = df["label"].str.strip().str.lower()
    df["text"] = df["text"].astype(str).str.strip()
    print(f"raw rows:            {len(df)}")

    labels_per_text = df.groupby("text")["label"].nunique()
    conflicting = labels_per_text[labels_per_text > 1].index
    df = df[~df["text"].isin(conflicting)]
    print(f"conflicting texts:   {len(conflicting)} dropped")

    before = len(df)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    print(f"duplicate rows:      {before - len(df)} dropped")
    print(f"clean rows:          {len(df)}")

    unknown = set(df["label"]) - set(LABELS)
    assert not unknown, f"unexpected labels: {unknown}"
    return df


def make_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """70 / 15 / 15 train / val / test, stratified so each keeps the class mix."""
    train, rest = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=SEED)
    return {"train": train, "val": val, "test": test}


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(SPLIT_DIR / f"{name}.csv", encoding="utf-8")


def load_train(augment: bool = False) -> pd.DataFrame:
    """Train split, optionally plus the synthetic Indian messages from src.augment."""
    train = load_split("train")
    if not augment:
        return train
    extra = pd.read_csv(ROOT / "data" / "augment" / "india_train.csv", encoding="utf-8")
    return pd.concat([train, extra], ignore_index=True)


def main() -> None:
    df = load_clean()
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    print()
    for name, part in make_splits(df).items():
        part.to_csv(SPLIT_DIR / f"{name}.csv", index=False, encoding="utf-8")
        counts = part["label"].value_counts().reindex(LABELS).to_dict()
        print(f"{name:5} {len(part):5}  {counts}")


if __name__ == "__main__":
    main()
