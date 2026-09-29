"""Score every model on the test split and on each CSV in eval/.

Run:  python -m src.evaluate
Writes results/summary.json and prints a comparison table plus the misses.
"""

import json

import pandas as pd

from .data import ROOT, load_split
from .metrics import HARMFUL, report
from .predict import available, predict

EVAL_DIR = ROOT / "eval"
SUMMARY_PATH = ROOT / "results" / "summary.json"


def datasets() -> dict[str, pd.DataFrame]:
    sets = {"test": load_split("test")}
    for csv in sorted(EVAL_DIR.glob("*.csv")):
        sets[csv.stem] = pd.read_csv(csv, encoding="utf-8")
    return sets


def main() -> None:
    summary = {}
    for set_name, df in datasets().items():
        for model in available():
            pred = predict(df["text"].tolist(), model)
            print()
            r = report(df["label"], pred, title=f"{model} on {set_name} ({len(df)} messages)")
            summary[f"{model}/{set_name}"] = r

            if set_name != "test":
                missed = [t for t, y, p in zip(df["text"], df["label"], pred) if y in HARMFUL and p not in HARMFUL]
                flagged = [t for t, y, p in zip(df["text"], df["label"], pred) if y not in HARMFUL and p in HARMFUL]
                for t in missed:
                    print(f"  MISSED  {t[:110]}")
                for t in flagged:
                    print(f"  FALSE ALARM  {t[:110]}")

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 78)
    print(f"{'model / set':32}{'macro F1':>10}{'caught':>12}{'recall':>9}{'false alarms':>15}")
    for key, r in summary.items():
        caught = f"{r['n_harmful'] - r['harmful_missed']}/{r['n_harmful']}"
        print(f"{key:32}{r['macro_f1']:>10.3f}{caught:>12}{r['harmful_recall']:>9.3f}{r['false_alarms']:>15}")
    print(f"\nsaved {SUMMARY_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
