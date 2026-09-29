"""One scoring function shared by every model, so the numbers are comparable."""

from sklearn.metrics import classification_report, confusion_matrix, f1_score

from .data import LABELS

HARMFUL = {"spam", "smishing"}


def report(y_true, y_pred, title: str) -> dict:
    """Print and return 3-class scores plus the binary harmful-vs-ham view.

    The binary view is what a user actually cares about: did a scam get
    through (missed = false negative) or did a real message get flagged
    (false alarm = false positive)?
    """
    y_true, y_pred = list(y_true), list(y_pred)
    print(title)
    print("-" * len(title))
    print(classification_report(y_true, y_pred, labels=LABELS, digits=3, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    print("confusion matrix (rows = true, columns = predicted)")
    print(" " * 10 + "".join(f"{l:>10}" for l in LABELS))
    for label, row in zip(LABELS, cm):
        print(f"{label:>10}" + "".join(f"{n:>10}" for n in row))

    t = [y in HARMFUL for y in y_true]
    p = [y in HARMFUL for y in y_pred]
    missed = sum(a and not b for a, b in zip(t, p))
    false_alarms = sum(b and not a for a, b in zip(t, p))
    harmful_recall = 1 - missed / sum(t)
    print(f"\nharmful vs ham: caught {sum(t) - missed}/{sum(t)} harmful "
          f"(recall {harmful_recall:.3f}), {false_alarms} false alarms on {len(t) - sum(t)} ham")

    return {
        "macro_f1": round(f1_score(y_true, y_pred, labels=LABELS, average="macro"), 4),
        "per_class": {
            k: {m: round(v, 4) for m, v in d.items()}
            for k, d in classification_report(
                y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
            ).items() if k in LABELS
        },
        "confusion_matrix": {"labels": LABELS, "matrix": cm.tolist()},
        "n_harmful": sum(t),
        "harmful_recall": round(harmful_recall, 4),
        "harmful_missed": missed,
        "false_alarms": false_alarms,
        "n_test": len(y_true),
    }
