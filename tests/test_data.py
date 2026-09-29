from src.data import LABELS, load_clean, make_splits
from src.metrics import report


def test_clean_data_has_no_duplicates_or_odd_labels():
    df = load_clean()
    assert df["text"].is_unique
    assert set(df["label"]) == set(LABELS)


def test_splits_do_not_share_messages():
    splits = make_splits(load_clean())
    train, val, test = (set(splits[k]["text"]) for k in ("train", "val", "test"))
    assert not train & test
    assert not train & val
    assert not val & test


def test_report_counts_missed_scams_and_false_alarms():
    y_true = ["ham", "ham", "spam", "smishing"]
    y_pred = ["spam", "ham", "ham", "smishing"]
    r = report(y_true, y_pred, title="t")
    assert r["harmful_missed"] == 1
    assert r["false_alarms"] == 1
    assert r["harmful_recall"] == 0.5
