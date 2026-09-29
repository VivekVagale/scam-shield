"""Load a trained model and classify messages. Used by the evaluator and the app."""

from functools import lru_cache

import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import LABELS, ROOT
from .finetune import MAX_LEN

MODELS_DIR = ROOT / "models"
MODELS = ["baseline", "baseline_aug", "distilbert", "distilbert_aug"]


def available() -> list[str]:
    """Models that have actually been trained on this machine."""
    return [m for m in MODELS if (MODELS_DIR / f"{m}.joblib").exists() or (MODELS_DIR / m).is_dir()]


@lru_cache(maxsize=None)
def _sklearn(name: str):
    return joblib.load(MODELS_DIR / f"{name}.joblib")


@lru_cache(maxsize=None)
def _transformer(name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODELS_DIR / name)
    model = AutoModelForSequenceClassification.from_pretrained(MODELS_DIR / name).to(device).eval()
    return tok, model, device


def predict_proba(texts: list[str], model: str = "distilbert_aug") -> list[dict[str, float]]:
    """Return one {label: probability} dict per message."""
    if model.startswith("baseline"):
        clf = _sklearn(model)
        order = list(clf.classes_)
        return [{l: float(p[order.index(l)]) for l in LABELS} for p in clf.predict_proba(texts)]

    tok, net, device = _transformer(model)
    out = []
    for i in range(0, len(texts), 64):
        enc = tok(texts[i:i + 64], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            probs = net(**enc).logits.softmax(-1).cpu().tolist()
        out += [dict(zip(LABELS, p)) for p in probs]
    return out


def predict(texts: list[str], model: str = "distilbert_aug") -> list[str]:
    return [max(p, key=p.get) for p in predict_proba(texts, model)]
