"""Fine-tune DistilBERT to classify SMS as ham / spam / smishing.

Run:  python -m src.finetune

How it works, in plain words:
  1. The tokenizer turns each SMS into word-piece ids (max 128 of them).
  2. DistilBERT (66M parameters, pre-trained on English text) reads them and
     a small new layer on top outputs 3 scores, one per label.
  3. We train the whole thing for a few epochs on our train split. After each
     epoch we score the validation split and keep the best epoch.
  4. Only the kept model is scored on the test split, once, so the test number
     is never used to make choices.
"""

import argparse
import json
import random
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .data import LABELS, ROOT, SEED, load_split, load_train
from .metrics import report

MODEL_NAME = "distilbert-base-uncased"

MAX_LEN = 128      # 99% of messages fit; longer ones are cut
BATCH = 16
EPOCHS = 4
LR = 3e-5          # small, so pre-trained knowledge is adjusted, not overwritten

label_to_id = {l: i for i, l in enumerate(LABELS)}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(tokenizer, texts, labels, shuffle: bool) -> DataLoader:
    enc = tokenizer(list(texts), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
    y = torch.tensor([label_to_id[l] for l in labels])
    ds = torch.utils.data.TensorDataset(enc["input_ids"], enc["attention_mask"], y)
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle)


@torch.no_grad()
def predict(model, loader, device) -> list[str]:
    model.eval()
    out = []
    for ids, mask, _ in loader:
        logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        out += [LABELS[i] for i in logits.argmax(-1).tolist()]
    return out


def main(augment: bool = False) -> None:
    name = "distilbert_aug" if augment else "distilbert"
    out_dir = ROOT / "models" / name
    results_path = ROOT / "results" / f"{name}.json"
    seed_everything(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train, val, test = load_train(augment), load_split("val"), load_split("test")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(LABELS)).to(device)

    train_dl = make_loader(tokenizer, train["text"], train["label"], shuffle=True)
    val_dl = make_loader(tokenizer, val["text"], val["label"], shuffle=False)
    test_dl = make_loader(tokenizer, test["text"], test["label"], shuffle=False)

    # weight rare classes up, same idea as class_weight="balanced" in the baseline
    counts = train["label"].value_counts().reindex(LABELS).to_numpy()
    weights = torch.tensor(len(train) / (len(LABELS) * counts), dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = EPOCHS * len(train_dl)
    sched = get_linear_schedule_with_warmup(optim, int(0.1 * total_steps), total_steps)

    best_f1, best_epoch = -1.0, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        start, running = time.time(), 0.0
        for ids, mask, y in train_dl:
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            loss = loss_fn(logits, y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad()
            running += loss.item()

        val_f1 = f1_score(val["label"], predict(model, val_dl, device), labels=LABELS, average="macro")
        print(f"epoch {epoch}  loss {running / len(train_dl):.4f}  val macro F1 {val_f1:.4f}  ({time.time() - start:.0f}s)")
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)

    print(f"\nbest epoch {best_epoch} (val macro F1 {best_f1:.4f}); scoring it on test\n")
    model = AutoModelForSequenceClassification.from_pretrained(out_dir).to(device)
    pred = predict(model, test_dl, device)
    results = report(test["label"], pred, title=f"{name} (test set)")
    results.update({"best_epoch": best_epoch, "val_macro_f1": round(best_f1, 4)})
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {out_dir.relative_to(ROOT)} and {results_path.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--augment", action="store_true", help="add data/augment/india_train.csv to training")
    main(ap.parse_args().augment)
