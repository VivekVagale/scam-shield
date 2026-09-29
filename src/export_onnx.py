"""Export the best model to ONNX so it runs inside a web browser.

Run:  python -m src.export_onnx

Steps:
  1. export distilbert_aug from PyTorch to ONNX (a portable model format)
  2. quantize the weights from 32-bit floats to 8-bit integers: ~4x smaller
     download, a little faster on CPU, tiny accuracy cost
  3. check the quantized model against the original on the whole test set,
     so the web demo is known to behave like the numbers in the README

Output goes to web/models/scam-shield/ in the layout transformers.js expects.
"""

import json
import shutil

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import LABELS, ROOT, load_split
from .finetune import MAX_LEN
from .metrics import report

SRC = ROOT / "models" / "distilbert_aug"
OUT = ROOT / "web" / "models" / "scam-shield"


def main() -> None:
    (OUT / "onnx").mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(SRC)
    model = AutoModelForSequenceClassification.from_pretrained(SRC).eval()

    # name the outputs so the browser shows "smishing", not "LABEL_2"
    model.config.id2label = dict(enumerate(LABELS))
    model.config.label2id = {l: i for i, l in enumerate(LABELS)}
    model.config.save_pretrained(OUT)
    tok.save_pretrained(OUT)

    fp32 = OUT / "onnx" / "model.onnx"
    dummy = tok(["verify your account now"], return_tensors="pt")
    torch.onnx.export(
        model, (dummy["input_ids"], dummy["attention_mask"]), fp32,
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    q8 = OUT / "onnx" / "model_quantized.onnx"
    quantize_dynamic(fp32, q8, weight_type=QuantType.QInt8)
    fp32.unlink()  # the browser only needs the small one
    print(f"quantized model: {q8.stat().st_size / 1e6:.0f} MB")

    # parity check on the full test set
    test = load_split("test")
    sess = ort.InferenceSession(str(q8), providers=["CPUExecutionProvider"])
    ort_pred, torch_pred = [], []
    texts = test["text"].tolist()
    for i in range(0, len(texts), 32):
        enc = tok(texts[i:i + 32], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="np")
        logits = sess.run(["logits"], {"input_ids": enc["input_ids"].astype(np.int64),
                                       "attention_mask": enc["attention_mask"].astype(np.int64)})[0]
        ort_pred += [LABELS[j] for j in logits.argmax(-1)]
        with torch.no_grad():
            pt = model(input_ids=torch.tensor(enc["input_ids"]), attention_mask=torch.tensor(enc["attention_mask"])).logits
        torch_pred += [LABELS[j] for j in pt.argmax(-1).tolist()]

    agree = sum(a == b for a, b in zip(ort_pred, torch_pred)) / len(texts)
    print(f"int8 ONNX agrees with PyTorch on {agree:.1%} of {len(texts)} test messages\n")
    r = report(test["label"], ort_pred, title="int8 ONNX model (test set)")
    r["agreement_with_pytorch"] = round(agree, 4)
    r["size_mb"] = round(q8.stat().st_size / 1e6, 1)
    (ROOT / "results" / "onnx_int8.json").write_text(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
