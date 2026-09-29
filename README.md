# Scam Shield

Detects scam SMS: sorts a message into **ham** (legitimate), **spam** (unwanted
promotion) or **smishing** (phishing by SMS: fake KYC, refund, prize, delivery links).

Work in progress.

## Setup

```powershell
uv venv --python 3.11 .venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```

## Run

```powershell
python -m src.data        # download (245 KB), clean, split 70/15/15
python -m src.baseline    # train TF-IDF + logistic regression, score on test
python -m pytest
```

## Results so far (test set, 870 messages)

| Model | Macro F1 | Scams caught | False alarms on ham |
|---|---|---|---|
| TF-IDF + logistic regression | 0.920 | 131 / 144 (91.0%) | 0 / 726 |

## Data

Mishra, S. & Soni, D. (2022). *SMS Phishing Dataset for Machine Learning and
Pattern Recognition*. Mendeley Data, V1. DOI: 10.17632/f45bkkt8pr.1

5,971 raw messages become 5,797 after cleaning: 34 messages that appear with two
different labels are dropped, and 106 exact duplicates are removed so no message
can sit in both train and test.
