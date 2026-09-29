# Interview notes: Scam Shield

Every major decision, what else was possible, and why this one won.

## Problem framing

**Three classes, not two.** Spam (ads) and smishing (fraud) have very different
costs. A missed smishing message can empty a bank account; a missed ad is
annoying. Keeping them separate lets the app say "scam" only when it means it.

**Headline metric: scams caught and false alarms, not accuracy.** 83% of the
data is ham, so a model that always says "ham" gets 83% accuracy and is
useless. Macro F1 treats all three classes equally; "caught / false alarms"
is what a user feels.

## Data

**Why this dataset.** It separates smishing from spam (the classic UCI SMS set
only has spam/ham) and is small enough to train in minutes.

**Cleaning.** 34 messages had two different labels: dropped, since there is no
right answer to learn. 106 exact duplicates: removed *before* splitting, because
a copy in train and test lets the model pass by memory. That would inflate the
test score.

**Split 70/15/15, stratified, seed 42.** Stratified keeps the rare classes in
every split. The validation split picks the best epoch; the test split is
touched once at the end. If I chose epochs on the test set, the test number
would stop being an honest estimate.

## Models

**Baseline first (TF-IDF + logistic regression).** Trains in seconds, every
weight is inspectable, and it sets the number a neural model must beat. Word
n-grams catch phrases like "verify your"; character n-grams catch URLs,
misspellings and obfuscation like "v3rify".

**Class weights.** Smishing is ~9% of training data. Without weights the loss
is dominated by ham and the model under-predicts scams. Both models weight
each class by inverse frequency.

**DistilBERT over BERT or a large LLM.** 66M parameters: 40% smaller and 60%
faster than BERT-base with ~97% of its quality; fine-tunes in about 2 minutes
on an RTX 4060 and runs on CPU for the demo. An LLM would be slower, costlier
per message, and harder to evaluate. SMS classification does not need one.

**Hyperparameters.** lr 3e-5 (small, so pre-trained knowledge is adjusted
rather than overwritten), 4 epochs with linear warmup/decay, batch 16, max 128
tokens (covers ~99% of messages), gradient clipping at 1.0 for stability.

## The India finding (best talking point)

Both models trained on the public data flagged 9-11 of 25 Indian legitimate
messages (OTPs, UPI alerts, delivery updates). **Cause: dataset shift.** The
training "ham" is mostly personal chat; nearly every company-style message in
it is spam. So the model learned "sounds like a business = spam".

**Fix: targeted augmentation.** 458 template-generated Indian messages (bank
alerts, OTPs, deliveries, bills, college notices, plus Indian smishing so scam
recall would not collapse). Result: India false alarms 11 → 2, and the
original test F1 went *up* (0.906 → 0.925), which shows it did not just
memorise Indian phrasing.

**Honest limits.** The India test set and the templates were written by the
same person, so the India gain is optimistic. Templates never copy test
messages (the code asserts this), but style overlap is unavoidable. The fix is
a real, forwarded-message test set.

**New failure the fix caused.** "Share the OTP to process refund" is now
missed: the model learned "mentions OTP = safe". That is why the app keeps the
rule-based red flags alongside the model. A rule for "asks you to share an
OTP" is precise and does not depend on training data.

## Why rules *and* a model

The model decides; the rules explain. A user who sees "Asks for OTP / PIN: no
bank ever asks this" learns to spot the next scam without the app. Rules are
also a safety net for patterns the model gets wrong.

## What I would do next

1. Collect 50+ real forwarded scam SMS from classmates for an honest India test.
2. Multilingual model (e.g. MuRIL or IndicBERT) for Hindi/Kannada/Hinglish.
3. Calibrate the probability threshold on validation: trade a few more false
   alarms for fewer missed scams, since a missed scam costs more.
4. Check links against a live phishing blocklist.
