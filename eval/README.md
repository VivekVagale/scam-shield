# India test set

`india_synthetic.csv` holds **60 synthetic messages**: 25 smishing, 10 spam, 25 ham.
They are **not real messages**. They were written for this project to imitate
scam types that Indian banks, police cyber cells and news reports commonly warn
about (KYC or PAN block, electricity disconnection, courier held by customs,
Telegram part-time jobs, wrong UPI credit, lottery, fake "digital arrest")
alongside everyday Indian messages (OTPs, UPI debit alerts, food delivery,
friends in Hinglish). All phone numbers, links and account numbers are made up.

Why it exists: the training data is mostly non-Indian. This set checks whether
the model transfers to Indian-style scams. It is never used for training.

Replace or extend it with **real** forwarded scam SMS (names and numbers
removed) in `india_real.csv`, using the same columns. `python -m src.evaluate`
scores every CSV in this folder.
