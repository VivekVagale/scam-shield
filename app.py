"""Scam Shield demo.  Run:  streamlit run app.py"""

import streamlit as st

from src.predict import available, predict_proba
from src.redflags import find_flags

st.set_page_config(page_title="Scam Shield", page_icon="🛡️", layout="centered")

VERDICT = {
    "smishing": ("🚨 Likely a SCAM", "error", "Do not click links, call numbers or share any code from this message."),
    "spam": ("📢 Promotional spam", "warning", "Probably an ad. Annoying, not usually dangerous."),
    "ham": ("✅ Looks legitimate", "success", "Nothing scam-like found. Still, never share an OTP."),
}

EXAMPLES = {
    "KYC scam": "Dear Customer, your account will be blocked today. Update your KYC now: http://bank-kyc-verify.top/login",
    "Electricity scam": "Your power will be disconnected tonight at 9.30 PM as last month bill is not updated. Call officer 9876512345",
    "Bank alert (real)": "Rs 320.00 debited from A/c XX4321 to VPA zomato@hdfcbank. Not you? Call the number on the back of your card. -HDFC Bank",
    "Friend": "Bro reached college? Bring my charger pls",
}

st.title("🛡️ Scam Shield")
st.caption("Paste an SMS. A fine-tuned DistilBERT model says whether it is a scam, spam or safe, "
           "and simple rules point out the warning signs.")

models = available()
if not models:
    st.error("No trained model found. Run `python -m src.data` then `python -m src.finetune --augment`.")
    st.stop()

cols = st.columns(len(EXAMPLES))
for col, (label, text) in zip(cols, EXAMPLES.items()):
    if col.button(label, use_container_width=True):
        st.session_state["msg"] = text

msg = st.text_area("Message", key="msg", height=120, placeholder="Paste the SMS here")
model = st.selectbox("Model", models, index=len(models) - 1,
                     help="distilbert_aug is the best model; others are kept to compare")

if msg.strip():
    probs = predict_proba([msg.strip()], model)[0]
    label = max(probs, key=probs.get)
    title, kind, advice = VERDICT[label]
    getattr(st, kind)(f"**{title}** ({probs[label]:.0%} confident)\n\n{advice}")

    st.subheader("Model scores")
    for name in ["smishing", "spam", "ham"]:
        st.progress(probs[name], text=f"{name}: {probs[name]:.1%}")

    flags = find_flags(msg)
    st.subheader(f"Warning signs found: {len(flags)}")
    if not flags:
        st.write("None of the common scam patterns matched.")
    for f in flags:
        st.markdown(f"**{f.name}** — `{f.match}`  \n{f.why}")

    if label == "ham" and flags:
        st.info("The model thinks this is safe, but the message has warning signs. Double-check with the "
                "company through its official app or helpline before acting.")

st.divider()
st.caption("A student project, not a guarantee. When in doubt, report suspected fraud at "
           "cybercrime.gov.in or call 1930 (India's cyber fraud helpline).")
