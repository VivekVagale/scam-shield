"""Generate synthetic Indian-style training messages from templates.

Run:  python -m src.augment

Why: the Mendeley data has almost no Indian transactional SMS (OTPs, UPI
alerts, delivery updates), so models learn "sounds like a company = spam" and
flag real bank alerts. These templates add that missing kind of ham, plus some
Indian smishing so scam recall does not drop.

Honesty rule: templates are written separately from eval/india_synthetic.csv
and never copy its messages, and eval/ is never used for training. Both were
still written by the same person, so gains on that eval set are optimistic
until checked on real forwarded messages.
"""

import random

import pandas as pd

from .data import ROOT, SEED

OUT_PATH = ROOT / "data" / "augment" / "india_train.csv"

BANKS = ["ICICI Bank", "Kotak Bank", "PNB", "Canara Bank", "Bank of Baroda", "IDFC FIRST", "Union Bank", "Yes Bank"]
MERCHANTS = ["zepto", "blinkit", "bigbasket", "myntra", "bookmyshow", "irctc", "rapido", "dominos", "nykaa", "meesho"]
NAMES = ["ANIL", "PRIYA S", "KIRAN M", "DEEPA R", "ARJUN", "MEENA K", "SANJAY", "NIKHIL P"]
APPS = ["Zepto", "Blinkit", "BigBasket", "Dunzo", "Myntra", "Meesho", "Nykaa", "Ajio"]
FAKE_DOMAINS = ["secure-verify-{b}.in", "{b}-kyc-update.co", "{b}-support.online", "update-{b}-account.site"]


def amount(r):
    return f"{r.choice([49, 99, 150, 230, 499, 675, 1200, 2450, 3999, 8600]):,}.{r.choice(['00', '50'])}"


def acct(r):
    return f"XX{r.randint(1000, 9999)}"


HAM = [
    lambda r: f"{r.randint(100000, 999999)} is the OTP for your transaction at {r.choice(MERCHANTS).title()}. Valid for 10 min. Never share OTP. -{r.choice(BANKS)}",
    lambda r: f"Use OTP {r.randint(1000, 9999)} to log in to {r.choice(APPS)}. Do not share this code with anyone.",
    lambda r: f"INR {amount(r)} spent on your {r.choice(BANKS)} card {acct(r)} at {r.choice(MERCHANTS).upper()} on {r.randint(1, 28)}-Sep. Not you? Block card from the app.",
    lambda r: f"A/c {acct(r)} debited INR {amount(r)} for UPI to {r.choice(MERCHANTS)}@okaxis. Ref {r.randint(10**9, 10**10 - 1)}. -{r.choice(BANKS)}",
    lambda r: f"A/c {acct(r)} credited with INR {amount(r)} from {r.choice(NAMES)} via UPI. Avl bal INR {amount(r)}. -{r.choice(BANKS)}",
    lambda r: f"Your {r.choice(APPS)} order has been shipped and will arrive by {r.choice(['tomorrow', 'Friday', 'Monday'])}. Track it in the app.",
    lambda r: f"Order delivered! Your {r.choice(APPS)} package was handed to you at {r.randint(1, 12)}:{r.choice(['05', '20', '45'])} PM. Rate us in the app.",
    lambda r: f"Your Rapido captain {r.choice(['Ravi', 'Manju', 'Imran', 'Shiva'])} is {r.randint(2, 8)} min away. Bike KA{r.randint(1, 60):02d}X{r.randint(1000, 9999)}.",
    lambda r: f"Your water bill of Rs {amount(r)} is generated. Due date {r.randint(1, 28)}-Oct. Pay via the official BWSSB portal.",
    lambda r: f"Your {r.choice(['Vi', 'BSNL', 'Airtel', 'Jio'])} plan expires in {r.randint(1, 5)} days. Recharge in the official app to continue services.",
    lambda r: f"Your booking for {r.choice(['Devara', 'Kantara 2', 'Pushpa 3', 'Jawan'])} at PVR is confirmed. Seats {r.choice('ABCDEFG')}{r.randint(1, 20)}. Show at {r.randint(1, 10)} PM.",
    lambda r: f"Refund of Rs {amount(r)} for your cancelled {r.choice(APPS)} order has been initiated. It will reach your account in 3-5 working days.",
    lambda r: f"Dear Student, {r.choice(['IA-1', 'IA-2', 'IA-3'])} for {r.choice(['DBMS', 'OS', 'ML', 'CN', 'DAA'])} will be held on {r.randint(1, 28)}-Oct. Check the timetable on the college portal.",
    lambda r: f"Your LPG cylinder booking is confirmed. Booking no {r.randint(10**6, 10**7 - 1)}. Delivery in 2 days. -Indane",
    lambda r: f"Your FASTag {acct(r)} was charged Rs {r.choice([45, 90, 135, 230])} at {r.choice(['Nelamangala', 'Hoskote', 'Attibele'])} toll plaza. Bal Rs {amount(r)}.",
    lambda r: f"Your appointment with Dr {r.choice(['Rao', 'Shetty', 'Iyer', 'Patil'])} is confirmed for {r.randint(1, 28)}-Oct at {r.randint(9, 12)}:30 AM.",
    lambda r: f"{r.choice(['Bro', 'Da', 'Machaa'])} {r.choice(['send the lab record pic', 'where are you', 'canteen?', 'did you submit the assignment', 'what time is the bus'])}",
    lambda r: f"{r.choice(['Ok', 'Sure', 'Done', 'Haan'])} {r.choice(['will call you after class', 'reached home', 'see you tomorrow', 'I am coming in 10 min'])}",
]

SMISHING = [
    lambda r: f"Dear customer your {r.choice(BANKS)} account is suspended due to incomplete KYC. Update now: http://{r.choice(FAKE_DOMAINS).format(b=r.choice(['icici', 'kotak', 'pnb', 'canara']))}",
    lambda r: f"Your {r.choice(['gas', 'electricity', 'water'])} connection will be disconnected tonight due to unpaid bill. Call officer {r.choice(['98', '97', '70', '81'])}{r.randint(10**7, 10**8 - 1)} now",
    lambda r: f"Congratulations! You won Rs {r.choice(['5,00,000', '25,00,000', '10,00,000'])} in {r.choice(['Jio', 'Airtel', 'Flipkart'])} lucky draw. Pay registration fee to claim: {r.choice(['bit.ly', 'tinyurl.com', 'cutt.ly'])}/{r.randint(1000, 9999)}w",
    lambda r: f"Part time job: earn Rs {r.choice(['2000', '4000', '6000'])} daily by {r.choice(['liking videos', 'rating products', 'reviewing hotels'])}. Message HR on Telegram t.me/{r.choice(['hr_neha', 'jobdesk_ind', 'work_asha'])}{r.randint(1, 99)}",
    lambda r: f"Your parcel is held at customs for {r.choice(['drugs', 'fake passports', 'illegal items'])}. Contact {r.choice(['CBI', 'Narcotics', 'Customs'])} officer on {r.choice(['99', '90', '88'])}{r.randint(10**7, 10**8 - 1)} to avoid arrest",
    lambda r: f"Rs {amount(r)} sent to your UPI by mistake. Kindly return it immediately using this link upi://pay?pa={r.choice(['help', 'refund', 'return'])}{r.randint(1, 99)}@ybl",
    lambda r: f"Share the OTP you received to {r.choice(['activate your new card', 'stop the unauthorized debit', 'claim your cashback'])}. Our executive will call you. -{r.choice(BANKS)} team",
]


def generate(n_ham: int = 360, n_smishing: int = 120) -> pd.DataFrame:
    r = random.Random(SEED)
    rows = [("ham", r.choice(HAM)(r)) for _ in range(n_ham)]
    rows += [("smishing", r.choice(SMISHING)(r)) for _ in range(n_smishing)]
    df = pd.DataFrame(rows, columns=["label", "text"]).drop_duplicates("text")

    held_out = set(pd.read_csv(ROOT / "eval" / "india_synthetic.csv")["text"])
    assert not held_out & set(df["text"]), "augment must not copy eval messages"
    return df


def main() -> None:
    df = generate()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"{len(df)} messages -> {OUT_PATH.relative_to(ROOT)}  {df['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
