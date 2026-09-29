"""Plain rules that point at the parts of a message that look like a scam.

These do not decide the verdict (the model does). They explain it, so a user
learns what to look for. Each rule is a regex plus a sentence a non-technical
person can understand.
"""

import re
from dataclasses import dataclass


@dataclass
class Flag:
    name: str
    why: str
    match: str


RULES = [
    ("Asks for OTP / PIN",
     r"\b(share|send|tell|give)\b[^.]{0,30}?\b(otp|pin|cvv|password)\b",
     "No bank, app or company ever asks you to share an OTP, PIN or CVV."),
    ("UPI collect / pay link",
     r"upi://\S+",
     "A UPI link sends money FROM you. You never need to pay to receive money."),
    ("Short or unofficial link",
     r"\b(bit\.ly|tinyurl\.com|cutt\.ly|t\.me|wa\.me|is\.gd|smsg\.io)/\S+|https?://[^\s/]+\.(top|site|online|info|xyz|co|in)/?\S*",
     "Shortened or look-alike links hide where they go. Official links use the company's own domain."),
    ("Threat or deadline",
     r"\b(blocked?|suspend(ed)?|deactivat\w*|disconnect\w*|expire[sd]?|lapse|arrest\w*|fir|legal action|within \d+ ?(hrs?|hours|minutes)|tonight|immediately|urgent)\b",
     "Scams create panic so you act before you think."),
    ("Prize, refund or easy money",
     r"\b(won|winner|lottery|lucky draw|cashback|refund|reward points?|daily income|earn rs|part time job|work from home)\b",
     "Unexpected money is the most common bait."),
    ("Asks you to call or chat a private number",
     r"\b(call|contact|whatsapp|telegram)\b[^.]{0,40}?\b[6-9]\d{9}\b",
     "Real companies use listed helplines, not personal 10-digit mobile numbers."),
    ("Asks for KYC / PAN / Aadhaar update",
     r"\b(kyc|pan|aadhaar)\b[^.]{0,40}?\b(update|link|verify|pending|expired?)\b|\b(update|link|verify)\b[^.]{0,30}?\b(kyc|pan|aadhaar)\b",
     "KYC is done in the bank's app or branch, never through an SMS link."),
]

_COMPILED = [(name, re.compile(rx, re.IGNORECASE), why) for name, rx, why in RULES]


def find_flags(text: str) -> list[Flag]:
    flags = []
    for name, rx, why in _COMPILED:
        m = rx.search(text)
        if m:
            flags.append(Flag(name, why, m.group(0)))
    return flags
