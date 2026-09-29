from src.redflags import find_flags


def names(text):
    return {f.name for f in find_flags(text)}


def test_otp_request_is_flagged():
    assert "Asks for OTP / PIN" in names("Share the OTP received to process your refund")


def test_plain_otp_message_is_not_flagged_as_request():
    assert "Asks for OTP / PIN" not in names("482913 is your OTP for login. Do not share it with anyone.")


def test_upi_collect_link_is_flagged():
    assert "UPI collect / pay link" in names("Return money using upi://pay?pa=refund@ybl&am=5000")


def test_private_number_is_flagged():
    assert "Asks you to call or chat a private number" in names("Call officer 9876543210 now")


def test_friend_message_has_no_flags():
    assert names("Bro where are you, canteen?") == set()
