
import os
from twilio.rest import Client

# ======================
# Load Twilio credentials
# ======================
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token  = os.getenv("TWILIO_AUTH_TOKEN")

client = Client(account_sid, auth_token)

# Twilio phone number (must be purchased or trial-provided)
TWILIO_NUMBER = "**********"  # Replace with your Twilio number
# Recipient (must be verified in trial mode)
TO_NUMBER = "**********"  # Replace with your phone number

try:
    # ======================
    # 1. Send SMS
    # ======================
    sms = client.messages.create(
        body="🚨 ALERT: This is a test SMS notification from Twilio!",
        from_=TWILIO_NUMBER,
        to=TO_NUMBER
    )
    print(f"✅ SMS sent! SID: {sms.sid}")

    # ======================
    # 2. Make Voice Call
    # ======================
    call = client.calls.create(
        twiml="""
        <Response>
            <Say voice="alice">🚨 Emergency Alert! Please take immediate action. This is a test alert from Twilio.</Say>
        </Response>
        """,
        from_=TWILIO_NUMBER,
        to=TO_NUMBER
    )
    print(f"✅ Call initiated! SID: {call.sid}")

except Exception as e:
    print(f"❌ Error: {e}")