
import webbrowser
from fyers_apiv3 import fyersModel
import os
from pathlib import Path
from dotenv import load_dotenv

DATA_DIR = Path(__file__).resolve().parent

# .env lives in data/ -- load it explicitly (load_dotenv() with no args only
# checks the current working directory, which is wrong when run from the root).
load_dotenv(DATA_DIR / ".env")

# --- Configuration ---
APP_ID = os.getenv("FYERS_APP_ID")
SECRET_KEY = os.getenv("FYERS_SECRET_KEY")
REDIRECT_URI = os.getenv("REDIRECT_URI")
TOKEN_PATH = DATA_DIR / "access_token.txt"

if not all([APP_ID, SECRET_KEY, REDIRECT_URI]):
    raise SystemExit(
        "Missing FYERS_APP_ID / FYERS_SECRET_KEY / REDIRECT_URI in data/.env"
    )


def generate_access_token():
    session = fyersModel.SessionModel(
        client_id=APP_ID,
        secret_key=SECRET_KEY,
        redirect_uri=REDIRECT_URI,
        response_type="code",
        grant_type="authorization_code"
    )

    # 1. Generate authorization URL
    auth_url = session.generate_authcode()
    print("Opening browser for FYERS login...")
    webbrowser.open(auth_url)

    # 2. Get auth code from user after redirect
    print("\n1. Log in via the browser window.")
    print("2. After success, copy the 'auth_code' parameter from the redirected URL bar.")
    auth_code = input("\nPaste the auth_code here: ").strip()

    # 3. Exchange auth code for Access Token
    session.set_token(auth_code)
    response = session.generate_token()

    if response.get("s") == "ok":
        token = response["access_token"]
        print("\nSUCCESS! Your Access Token:")
        print(token)

        # Save token where data.py's load_access_token() reads it first
        TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TOKEN_PATH, "w", encoding="utf-8") as f:
            f.write(token)
        print(f"\nSaved token to '{TOKEN_PATH}'.")
        return token
    else:
        print(f"\nFailed to generate token: {response}")
        return None


if __name__ == "__main__":
    generate_access_token()