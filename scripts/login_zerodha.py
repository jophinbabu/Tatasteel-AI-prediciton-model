import os
from kiteconnect import KiteConnect
from dotenv import load_dotenv, set_key

def login():
    # Load credentials from .env
    load_dotenv()
    api_key = os.getenv("KITE_API_KEY")
    api_secret = os.getenv("KITE_API_SECRET")
    
    if not api_key or not api_secret:
        print("Error: KITE_API_KEY or KITE_API_SECRET not found in .env file.")
        return

    # Initialize KiteConnect
    kite = KiteConnect(api_key=api_key)

    # 1. Generate Login URL
    login_url = kite.login_url()
    print("==================================================")
    print("Login to Zerodha required (Tokens expire daily at 6:00 AM)")
    print("==================================================")
    print("1. Click this link and log in to your Zerodha account:")
    print(f"\n   {login_url}\n")
    print("2. After successful login, you will be redirected to an empty or error page at 127.0.0.1.")
    print("3. Look at the URL in your browser bar. It will look like this:")
    print("   http://127.0.0.1/?request_token=XXXXXXX&action=login")
    print("4. Copy the long 'request_token' value (the text replacing XXXXXXX).")
    print("==================================================")

    # 2. Get Request Token from user
    request_token = input("\nPaste the request_token here: ").strip()

    if not request_token:
        print("No request token provided. Exiting.")
        return

    # 3. Generate Session and Access Token
    try:
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data["access_token"]
        
        # Save to .env file
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
        set_key(env_path, "KITE_ACCESS_TOKEN", access_token)
        
        print("\n✅ SUCCESS!")
        print(f"Access Token generated and saved to .env file.")
        print("Your AI is now fully authorized to trade live today.")
        
    except Exception as e:
        print(f"\n❌ Login Failed: {str(e)}")
        print("Make sure you are copying the correct request_token and trying quickly (it expires in a few minutes).")

if __name__ == "__main__":
    login()
