from fyers_apiv3 import fyersModel
import webbrowser
import sys
import os
from urllib.parse import urlparse, parse_qs
import json
import time
import hashlib
import requests

# Directory setup
current_dir = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(current_dir)
sys.path.append(PARENT_DIR)
import config
from config import client_id, secret_key, redirect_uri, state, grant_type, response_type, pin

# File paths
TOKEN_STORE_FILE = os.path.join(current_dir, 'token_store.json')

def load_token_store():
    """Load saved tokens and expiry info from disk."""
    if os.path.exists(TOKEN_STORE_FILE):
        try:
            with open(TOKEN_STORE_FILE, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    return {}

def save_token_store(store):
    """Save tokens and expiry info to disk."""
    try:
        with open(TOKEN_STORE_FILE, 'w') as f:
            json.dump(store, f, indent=4)
    except Exception as e:
        print(f"Error saving token store: {e}")

def save_access_token(access_token, refresh_token=None, expires_in=86400):
    """Save access and refresh tokens with expiration info."""
    try:
        now = time.time()
        token_store = {
            "access_token": access_token,
            "expires_at": now + expires_in
        }
        
        if refresh_token:
            token_store["refresh_token"] = refresh_token
            token_store["refresh_expiry"] = now + (15 * 24 * 3600)  # 15 days
        
        save_token_store(token_store)
    except Exception as e:
        print(f"Error saving tokens: {e}")

def is_access_token_valid():
    """Check if the current access token is valid and not expired."""
    try:
        token_store = load_token_store()
        access_token = token_store.get("access_token")
        expires_at = token_store.get("expires_at", 0)
        
        if not access_token or time.time() >= expires_at - 300:  # 5 minutes buffer
            return False
            
        # Quick validation with API
        fyers = fyersModel.FyersModel(token=access_token, is_async=False, client_id=client_id)
        profile_data = fyers.get_profile()
        return "code" in profile_data and profile_data["code"] == 200
    except Exception as e:
        print(f"Access token validation failed: {e}")
        return False

def get_auth_code():
    """Generate a new auth code by redirecting the user."""
    try:
        appSession = fyersModel.SessionModel(
            client_id=client_id,
            redirect_uri=redirect_uri,
            response_type=response_type,
            state=state,
            secret_key=secret_key,
            grant_type=grant_type
        )

        generateTokenUrl = appSession.generate_authcode()
        print(f"Open the following URL in your browser to log in:\n{generateTokenUrl}")
        webbrowser.open(generateTokenUrl)

        redirected_url = input("\nPaste the full redirected URL here: ").strip()
        parsed_url = urlparse(redirected_url)
        auth_code = parse_qs(parsed_url.query).get("auth_code", [None])[0]

        if not auth_code:
            print("Failed to extract auth_code from the URL.")
            return None

        return auth_code
    except Exception as e:
        print(f"Error getting auth code: {e}")
        return None

def generate_access_token(auth_code):
    """Generate a new access token using the auth code."""
    try:
        session = fyersModel.SessionModel(
            client_id=client_id,
            redirect_uri=redirect_uri,
            response_type=response_type,
            state=state,
            secret_key=secret_key,
            grant_type=grant_type,
        )
        session.set_token(auth_code)
        response = session.generate_token()
        
        if "access_token" not in response:
            print(f"Failed to generate access token: {response}")
            return None
            
        return response
    except Exception as e:
        print(f"Error generating access token: {e}")
        return None

def refresh_access_token():
    """Use refresh token to get new access token via Fyers API"""
    try:
        token_store = load_token_store()
        refresh_token = token_store.get("refresh_token")
        refresh_expiry = token_store.get("refresh_expiry", 0)
        
        if not refresh_token or time.time() >= refresh_expiry:
            print("Refresh token expired or not available")
            return None
        
        print("Using refresh token to get new access token...")
        
        # Calculate appIdHash
        app_secret = f"{client_id}:{secret_key}"
        app_id_hash = hashlib.sha256(app_secret.encode()).hexdigest()
        
        # Make refresh request
        response = requests.post(
            "https://api-t1.fyers.in/api/v3/validate-refresh-token",
            headers={"Content-Type": "application/json"},
            json={
                "grant_type": "refresh_token",
                "appIdHash": app_id_hash,
                "refresh_token": refresh_token,
                "pin": pin
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("s") == "ok" and "access_token" in result:
                print("Successfully refreshed access token!")
                access_token = result["access_token"]
                save_access_token(access_token, refresh_token, 86400)
                return access_token
            else:
                print(f"Failed to refresh token: {result}")
                return None
        else:
            print(f"HTTP error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error refreshing access token: {e}")
        return None

def get_access_token():
    """
    Get an access token using priority:
    1) Cached token if valid
    2) Refresh token if available
    3) Existing auth code if available
    4) New auth code flow
    """
    try:
        # Check cached token
        token_store = load_token_store()
        access_token = token_store.get("access_token")
        
        if is_access_token_valid():
            print("Using cached access token.")
            return access_token
        
        # Try refresh token
        refreshed_token = refresh_access_token()
        if refreshed_token:
            return refreshed_token
        
        # Try existing auth code
        if os.path.exists(os.path.join(PARENT_DIR, 'authcode.txt')):
            try:
                with open(os.path.join(PARENT_DIR, 'authcode.txt'), 'r') as f:
                    auth_code = f.read().strip()
                
                if auth_code:
                    print("Trying existing auth code...")
                    token_response = generate_access_token(auth_code)
                    if token_response and "access_token" in token_response:
                        print("Existing auth code worked!")
                        access_token = token_response["access_token"]
                        refresh_token = token_response.get("refresh_token")
                        expires_in = token_response.get("expires_in", 86400)
                        save_access_token(access_token, refresh_token, expires_in)
                        return access_token
            except Exception as e:
                print(f"Error with existing auth code: {e}")
        
        # Generate new auth code
        print("Generating new auth code...")
        auth_code = get_auth_code()
        if not auth_code:
            return None
        
        token_response = generate_access_token(auth_code)
        if not token_response or "access_token" not in token_response:
            return None
            
        access_token = token_response["access_token"]
        refresh_token = token_response.get("refresh_token")
        expires_in = token_response.get("expires_in", 86400)
        
        save_access_token(access_token, refresh_token, expires_in)
        return access_token
    except Exception as e:
        print(f"Error in get_access_token: {e}")
        return None

if __name__ == "__main__":
    token = get_access_token()
    if token:
        print(f"Access token: {token[:10]}...")
    else:
        print("Failed to get access token")
