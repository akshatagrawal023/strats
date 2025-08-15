from dotenv import load_dotenv
import os

load_dotenv()

client_id = os.getenv("FYERS_CLIENT_ID")
secret_key = os.getenv("FYERS_SECRET_KEY")
redirect_uri = "https://127.0.0.1"
   
user_name = os.getenv("FYERS_USERNAME")
cell = os.getenv("MOBILE_NUMBER")
pin = os.getenv("PIN")

grant_type = "authorization_code"                
response_type = "code"                          
state = "sample"                                 
opt_required = True
chrome_driver_path = "/usr/local/bin/chromedriver"

db_name = os.getenv("postgrey_db")
db_user = os.getenv("user_db")
db_password = os.getenv("password_db")

if not all([client_id, secret_key, redirect_uri]):
    raise ValueError("Missing environment variables. Check your .env file.")
                        ## app_secret key which you got after creating the app                                 ##  The state field here acts as a session manager. you will be sent with the state field after successfull generation of auth_code 
