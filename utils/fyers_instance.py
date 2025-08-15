import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.fyers_api import get_access_token
from config import client_id 
from fyers_apiv3 import fyersModel


class FyersInstance:
    """
    Singleton class for Fyers API connection.
    Ensures only one instance of the Fyers API client exists throughout the application.
    """
    _instance = None
    _access_token = None

    @staticmethod
    def get_instance():
        """
        Returns the singleton instance of the Fyers API client.
        Creates a new instance if one doesn't exist yet.
        """
        if FyersInstance._instance is None:
            # Get access token using the function from fyers_api.py
            access_token = FyersInstance.get_access_token()
            
            # Create the Fyers instance
            FyersInstance._instance = fyersModel.FyersModel(
                token=access_token, 
                is_async=False, 
                client_id=client_id
            )
        
        return FyersInstance._instance
    
    @staticmethod
    def get_access_token():
        """
        Get the current access token.
        This will initialize the access token if it doesn't exist using the logic in fyers_api.py.
        """
        if FyersInstance._access_token is None:
            # Use the centralized function from fyers_api.py
            FyersInstance._access_token = get_access_token()
                
        return FyersInstance._access_token
    
    @staticmethod
    def reset_instance():
        """
        Reset the singleton instance.
        Useful for testing or for reconnecting after errors.
        """
        FyersInstance._instance = None
        FyersInstance._access_token = None

if __name__ == "__main__":
    # Test the singleton pattern
    instance1 = FyersInstance.get_instance()
    instance2 = FyersInstance.get_instance()
    
    # Verify both instances are the same object
    print(f"Instances are the same object: {instance1 is instance2}")
    
    # Test access token
    token = FyersInstance.get_access_token()
    print(f"Access token: {token[:10]}...")  # Show only first 10 characters for security
    
    # Example API call
    profile = instance1.get_profile()
    print(f"Profile data: {profile}")