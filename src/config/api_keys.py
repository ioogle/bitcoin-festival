"""API key configuration."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Glassnode API key
GLASSNODE_API_KEY = os.getenv('GLASSNODE_API_KEY')

# Function to check if all required API keys are present
def check_api_keys():
    """Check if all required API keys are present in environment variables."""
    missing_keys = []
    
    if not GLASSNODE_API_KEY:
        missing_keys.append('GLASSNODE_API_KEY')
    
    if missing_keys:
        raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")
    
    return True 