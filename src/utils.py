# src/utils.py
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

def get_tmdb_api_key():
    """Get TMDB API key from environment variables"""
    api_key = os.getenv('TMDB_API_KEY')
    if not api_key:
        raise ValueError("TMDB_API_KEY not found in .env file")
    return api_key

def setup_logging():
    """Setup basic logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )