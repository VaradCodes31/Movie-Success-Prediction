# tests/test_api.py
import sys
import os
# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import get_tmdb_api_key
import requests

def test_tmdb_connection():
    try:
        # Get API key
        api_key = get_tmdb_api_key()
        print("✅ API Key loaded successfully")
        
        # Test with a popular movie (The Dark Knight)
        url = f"https://api.themoviedb.org/3/movie/155?api_key={api_key}"
        response = requests.get(url)
        
        if response.status_code == 200:
            movie_data = response.json()
            print("✅ TMDB API connection successful!")
            print(f"🎬 Movie: {movie_data.get('title')}")
            print(f"📅 Release: {movie_data.get('release_date')}")
            print(f"⭐ Rating: {movie_data.get('vote_average')}")
            print(f"📊 Votes: {movie_data.get('vote_count')}")
        else:
            print(f"❌ API connection failed. Status: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_tmdb_connection()