# src/data_collection.py
import sys
import os
import requests
import pandas as pd
import time
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import get_tmdb_api_key, setup_logging
import logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

class TMDBDataCollector:
    def __init__(self):
        self.api_key = get_tmdb_api_key()
        self.base_url = "https://api.themoviedb.org/3"
        self.session = requests.Session()
        
    def get_popular_movies(self, page=1):
        """Get popular movies from TMDB"""
        url = f"{self.base_url}/movie/popular"
        params = {
            'api_key': self.api_key,
            'page': page,
            'language': 'en-US'
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching popular movies page {page}: {e}")
            return None
    
    def get_movie_details(self, movie_id):
        """Get detailed information for a specific movie"""
        url = f"{self.base_url}/movie/{movie_id}"
        params = {
            'api_key': self.api_key,
            'language': 'en-US',
            'append_to_response': 'credits,keywords'  # Get cast, crew, and keywords
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching details for movie {movie_id}: {e}")
            return None
    
    def collect_movie_data(self, num_movies=500):
        """Collect data for multiple movies"""
        logger.info(f"Starting collection of {num_movies} movies...")
        
        all_movies = []
        page = 1
        
        with tqdm(total=num_movies, desc="Collecting movies") as pbar:
            while len(all_movies) < num_movies:
                # Get popular movies for current page
                popular_data = self.get_popular_movies(page)
                
                if not popular_data or 'results' not in popular_data:
                    break
                
                # Process each movie on this page
                for movie in popular_data['results']:
                    if len(all_movies) >= num_movies:
                        break
                    
                    movie_id = movie['id']
                    logger.info(f"Fetching details for: {movie['title']} (ID: {movie_id})")
                    
                    # Get detailed movie information
                    details = self.get_movie_details(movie_id)
                    if details:
                        all_movies.append(details)
                        pbar.update(1)
                    
                    # Be nice to the API - add delay
                    time.sleep(0.1)  # 100ms delay between requests
                
                page += 1
                
                # Safety break
                if page > 50:  # Don't go beyond 50 pages
                    break
        
        logger.info(f"Successfully collected data for {len(all_movies)} movies")
        return all_movies

def collect_large_dataset():
    """Collect a larger dataset for modeling"""
    collector = TMDBDataCollector()
    
    print("🎬 COLLECTING LARGER DATASET FOR MODELING")
    print("=" * 50)
    
    # Collect 200 movies (good for initial modeling)
    print("Collecting 200 movies... (this will take 3-5 minutes)")
    movies = collector.collect_movie_data(num_movies=200)
    
    if movies:
        # Save raw data
        df = pd.DataFrame(movies)
        raw_path = 'data/raw/large_movie_dataset.csv'
        df.to_csv(raw_path, index=False)
        print(f"✅ Saved {len(movies)} movies to {raw_path}")
        
        return df
    else:
        print("❌ Failed to collect movie data")
        return None

if __name__ == "__main__":
    # Test with small dataset
    collector = TMDBDataCollector()
    print("🧪 Testing with 5 movies...")
    test_movies = collector.collect_movie_data(num_movies=5)
    
    if test_movies:
        print(f"✅ Test successful - collected {len(test_movies)} movies")
        
        # Ask if user wants to collect larger dataset
        response = input("\n🎯 Collect larger dataset (200 movies) for modeling? (y/n): ")
        if response.lower() == 'y':
            collect_large_dataset()
    else:
        print("❌ Test failed")