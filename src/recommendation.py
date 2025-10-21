# src/recommendation.py
import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import ast
import joblib

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)

class MovieRecommender:
    def __init__(self):
        self.df = None
        self.similarity_matrix = None
        self.feature_columns = []
        self.sentence_model = None
        
    def load_data(self, filepath='data/processed/final_movie_dataset.csv'):
        """Load the movie dataset"""
        self.df = pd.read_csv(filepath)
        
        # Convert string representations back to lists
        list_columns = ['genres_list', 'production_companies_list', 'top_cast']
        for col in list_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(
                    lambda x: ast.literal_eval(x) if pd.notna(x) and x.startswith('[') else []
                )
        
        logger.info(f"Loaded {len(self.df)} movies for recommendation system")
        return self.df
    
    def create_content_features(self):
        """Create features for content-based filtering"""
        # Combine genres, overview, and cast into a text feature
        self.df['content_features'] = self.df.apply(self._create_movie_profile, axis=1)
        
        # Use Sentence Transformers for better text understanding
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create embeddings for each movie
        content_embeddings = self.sentence_model.encode(self.df['content_features'].tolist())
        
        # Calculate similarity matrix
        self.similarity_matrix = cosine_similarity(content_embeddings)
        
        logger.info("✅ Created content-based similarity matrix")
    
    def _create_movie_profile(self, movie):
        """Create a text profile for a movie"""
        profile_parts = []
        
        # Add genres
        if 'genres_list' in movie and movie['genres_list']:
            profile_parts.extend(movie['genres_list'])
        
        # Add top cast members
        if 'top_cast' in movie and movie['top_cast']:
            profile_parts.extend(movie['top_cast'][:3])  # Top 3 cast members
        
        # Add overview
        if pd.notna(movie.get('overview')):
            profile_parts.append(movie['overview'])
        
        # Add production companies
        if 'production_companies_list' in movie and movie['production_companies_list']:
            profile_parts.extend(movie['production_companies_list'][:2])  # Top 2 companies
        
        return ' '.join(str(part) for part in profile_parts)
    
    def get_similar_movies(self, movie_title, n_recommendations=5):
        """Get similar movies based on content"""
        if self.similarity_matrix is None:
            self.create_content_features()
        
        # Find the movie index
        movie_idx = self.df[self.df['title'].str.lower() == movie_title.lower()].index
        if len(movie_idx) == 0:
            return f"Movie '{movie_title}' not found in database"
        
        movie_idx = movie_idx[0]
        
        # Get similarity scores for this movie
        similarity_scores = list(enumerate(self.similarity_matrix[movie_idx]))
        
        # Sort by similarity score (descending)
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N similar movies (excluding the movie itself)
        similar_movies = []
        for i, score in similarity_scores[1:n_recommendations+1]:
            movie = self.df.iloc[i]
            similar_movies.append({
                'title': movie['title'],
                'similarity_score': round(score, 3),
                'genres': movie.get('genres_list', []),
                'rating': movie.get('vote_average', 'N/A'),
                'year': movie.get('release_year', 'N/A'),
                'overview': movie.get('overview', '')[:150] + '...' if pd.notna(movie.get('overview')) else 'No description'
            })
        
        return similar_movies
    
    def get_recommendations_by_features(self, preferred_genres=[], min_rating=6.0, n_recommendations=5):
        """Get recommendations based on preferred features"""
        filtered_movies = self.df.copy()
        
        # Filter by genres if provided
        if preferred_genres:
            genre_filter = filtered_movies['genres_list'].apply(
                lambda genres: any(genre in genres for genre in preferred_genres)
            )
            filtered_movies = filtered_movies[genre_filter]
        
        # Filter by minimum rating
        if min_rating > 0:
            filtered_movies = filtered_movies[filtered_movies['vote_average'] >= min_rating]
        
        # Sort by popularity and rating
        recommendations = filtered_movies.sort_values(
            ['popularity', 'vote_average'], ascending=[False, False]
        ).head(n_recommendations)
        
        result = []
        for _, movie in recommendations.iterrows():
            result.append({
                'title': movie['title'],
                'genres': movie.get('genres_list', []),
                'rating': movie.get('vote_average', 'N/A'),
                'popularity': round(movie.get('popularity', 0), 1),
                'year': movie.get('release_year', 'N/A'),
                'overview': movie.get('overview', '')[:150] + '...' if pd.notna(movie.get('overview')) else 'No description'
            })
        
        return result
    
    def hybrid_recommendation(self, movie_title=None, preferred_genres=[], min_rating=6.0, n_recommendations=5):
        """Hybrid recommendation combining content-based and feature-based"""
        recommendations = []
        
        # Content-based recommendations if movie title provided
        if movie_title:
            content_recs = self.get_similar_movies(movie_title, n_recommendations)
            if isinstance(content_recs, list):
                recommendations.extend(content_recs)
        
        # Feature-based recommendations
        feature_recs = self.get_recommendations_by_features(preferred_genres, min_rating, n_recommendations)
        recommendations.extend(feature_recs)
        
        # Remove duplicates and return top N
        seen_titles = set()
        unique_recommendations = []
        
        for rec in recommendations:
            if rec['title'] not in seen_titles:
                seen_titles.add(rec['title'])
                unique_recommendations.append(rec)
        
        return unique_recommendations[:n_recommendations]

def main():
    """Test the recommendation system"""
    recommender = MovieRecommender()
    recommender.load_data()
    
    print("🎬 MOVIE RECOMMENDATION SYSTEM")
    print("=" * 60)
    
    # Test content-based recommendations
    test_movies = ['The Dark Knight', 'Inception', 'Avatar']  # Try these if in dataset
    
    for movie in test_movies:
        print(f"\n🔍 If you liked '{movie}', you might like:")
        print("-" * 40)
        
        recommendations = recommender.get_similar_movies(movie, 3)
        
        if isinstance(recommendations, str):
            print(f"   {recommendations}")
        else:
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec['title']} (Similarity: {rec['similarity_score']})")
                print(f"      Genres: {', '.join(rec['genres'])}")
                print(f"      Rating: {rec['rating']} | Year: {rec['year']}")
                print(f"      {rec['overview']}")
    
    # Test feature-based recommendations
    print(f"\n🎯 RECOMMENDATIONS FOR ACTION & ADVENTURE FANS:")
    print("-" * 40)
    
    feature_recs = recommender.get_recommendations_by_features(
        preferred_genres=['Action', 'Adventure'],
        min_rating=7.0,
        n_recommendations=3
    )
    
    for i, rec in enumerate(feature_recs, 1):
        print(f"   {i}. {rec['title']} (Rating: {rec['rating']}, Popularity: {rec['popularity']})")
        print(f"      Genres: {', '.join(rec['genres'])}")
        print(f"      {rec['overview']}")
    
    print(f"\n🎉 RECOMMENDATION SYSTEM READY!")

if __name__ == "__main__":
    main()