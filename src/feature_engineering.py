# src/feature_engineering.py
import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import ast

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.df = None
        self.genre_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def load_processed_data(self, filepath='data/processed/processed_movies.csv'):
        """Load processed movie data"""
        self.df = pd.read_csv(filepath)
        
        # Convert string representations back to lists
        list_columns = ['genres_list', 'production_companies_list', 'top_cast']
        for col in list_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(
                    lambda x: ast.literal_eval(x) if pd.notna(x) and x.startswith('[') else []
                )
        
        logger.info(f"Loaded {len(self.df)} processed movies")
        return self.df
    
    def create_genre_features(self):
        """Create genre-based features"""
        if self.df is None:
            raise ValueError("No data loaded")
        
        # One-hot encode top genres
        all_genres = []
        for genres in self.df['genres_list']:
            all_genres.extend(genres)
        
        top_genres = pd.Series(all_genres).value_counts().head(10).index
        
        for genre in top_genres:
            self.df[f'genre_{genre.lower().replace(" ", "_")}'] = self.df['genres_list'].apply(
                lambda x: 1 if genre in x else 0
            )
        
        # Number of genres
        self.df['num_genres'] = self.df['genres_list'].apply(len)
        
        logger.info("✅ Created genre features")
    
    def create_temporal_features(self):
        """Create time-based features"""
        if 'release_month' in self.df.columns:
            # Season features
            self.df['release_season'] = self.df['release_month'].apply(
                lambda x: 'Winter' if x in [12, 1, 2] else
                         'Spring' if x in [3, 4, 5] else
                         'Summer' if x in [6, 7, 8] else 'Fall'
            )
            
            # Holiday season (Nov-Dec releases)
            self.df['holiday_release'] = self.df['release_month'].isin([11, 12]).astype(int)
            
            # Summer blockbuster season (May-Aug)
            self.df['summer_release'] = self.df['release_month'].isin([5, 6, 7, 8]).astype(int)
        
        logger.info("✅ Created temporal features")
    
    def create_text_features(self):
        """Create features from text data"""
        if 'overview' in self.df.columns:
            # Overview length
            self.df['overview_length'] = self.df['overview'].str.len().fillna(0)
            
            # Number of words in overview
            self.df['overview_word_count'] = self.df['overview'].str.split().str.len().fillna(0)
        
        logger.info("✅ Created text features")
    
    def create_success_metrics(self):
        """Define success metrics for prediction"""
        # Success based on rating (you can modify this)
        self.df['high_rating'] = (self.df['vote_average'] >= 7.0).astype(int)
        
        # Success based on profit (if we have enough financial data)
        if 'profit' in self.df.columns:
            self.df['profitable'] = (self.df['profit'] > 0).astype(int)
            self.df['high_roi'] = (self.df['roi'] > 100).astype(int)
        
        logger.info("✅ Created success metrics")
    
    def engineer_all_features(self, filepath='data/processed/processed_movies.csv'):
        """Run all feature engineering steps"""
        self.load_processed_data(filepath)
        self.create_genre_features()
        self.create_temporal_features()
        self.create_text_features()
        self.create_success_metrics()
        
        # Save engineered features
        output_path = 'data/processed/engineered_features.csv'
        self.df.to_csv(output_path, index=False)
        logger.info(f"💾 Saved engineered features to {output_path}")
        
        # Show feature summary
        print(f"\n🎯 FEATURE ENGINEERING SUMMARY")
        print("=" * 50)
        print(f"Total movies: {len(self.df)}")
        print(f"Total features: {len(self.df.columns)}")
        print(f"Success metrics created:")
        success_cols = [col for col in self.df.columns if any(term in col for term in ['high_', 'profitable'])]
        for col in success_cols:
            if col in self.df.columns:
                print(f"  - {col}: {self.df[col].sum()} positive cases")
        
        return self.df

def main():
    """Test feature engineering"""
    engineer = FeatureEngineer()
    
    try:
        engineered_df = engineer.engineer_all_features()
        print(f"\n✅ Successfully engineered features for {len(engineered_df)} movies")
        
        # Show new features
        new_feature_cols = [col for col in engineered_df.columns 
                           if any(keyword in col for keyword in ['genre_', 'num_', 'release_', 'overview_', 'high_'])]
        print(f"🎭 New features created: {len(new_feature_cols)}")
        for col in new_feature_cols[:10]:  # Show first 10
            print(f"   - {col}")
            
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")

if __name__ == "__main__":
    main()