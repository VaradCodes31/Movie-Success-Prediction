# src/data_processing.py
import sys
import os
import pandas as pd
import numpy as np
import ast
import json

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.df = None
    
    def load_data(self, filepath):
        """Load raw movie data"""
        self.df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(self.df)} movies from {filepath}")
        return self.df
    
    def parse_json_columns(self):
        """Parse JSON-like columns into usable formats"""
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Parse genres
        if 'genres' in self.df.columns:
            self.df['genres_list'] = self.df['genres'].apply(
                lambda x: [g['name'] for g in ast.literal_eval(x)] if pd.notna(x) and x != '[]' else []
            )
        
        # Parse production companies
        if 'production_companies' in self.df.columns:
            self.df['production_companies_list'] = self.df['production_companies'].apply(
                lambda x: [c['name'] for c in ast.literal_eval(x)] if pd.notna(x) and x != '[]' else []
            )
        
        # Parse credits (cast and crew)
        if 'credits' in self.df.columns:
            def extract_top_cast(credits_str, n=5):
                try:
                    credits = ast.literal_eval(credits_str)
                    cast = credits.get('cast', [])
                    return [actor['name'] for actor in cast[:n]]
                except:
                    return []
            
            self.df['top_cast'] = self.df['credits'].apply(extract_top_cast)
        
        logger.info("✅ Parsed JSON columns")
    
    def calculate_profit(self):
        """Calculate profit and ROI"""
        if 'revenue' in self.df.columns and 'budget' in self.df.columns:
            self.df['profit'] = self.df['revenue'] - self.df['budget']
            # Avoid division by zero
            self.df['roi'] = np.where(
                self.df['budget'] > 0,
                (self.df['profit'] / self.df['budget']) * 100,
                0
            )
            logger.info("✅ Calculated profit and ROI")
    
    def extract_release_features(self):
        """Extract features from release date"""
        if 'release_date' in self.df.columns:
            self.df['release_date'] = pd.to_datetime(self.df['release_date'])
            self.df['release_year'] = self.df['release_date'].dt.year
            self.df['release_month'] = self.df['release_date'].dt.month
            self.df['release_quarter'] = self.df['release_date'].dt.quarter
            logger.info("✅ Extracted release date features")
    
    def process_all(self, filepath):
        """Run all processing steps"""
        self.load_data(filepath)
        self.parse_json_columns()
        self.calculate_profit()
        self.extract_release_features()
        
        # Save processed data
        output_path = 'data/processed/processed_movies.csv'
        self.df.to_csv(output_path, index=False)
        logger.info(f"💾 Saved processed data to {output_path}")
        
        return self.df

def main():
    """Test the data processor"""
    processor = DataProcessor()
    
    try:
        processed_df = processor.process_all('data/raw/sample_movies.csv')
        print(f"✅ Processed {len(processed_df)} movies")
        print(f"📊 New columns: {list(processed_df.columns)}")
        
        # Show some processed data
        print("\n🎭 Sample processed genres:")
        for i, row in processed_df.head(3).iterrows():
            print(f"   {row['title']}: {row.get('genres_list', [])}")
            
    except Exception as e:
        print(f"❌ Processing failed: {e}")

if __name__ == "__main__":
    main()