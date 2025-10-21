# run_full_pipeline.py
import sys
import os
import pandas as pd

# Add src to path
sys.path.append('src')

from src.data_processing import DataProcessor
from src.feature_engineering import FeatureEngineer

def run_full_pipeline():
    """Run the complete data processing and feature engineering pipeline"""
    print("🚀 RUNNING FULL DATA PIPELINE")
    print("=" * 60)
    
    # Step 1: Process the raw data
    print("\n📊 STEP 1: PROCESSING RAW DATA")
    print("-" * 40)
    processor = DataProcessor()
    processed_df = processor.process_all('data/raw/large_movie_dataset.csv')
    print(f"✅ Processed {len(processed_df)} movies")
    print(f"📈 New columns: {len(processed_df.columns)}")
    
    # Step 2: Engineer features
    print("\n🎯 STEP 2: ENGINEERING FEATURES")
    print("-" * 40)
    engineer = FeatureEngineer()
    engineered_df = engineer.engineer_all_features('data/processed/processed_movies.csv')
    print(f"✅ Engineered {len(engineered_df.columns)} total features")
    
    # Show dataset summary
    print("\n📋 FINAL DATASET SUMMARY")
    print("=" * 60)
    print(f"Total movies: {len(engineered_df)}")
    print(f"Total features: {len(engineered_df.columns)}")
    
    # Show success metrics distribution
    success_cols = ['high_rating', 'profitable', 'high_roi']
    print(f"\n🎯 SUCCESS METRICS DISTRIBUTION:")
    for col in success_cols:
        if col in engineered_df.columns:
            positive = engineered_df[col].sum()
            percentage = (positive / len(engineered_df)) * 100
            print(f"   {col:12}: {positive:3d} movies ({percentage:5.1f}%)")
    
    # Show genre distribution
    print(f"\n🎭 TOP GENRES:")
    genre_cols = [col for col in engineered_df.columns if col.startswith('genre_')]
    genre_counts = engineered_df[genre_cols].sum().sort_values(ascending=False)
    for genre, count in genre_counts.head(8).items():
        percentage = (count / len(engineered_df)) * 100
        print(f"   {genre.replace('genre_', ''):15}: {count:3d} movies ({percentage:5.1f}%)")
    
    # Show financial stats
    print(f"\n💰 FINANCIAL OVERVIEW:")
    if 'budget' in engineered_df.columns:
        total_budget = engineered_df['budget'].sum()
        total_revenue = engineered_df['revenue'].sum()
        total_profit = engineered_df['profit'].sum()
        print(f"   Total Budget:  ${total_budget:>15,}")
        print(f"   Total Revenue: ${total_revenue:>15,}")
        print(f"   Total Profit:  ${total_profit:>15,}")
    
    print(f"\n🎉 PIPELINE COMPLETE! Ready for machine learning.")
    return engineered_df

if __name__ == "__main__":
    final_df = run_full_pipeline()
    
    # Save the final dataset
    final_df.to_csv('data/processed/final_movie_dataset.csv', index=False)
    print(f"💾 Final dataset saved to: data/processed/final_movie_dataset.csv")