# src/model_training.py
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)

class MovieSuccessPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def load_data(self, filepath='data/processed/final_movie_dataset.csv'):
        """Load the final engineered dataset"""
        self.df = pd.read_csv(filepath)
        logger.info(f"Loaded dataset with {len(self.df)} movies and {len(self.df.columns)} features")
        return self.df
    
    def prepare_features(self, target='high_rating'):
        """Prepare features for modeling"""
        # Select feature columns (numeric and encoded features)
        exclude_cols = ['title', 'overview', 'release_date', 'genres_list', 
                       'production_companies_list', 'top_cast', 'credits', 'keywords',
                       'adult', 'backdrop_path', 'homepage', 'imdb_id', 'poster_path',
                       'tagline', 'video', 'belongs_to_collection']
        
        # Get numeric and boolean columns
        feature_cols = []
        for col in self.df.columns:
            if col not in exclude_cols and col != target:
                if self.df[col].dtype in ['int64', 'float64', 'bool']:
                    feature_cols.append(col)
        
        self.feature_columns = feature_cols
        logger.info(f"Selected {len(feature_cols)} features for modeling")
        
        # Prepare X and y
        X = self.df[feature_cols].fillna(0)
        y = self.df[target]
        
        logger.info(f"Target '{target}': {y.sum()} positive cases out of {len(y)} ({y.mean()*100:.1f}%)")
        
        return X, y
    
    def train_models(self, target='high_rating', test_size=0.2):
        """Train multiple models and compare performance"""
        X, y = self.prepare_features(target)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models to train
        models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
            'LightGBM': LGBMClassifier(random_state=42)
        }
        
        results = {}
        
        print(f"\n🎯 TRAINING MODELS FOR: {target.upper()}")
        print("=" * 60)
        
        for name, model in models.items():
            print(f"\n📊 Training {name}...")
            
            # Train model
            if name == 'Logistic Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation
            if name == 'Logistic Regression':
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Store results
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'feature_importance': None
            }
            
            # Get feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                results[name]['feature_importance'] = importance_df
            
            print(f"   ✅ Accuracy: {accuracy:.3f}")
            print(f"   📊 CV Score: {cv_mean:.3f} ± {cv_std:.3f}")
        
        self.models = results
        self.X_test = X_test
        self.y_test = y_test
        
        return results
    
    def show_model_comparison(self):
        """Compare all trained models"""
        if not self.models:
            print("No models trained yet. Call train_models() first.")
            return
        
        print(f"\n🏆 MODEL COMPARISON")
        print("=" * 60)
        
        comparison_df = pd.DataFrame({
            'Model': list(self.models.keys()),
            'Accuracy': [result['accuracy'] for result in self.models.values()],
            'CV Mean': [result['cv_mean'] for result in self.models.values()],
            'CV Std': [result['cv_std'] for result in self.models.values()]
        }).sort_values('Accuracy', ascending=False)
        
        print(comparison_df.to_string(index=False))
        
        # Show feature importance for best model
        best_model_name = comparison_df.iloc[0]['Model']
        best_model_result = self.models[best_model_name]
        
        if best_model_result['feature_importance'] is not None:
            print(f"\n🔍 TOP 10 FEATURES - {best_model_name}:")
            print("-" * 40)
            top_features = best_model_result['feature_importance'].head(10)
            for _, row in top_features.iterrows():
                print(f"   {row['feature']:25}: {row['importance']:.4f}")
    
    def save_models(self, directory='models'):
        """Save trained models"""
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        for name, result in self.models.items():
            model = result['model']
            filename = f"{directory}/{name.lower().replace(' ', '_')}.pkl"
            joblib.dump(model, filename)
            logger.info(f"Saved {name} to {filename}")
        
        # Save scaler and feature columns
        joblib.dump(self.scaler, f"{directory}/scaler.pkl")
        joblib.dump(self.feature_columns, f"{directory}/feature_columns.pkl")
        
        print(f"\n💾 All models saved to '{directory}/' directory")

def main():
    """Train and evaluate models"""
    predictor = MovieSuccessPredictor()
    
    # Load data
    predictor.load_data()
    
    # Train models for different success metrics
    targets = ['high_rating', 'profitable', 'high_roi']
    
    for target in targets:
        try:
            predictor.train_models(target=target)
            predictor.show_model_comparison()
            print("\n" + "="*60 + "\n")
        except Exception as e:
            print(f"❌ Could not train model for {target}: {e}")
    
    # Save all models
    predictor.save_models()

if __name__ == "__main__":
    main()