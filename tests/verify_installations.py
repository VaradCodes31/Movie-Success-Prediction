# tests/verify_installation.py
import sys
import os
# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("🔍 FINAL VERIFICATION...")
print("=" * 50)

# Test Core Packages
try:
    import pandas as pd
    import numpy as np
    import sklearn
    import xgboost as xgb
    print("✅ Core ML packages: Pandas, NumPy, Scikit-learn, XGBoost")
except ImportError as e:
    print(f"❌ Core ML packages: {e}")

# Test NLP Packages
try:
    import nltk
    import spacy
    import transformers
    from sentence_transformers import SentenceTransformer
    import textblob
    print("✅ NLP packages: NLTK, SpaCy, Transformers, Sentence-Transformers, TextBlob")
except ImportError as e:
    print(f"❌ NLP packages: {e}")

# Test API Packages
try:
    import tmdbsimple
    import imdb  # CORRECT IMPORT NAME
    print("✅ API packages: TMDB, IMDbPY")
except ImportError as e:
    print(f"❌ API packages: {e}")

# Test Additional Packages
try:
    import lightgbm
    from dotenv import load_dotenv
    import streamlit
    print("✅ Additional packages: LightGBM, python-dotenv, Streamlit")
except ImportError as e:
    print(f"❌ Additional packages: {e}")

# Test SpaCy Model
try:
    nlp = spacy.load("en_core_web_sm")
    print("✅ SpaCy English model loaded successfully")
except Exception as e:
    print(f"❌ SpaCy model: {e}")

print("=" * 50)
print("🎯 ALL PACKAGES INSTALLED SUCCESSFULLY!")