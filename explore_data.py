# explore_data.py
import pandas as pd
import json
import ast
import matplotlib.pyplot as plt
import seaborn as sns

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")

print("🎬 MOVIE DATA EXPLORATION")
print("=" * 60)

# Load the data
df = pd.read_csv('data/raw/sample_movies.csv')
print(f"✅ Loaded {len(df)} movies")
print(f"📊 Dataset shape: {df.shape}")

print("\n📋 COLUMNS AVAILABLE:")
print("-" * 40)
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

print("\n🔍 BASIC INFO:")
print("-" * 40)
print(df.info())

print("\n📈 NUMERICAL STATISTICS:")
print("-" * 40)
numeric_cols = ['budget', 'revenue', 'runtime', 'vote_average', 'vote_count', 'popularity']
for col in numeric_cols:
    if col in df.columns:
        print(f"{col:15}: Min={df[col].min():>10,} | Max={df[col].max():>15,} | Mean={df[col].mean():>12,.2f}")

print("\n🎭 SAMPLE MOVIES:")
print("-" * 40)
for i in range(min(3, len(df))):
    movie = df.iloc[i]
    print(f"\n{i+1}. {movie['title']} ({movie.get('release_date', 'N/A')})")
    print(f"   📝 {movie.get('overview', 'No description')[:100]}...")
    print(f"   ⭐ Rating: {movie.get('vote_average', 'N/A')} (from {movie.get('vote_count', 0):,} votes)")
    print(f"   💰 Budget: ${movie.get('budget', 0):,} | Revenue: ${movie.get('revenue', 0):,}")
    print(f"   🎭 Genres: {movie.get('genres', 'N/A')}")

# Try to parse genres if they're in string format
print("\n🎪 GENRE ANALYSIS:")
print("-" * 40)
if 'genres' in df.columns:
    try:
        # Try to parse genres from string to list
        all_genres = []
        for genre_str in df['genres'].dropna():
            if isinstance(genre_str, str):
                try:
                    genres_list = ast.literal_eval(genre_str)
                    for genre in genres_list:
                        if isinstance(genre, dict) and 'name' in genre:
                            all_genres.append(genre['name'])
                except:
                    continue
        
        if all_genres:
            genre_counts = pd.Series(all_genres).value_counts()
            print("Genre distribution:")
            for genre, count in genre_counts.items():
                print(f"   {genre:20}: {count} movies")
    except Exception as e:
        print(f"Could not parse genres: {e}")

print("\n📅 RELEASE DATES:")
print("-" * 40)
if 'release_date' in df.columns:
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    print(f"Earliest: {df['release_date'].min()}")
    print(f"Latest: {df['release_date'].max()}")
    print(f"Time span: {(df['release_date'].max() - df['release_date'].min()).days} days")

print("\n✅ EXPLORATION COMPLETE!")