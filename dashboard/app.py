# dashboard/app.py
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import joblib
import ast

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recommendation import MovieRecommender
from src.model_training import MovieSuccessPredictor

# Page configuration
st.set_page_config(
    page_title="Movie Success Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-prediction {
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    .risk-prediction {
        background-color: #FF9800;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class MovieDashboard:
    def __init__(self):
        self.recommender = None
        self.predictor = None
        self.data = None
        
    def load_data(self):
        """Load data and models"""
        try:
            # Load data
            self.data = pd.read_csv('data/processed/final_movie_dataset.csv')
            
            # Initialize recommender
            self.recommender = MovieRecommender()
            self.recommender.load_data()
            self.recommender.create_content_features()
            
            # Initialize predictor (we'll load models on demand)
            self.predictor = MovieSuccessPredictor()
            self.predictor.load_data()
            
            return True
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False
    
    def show_overview(self):
        """Show dashboard overview"""
        st.markdown('<div class="main-header">🎬 Movie Success Predictor</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Movies", len(self.data))
        
        with col2:
            avg_rating = self.data['vote_average'].mean()
            st.metric("Average Rating", f"{avg_rating:.1f}/10")
        
        with col3:
            profitable = self.data['profitable'].sum()
            st.metric("Profitable Movies", profitable)
        
        with col4:
            high_rated = self.data['high_rating'].sum()
            st.metric("Highly Rated", high_rated)
        
        st.markdown("---")
    
    def recommendation_section(self):
        """Movie recommendation section"""
        st.header("🎯 Movie Recommendations")
        
        tab1, tab2, tab3 = st.tabs(["Similar Movies", "Genre-Based", "Hybrid"])
        
        with tab1:
            st.subheader("If you liked X, you might like Y")
            movie_titles = sorted(self.data['title'].tolist())
            selected_movie = st.selectbox("Select a movie you like:", movie_titles)
            
            if st.button("Find Similar Movies"):
                with st.spinner("Finding similar movies..."):
                    recommendations = self.recommender.get_similar_movies(selected_movie, 5)
                    
                    if isinstance(recommendations, list):
                        for i, rec in enumerate(recommendations, 1):
                            with st.container():
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.write(f"**{i}. {rec['title']}**")
                                    st.write(f"Genres: {', '.join(rec['genres'])}")
                                    st.write(f"Similarity: {rec['similarity_score']}")
                                    st.write(rec['overview'])
                                with col2:
                                    st.write(f"Rating: ⭐ {rec['rating']}")
                                    st.write(f"Year: {rec['year']}")
                            st.markdown("---")
        
        with tab2:
            st.subheader("Discover by Genre")
            all_genres = set()
            for genres in self.data['genres_list']:
                if isinstance(genres, str):
                    try:
                        genre_list = ast.literal_eval(genres)
                        all_genres.update(genre_list)
                    except:
                        continue
            
            selected_genres = st.multiselect("Select preferred genres:", sorted(all_genres))
            min_rating = st.slider("Minimum rating:", 0.0, 10.0, 7.0, 0.1)
            
            if st.button("Get Recommendations"):
                recommendations = self.recommender.get_recommendations_by_features(
                    selected_genres, min_rating, 5
                )
                
                for i, rec in enumerate(recommendations, 1):
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{i}. {rec['title']}**")
                            st.write(f"Genres: {', '.join(rec['genres'])}")
                            st.write(rec['overview'])
                        with col2:
                            st.write(f"Rating: ⭐ {rec['rating']}")
                            st.write(f"Popularity: {rec['popularity']}")
                            st.write(f"Year: {rec['year']}")
                    st.markdown("---")
        
        with tab3:
            st.subheader("Hybrid Recommendations")
            col1, col2 = st.columns(2)
            
            with col1:
                base_movie = st.selectbox("Base movie (optional):", [""] + sorted(self.data['title'].tolist()))
            
            with col2:
                preferred_genres = st.multiselect("Preferred genres (optional):", sorted(all_genres))
            
            if st.button("Get Hybrid Recommendations"):
                recommendations = self.recommender.hybrid_recommendation(
                    movie_title=base_movie if base_movie else None,
                    preferred_genres=preferred_genres,
                    n_recommendations=5
                )
                
                for i, rec in enumerate(recommendations, 1):
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{i}. {rec['title']}**")
                            st.write(f"Genres: {', '.join(rec['genres'])}")
                            if 'similarity_score' in rec:
                                st.write(f"Similarity: {rec['similarity_score']}")
                            st.write(rec['overview'])
                        with col2:
                            st.write(f"Rating: ⭐ {rec['rating']}")
                            st.write(f"Year: {rec['year']}")
                    st.markdown("---")
    
    def prediction_section(self):
        """Movie success prediction section for producers"""
        st.header("📊 Success Prediction for Producers")
        
        st.info("This section helps producers predict the potential success of movie concepts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Movie Concept")
            concept_title = st.text_input("Movie Title", "New Movie Concept")
            concept_overview = st.text_area("Plot Overview", "A compelling story about...")
            
            # Genre selection
            all_genres = set()
            for genres in self.data['genres_list']:
                if isinstance(genres, str):
                    try:
                        genre_list = ast.literal_eval(genres)
                        all_genres.update(genre_list)
                    except:
                        continue
            
            selected_concept_genres = st.multiselect("Genres", sorted(all_genres))
        
        with col2:
            st.subheader("Production Details")
            budget = st.number_input("Budget ($)", min_value=0, max_value=500000000, value=50000000, step=1000000)
            runtime = st.slider("Runtime (minutes)", 60, 240, 120)
            release_month = st.selectbox("Release Month", 
                                       ["January", "February", "March", "April", "May", "June",
                                        "July", "August", "September", "October", "November", "December"])
            
            # Convert month to number
            month_map = {month: i+1 for i, month in enumerate([
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ])}
            release_month_num = month_map[release_month]
        
        if st.button("Predict Success"):
            st.subheader("🎯 Prediction Results")
            
            # This is a simplified prediction - in a real scenario, we'd use the trained models
            # For now, we'll show some insights based on the data
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Simple heuristic based on budget and genre
                genre_risk = len(selected_concept_genres)  # More genres = more focused
                budget_risk = "Low" if budget > 20000000 else "Medium" if budget > 5000000 else "High"
                
                st.metric("Budget Risk Level", budget_risk)
            
            with col2:
                # Release timing insight
                summer_months = [5, 6, 7, 8]
                holiday_months = [11, 12]
                
                if release_month_num in summer_months:
                    timing_advice = "Summer Blockbuster Season"
                elif release_month_num in holiday_months:
                    timing_advice = "Holiday Season"
                else:
                    timing_advice = "Standard Release"
                
                st.metric("Release Timing", timing_advice)
            
            with col3:
                # Genre popularity insight
                if selected_concept_genres:
                    genre_popularity = "High" if any(genre in ['Action', 'Adventure', 'Drama'] for genre in selected_concept_genres) else "Medium"
                    st.metric("Genre Popularity", genre_popularity)
                else:
                    st.metric("Genre Popularity", "Select genres")
            
            # Success probability (simplified)
            success_prob = 0.65  # Placeholder - would come from ML model
            
            if success_prob > 0.7:
                st.markdown(f'<div class="success-prediction">High Success Probability: {success_prob:.0%}</div>', unsafe_allow_html=True)
                st.success("🎉 This concept shows strong potential for success!")
            else:
                st.markdown(f'<div class="risk-prediction">Moderate Success Probability: {success_prob:.0%}</div>', unsafe_allow_html=True)
                st.warning("💡 Consider optimizing budget, genre mix, or release timing.")
            
            # Recommendations
            st.subheader("📈 Recommended Strategies")
            
            rec_col1, rec_col2, rec_col3 = st.columns(3)
            
            with rec_col1:
                st.write("**Genre Strategy**")
                if len(selected_concept_genres) > 3:
                    st.write("🔻 Consider focusing on 2-3 core genres")
                else:
                    st.write("✅ Good genre focus")
            
            with rec_col2:
                st.write("**Budget Strategy**")
                if budget > 100000000:
                    st.write("🔻 High budget increases risk")
                elif budget < 10000000:
                    st.write("⚠️ Low budget may limit production quality")
                else:
                    st.write("✅ Optimal budget range")
            
            with rec_col3:
                st.write("**Release Strategy**")
                if release_month_num in [1, 2, 9]:
                    st.write("✅ Good timing - less competition")
                else:
                    st.write("⚠️ Competitive release window")
    
    def data_insights_section(self):
        """Show data insights and analytics"""
        st.header("📈 Industry Insights")
        
        tab1, tab2, tab3 = st.tabs(["Genre Analysis", "Financial Trends", "Release Strategy"])
        
        with tab1:
            st.subheader("Genre Performance")
            
            # Calculate genre success rates
            genre_success = {}
            for _, movie in self.data.iterrows():
                if isinstance(movie['genres_list'], str):
                    try:
                        genres = ast.literal_eval(movie['genres_list'])
                        for genre in genres:
                            if genre not in genre_success:
                                genre_success[genre] = {'count': 0, 'high_rated': 0, 'profitable': 0}
                            
                            genre_success[genre]['count'] += 1
                            if movie.get('high_rating', 0):
                                genre_success[genre]['high_rated'] += 1
                            if movie.get('profitable', 0):
                                genre_success[genre]['profitable'] += 1
                    except:
                        continue
            
            # Display top genres
            genre_df = pd.DataFrame([
                {
                    'Genre': genre,
                    'Movies': stats['count'],
                    'High Rating %': (stats['high_rated'] / stats['count']) * 100,
                    'Profitable %': (stats['profitable'] / stats['count']) * 100
                }
                for genre, stats in genre_success.items()
            ]).sort_values('Movies', ascending=False)
            
            st.dataframe(genre_df.head(10), use_container_width=True)
        
        with tab2:
            st.subheader("Budget vs Revenue Analysis")
            
            # Scatter plot data
            plot_data = self.data[['budget', 'revenue', 'vote_average', 'title']].copy()
            plot_data = plot_data[(plot_data['budget'] > 0) & (plot_data['revenue'] > 0)]
            
            if not plot_data.empty:
                st.scatter_chart(
                    plot_data,
                    x='budget',
                    y='revenue',
                    size='vote_average',
                    color='vote_average'
                )
            
            # ROI analysis
            if 'roi' in self.data.columns:
                high_roi_movies = self.data[self.data['roi'] > 100]
                st.write(f"**High ROI Movies (>100%):** {len(high_roi_movies)}")
        
        with tab3:
            st.subheader("Release Timing Analysis")
            
            if 'release_month' in self.data.columns:
                monthly_performance = self.data.groupby('release_month').agg({
                    'vote_average': 'mean',
                    'profitable': 'mean',
                    'popularity': 'mean'
                }).reset_index()
                
                st.line_chart(monthly_performance, x='release_month', y=['vote_average', 'profitable'])

def main():
    """Main dashboard function"""
    dashboard = MovieDashboard()
    
    if dashboard.load_data():
        # Sidebar navigation
        st.sidebar.title("Navigation")
        section = st.sidebar.radio(
            "Go to:",
            ["Overview", "Movie Recommendations", "Success Prediction", "Industry Insights"]
        )
        
        # Show selected section
        dashboard.show_overview()
        
        if section == "Movie Recommendations":
            dashboard.recommendation_section()
        elif section == "Success Prediction":
            dashboard.prediction_section()
        elif section == "Industry Insights":
            dashboard.data_insights_section()
    else:
        st.error("Failed to load data. Please check if the data files exist.")

if __name__ == "__main__":
    main()