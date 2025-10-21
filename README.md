# 🎬 Movie Success Prediction & Recommendation Engine

A comprehensive machine learning system that predicts movie success and provides personalized recommendations using TMDB data.

## 🚀 Features

### For Users
- **"If you liked X, you might like Y"** - Content-based recommendations
- **Genre-based discovery** - Find movies by preferred genres
- **Similarity matching** - Advanced NLP-based content similarity

### For Producers & Studios
- **Success Prediction** - ML models predicting box office success
- **Risk Assessment** - Budget and genre strategy analysis
- **Release Timing** - Optimal release window recommendations
- **Industry Insights** - Market trends and performance analytics

## 🛠️ Tech Stack

- **Python** - Core programming language
- **Pandas, NumPy** - Data manipulation
- **Scikit-learn, XGBoost, LightGBM** - Machine Learning
- **SpaCy, Transformers** - NLP processing
- **Streamlit** - Interactive dashboard
- **TMDB API** - Movie data source

## 📁 Project Structure
Movie_Success_Prediction/
├── data/ # Raw and processed datasets
├── src/ # Core Python modules
├── dashboard/ # Streamlit application
├── models/ # Trained ML models
├── notebooks/ # Jupyter notebooks
├── tests/ # Test scripts
└── requirements.txt # Dependencies



## 🏃‍♂️ Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/movie-success-predictor.git
cd movie-success-predictor

# Create virtual environment
python -m venv movie_env
source movie_env/bin/activate  # On Windows: movie_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLP models
python -m spacy download en_core_web_sm

# Run the dashboard 
streamlit run dashboard/app.py

# Run individual components

# Data collection
python src/data_collection.py

# Data processing
python src/data_processing.py

# Model training
python src/model_training.py

# Recommendation system
python src/recommendation.py

📊 Model Performance
Success Prediction Accuracy: 75-85% (depending on target metric)

Recommendation Quality: Content-based similarity with Sentence Transformers

Features Engineered: 55+ features including genres, financials, temporal patterns

🎯 Use Cases
Streaming Platforms - Personalized content discovery

Movie Studios - Greenlight decision support

Producers - Budget allocation and release strategy

Content Platforms - Acquisition and production planning

🔧 Configuration
Get TMDB API key from https://www.themoviedb.org/settings/api

Create .env file with:

env
TMDB_API_KEY=your_api_key_here
📈 Results
The system successfully:

Predicts movie success with high accuracy

Provides meaningful recommendations

Offers actionable insights for producers

Handles real-world movie data complexity

🤝 Contributing
Fork the project

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
TMDB for providing comprehensive movie data

Streamlit for the excellent dashboard framework

The open-source ML community for amazing libraries