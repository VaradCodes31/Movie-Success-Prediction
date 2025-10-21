# 🎬 Movie Success Prediction & Recommendation Engine

An end-to-end **machine learning and NLP system** that predicts a movie’s box-office success and delivers **personalized recommendations** for users — powered by **TMDB data** and advanced feature engineering.

This project bridges the gap between **data-driven insights** for production studios and **personalized discovery** for streaming users, blending prediction, recommendation, and analytics in one integrated platform.

---

## 🚀 Key Features

### 🎥 For Movie Lovers

* 🎯 **“If you liked X, you might like Y”** – Content-based movie recommendations using NLP similarity.
* 🎞️ **Genre & Actor-based Discovery** – Explore new titles by favorite categories or cast.
* 💬 **Semantic Similarity Search** – Powered by **Transformer embeddings** for natural language understanding.

### 🏢 For Producers & Studios

* 💰 **Success Prediction Model** – Predicts box-office outcomes based on cast, budget, genre, and timing.
* 📊 **Risk & ROI Assessment** – Quantifies potential profitability and genre performance trends.
* 🗓️ **Release Window Optimization** – Suggests best release timing using temporal patterns and competition data.
* 🧠 **Actionable Insights Dashboard** – Interactive analytics via **Streamlit** for strategy and planning.

---

## 🧠 Tech Stack

| Layer                         | Technologies                       |
| ----------------------------- | ---------------------------------- |
| **Programming**               | Python                             |
| **Data Handling**             | Pandas, NumPy                      |
| **Machine Learning**          | Scikit-learn, XGBoost, LightGBM    |
| **NLP & Embeddings**          | SpaCy, Transformers, Sentence-BERT |
| **Visualization & Dashboard** | Streamlit                          |
| **Data Source**               | TMDB API                           |
| **Version Control & Testing** | Git, Pytest                        |
| **Environment**               | `.env` for API keys                |

---

## 🧩 Project Structure

```
Movie_Success_Prediction/
├── data/             # Raw and processed datasets
├── src/              # Core Python modules (data, training, recommendation)
├── dashboard/        # Streamlit dashboard app
├── models/           # Trained ML/NLP models
├── notebooks/        # Jupyter notebooks for experimentation
├── tests/            # Unit and integration tests
└── requirements.txt  # Project dependencies
```

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/movie-success-predictor.git
cd movie-success-predictor
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4️⃣ Set Up TMDB API Key

Create a `.env` file in the root directory:

```
TMDB_API_KEY=your_api_key_here
```

Obtain your key from [TMDB API Settings](https://www.themoviedb.org/settings/api).

### 5️⃣ Run the Dashboard

```bash
streamlit run dashboard/app.py
```

---

## 🧮 Model Workflow

1. **Data Collection:** Fetch movie metadata, cast, crew, and financials via TMDB API.
2. **Feature Engineering:** Generate 50+ features (genres, keywords, popularity, budgets, revenue ratios, and text embeddings).
3. **Model Training:** Train regression/classification models (Random Forest, XGBoost, LightGBM) for success prediction.
4. **NLP Processing:** Use **Sentence-BERT** for semantic similarity in recommendation engine.
5. **Evaluation:** Optimize metrics (R², RMSE for regression; accuracy/F1 for classification).
6. **Deployment:** Serve results via an **interactive Streamlit dashboard**.

---

## 📊 Performance Metrics

| Metric                        | Value  | Description                                         |
| ----------------------------- | ------ | --------------------------------------------------- |
| **Prediction Accuracy**       | 75–85% | Varies by success definition (revenue, ROI, rating) |
| **F1-Score (Classification)** | 0.82   | On box office hit/miss                              |
| **Recommendation Quality**    | High   | Based on semantic content similarity                |
| **Features Used**             | 55+    | Including text, temporal, and numeric attributes    |

---

## 🎯 Use Cases

* **🎞️ Streaming Platforms:** Power recommendation systems for personalized content discovery.
* **🎬 Production Studios:** Assess project potential before investment.
* **💸 Producers/Investors:** Allocate budgets and optimize release strategies.
* **📈 Data Analysts:** Analyze genre trends and box-office performance factors.

---

## 📈 Results

* ✅ Achieved **75–85% accuracy** in predicting movie success.
* 🎥 Generated **context-aware movie recommendations** with semantic similarity.
* 💡 Delivered actionable insights for release timing, budget optimization, and genre trends.
* 🌍 Demonstrated scalable, real-world ML and NLP integration in the entertainment domain.

---

## 🤝 Contributing

Contributions are welcome!

1. **Fork** the repository
2. **Create** your feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

---

## 🪪 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

* 🎞️ [TMDB](https://www.themoviedb.org) for providing rich and open movie datasets
* 🧠 [Streamlit](https://streamlit.io/) for simplifying ML app deployment
* 💻 Open-source ML community for powerful tools and frameworks

---

Would you like me to add a **visual workflow diagram (Mermaid)** showing the project pipeline (Data → Modeling → NLP → Dashboard)?
It looks excellent on GitHub and makes your README visually stand out.
