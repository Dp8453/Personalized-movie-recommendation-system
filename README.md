# Personalized Movie Recommendation System

A content-based movie recommendation engine built using movie metadata, Natural Language Processing (NLP) techniques, vectorization, and similarity modeling.

> [!IMPORTANT]
> **CURRENT PROJECT STATUS: PHASE 3 COMPLETED**
> - **Phase 1 (Completed)**: Project Foundation, Data Ingestion, ID-based Merging (`movies.id == credits.movie_id`), and Initial EDA.
> - **Phase 2 (Completed)**: Data Preprocessing, JSON Feature Extraction, Entity Space Collapsing, Overview Imputation, and Unified `tags` Construction.
> - **Phase 3 (Completed)**: TF-IDF Vectorization (`max_features=5000`, `stop_words='english'`), Cosine Similarity Matrix Modeling, Case-Insensitive Title Lookup, and Content-Based Recommendation Engine.
> - **Future Phases (Upcoming)**: User Personalization & Hybrid Ranking, Evaluation Metrics (Precision@K, Recall@K, NDCG@K), FastAPI Service, and Streamlit UI.

---

## 🚀 End-to-End Project Architecture & Pipeline

```
  [ Dataset Ingestion & ID Merge ] (4,803 Unique Movies)
       │
       ▼  ◄── PHASE 1 (COMPLETED)
  [ Data Preprocessing & Feature Engineering ]
       │  ├── JSON Parsing (genres, keywords, cast, crew)
       │  ├── Top 3 Lead Cast & Director Extraction
       │  ├── Entity Space Collapsing ("Sam Worthington" -> "SamWorthington")
       │  └── Unified Tags Construction (overview + genres + keywords + cast + director)
       │
       ▼  ◄── PHASE 2 (COMPLETED: Exported to data/processed/clean_movies.csv)
  [ Text Vectorization (TF-IDF) & Cosine Similarity ]
       │  ├── TfidfVectorizer(max_features=5000, stop_words='english')
       │  ├── Sparse Feature Matrix Shape: (4803, 5000)
       │  └── Pairwise Cosine Similarity Matrix Shape: (4803, 4803)
       │
       ▼  ◄── PHASE 3 (COMPLETED: Classical Content-Based Engine in src/recommender.py)
  [ Content-Based Recommendation Engine ]
       │  ├── Case-Insensitive Title Lookup ("Avatar" == "avatar" == "AVATAR")
       │  ├── Query Movie Self-Exclusion & Rank Sorting
       │  └── Top-N Recommended Movies with Similarity Scores
       │
       ▼
  [ Personalization & Hybrid Ranking ]
       │
       ▼  ◄── PHASE 4 (UPCOMING)
  [ Offline Evaluation (Precision@K, Recall@K, NDCG@K) ]
       │
       ▼  ◄── PHASE 5 (UPCOMING)
  [ Production Web Service (FastAPI) & Frontend UI (Streamlit) ]
```

---

## 📊 Phase 3 Recommendation Engine Summary

- **Vectorization Technique**: `TfidfVectorizer` (max_features=5000, English stop words removed).
- **Matrix Shapes**:
  - **TF-IDF Feature Matrix**: `(4803, 5000)` (4,803 movies × 5,000 vocabulary terms).
  - **Cosine Similarity Matrix**: `(4803, 4803)` (Pairwise similarity across all movie vectors).
- **Core Engine Features (`src/recommender.py`)**:
  - `MovieRecommender` class fitted on `clean_movies.csv` `tags`.
  - Case-insensitive title resolution (`recommend("avatar")`, `recommend("AVATAR")`).
  - `ValueError` exception raised for non-existent movie titles.
  - Query movie self-exclusion (the query movie itself is never returned).
  - Descending score sorting & `top_n` bounds handling.

---

## 📁 Repository Structure

```
Personalized-movie-recommendation-system/
│
├── data/
│   ├── raw/                  # Raw TMDB 5000 CSV files (git-ignored)
│   │   ├── tmdb_5000_movies.csv
│   │   └── tmdb_5000_credits.csv
│   └── processed/            # Cleaned data output (git-ignored)
│       └── clean_movies.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   └── 03_recommendation_engine.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessor.py
│   └── recommender.py
│
├── docs/
│   └── reference-analysis.md
│
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_preprocessor.py
│   └── test_recommender.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── run.py
```

---

## 🛠️ Quick Start Guide

### 1. Prerequisites & Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Dp8453/Personalized-movie-recommendation-system.git
cd Personalized-movie-recommendation-system
pip install -r requirements.txt
```

### 2. Run Main Pipeline Verification

Execute end-to-end data loading, preprocessing, and recommendation generation:

```bash
python run.py
```

### 3. Run Unit Test Suite

Execute all 26 unit tests across Phase 1, Phase 2, and Phase 3:

```bash
python -m unittest discover -s tests -v
```

### 4. Explore Recommendation Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook notebooks/03_recommendation_engine.ipynb
```

---

## 📌 Implementation Roadmap

- [x] **Phase 1**: Project Foundation, Dataset Setup (ID-based merge), Modular Data Loader & Initial EDA
- [x] **Phase 2**: Data Preprocessing, JSON Feature Extraction, Entity Space Collapsing & Tags Construction
- [x] **Phase 3**: Vectorization (TF-IDF), Similarity Computation & Content-Based Recommendation Engine
- [ ] **Phase 4**: User Personalization & Hybrid Ranking Logic
- [ ] **Phase 5**: Model Evaluation (Precision@K, Recall@K, NDCG@K)
- [ ] **Phase 6**: Web API Deployment (FastAPI) & Frontend UI (Streamlit)
