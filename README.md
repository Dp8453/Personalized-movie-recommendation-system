# Personalized Movie Recommendation System

A content-based movie recommendation engine built using movie metadata, Natural Language Processing (NLP) techniques, vectorization, user preference modeling, and similarity ranking.

> [!IMPORTANT]
> **CURRENT PROJECT STATUS: PHASE 4 COMPLETED**
> - **Phase 1 (Completed)**: Project Foundation, Data Ingestion, ID-based Merging (`movies.id == credits.movie_id`), and Initial EDA.
> - **Phase 2 (Completed)**: Data Preprocessing, JSON Feature Extraction, Entity Space Collapsing, Overview Imputation, and Unified `tags` Construction.
> - **Phase 3 (Completed)**: TF-IDF Vectorization (`max_features=5000`, `stop_words='english'`), Cosine Similarity Matrix Modeling, Case-Insensitive Title Lookup, and Content-Based Recommendation Engine.
> - **Phase 4 (Completed)**: User Rating Preference Modeling, Weighted User Profile Construction ($\mathbf{u} = \frac{\sum w_i \mathbf{v}_i}{\sum |w_i|}$), Rated Movie Exclusion, and Content-Based Personalization Engine.
> - **Future Phases (Upcoming)**: Offline Evaluation Metrics (Precision@K, Recall@K, NDCG@K), FastAPI Service, and Streamlit UI.
>
> **NOTE ON PERSONALIZATION MODEL**: This phase implements **CONTENT-BASED PERSONALIZATION** by aggregating TF-IDF feature vectors of user-rated movies into a personalized user preference profile. It does **NOT** use Collaborative Filtering, deep learning, or external user databases.

---

## 🚀 End-to-End Project Architecture & Pipeline

```
  [ Raw Movie Metadata & Credits CSVs ]
       │
       ▼  ◄── PHASE 1 (COMPLETED: Primary Key ID Merging)
  [ Data Preprocessing & Feature Engineering ]
       │  ├── JSON Parsing (genres, keywords, cast, crew)
       │  ├── Top 3 Lead Cast & Director Extraction
       │  └── Unified Tags Construction (overview + genres + keywords + cast + director)
       │
       ▼  ◄── PHASE 2 (COMPLETED: Exported to data/processed/clean_movies.csv)
  [ Text Vectorization (TF-IDF) ]
       │  ├── TfidfVectorizer(max_features=5000, stop_words='english')
       │  └── TF-IDF Feature Matrix V Shape: (4803, 5000)
       │
       ▼  ◄── PHASE 3 (COMPLETED: Single-Movie Content Engine)
  [ User Rating History & Preference Modeling ]
       │  ├── Input Ratings: 1..5 Integer Ratings [("Avatar", 5), ("Titanic", 1)]
       │  ├── Preference Weight Mapping: 1 -> -1.0, 2 -> -0.5, 3 -> 0.0, 4 -> +0.5, 5 -> +1.0
       │  └── User Preference Profile Vector: u = sum(w_i * v_i) / sum(|w_i|)  Shape: (1, 5000)
       │
       ▼  ◄── PHASE 4 (COMPLETED: Content-Based Personalization Engine in src/personalizer.py)
  [ Personalized Cosine Similarity & Ranking ]
       │  ├── Single-Vector Cosine Similarity: cosine_similarity(u, V) Shape: (1, 4803)
       │  ├── Rated Movie Exclusion (History movies never returned)
       │  └── Top-N Personalized Movie Recommendations with Scores
       │
       ▼  ◄── PHASE 5 (UPCOMING)
  [ Offline Evaluation (Precision@K, Recall@K, NDCG@K) ]
       │
       ▼  ◄── PHASE 6 (UPCOMING)
  [ Production Web Service (FastAPI) & Frontend UI (Streamlit) ]
```

---

## 📊 Phase 4 Personalization Summary

- **User Rating Model**: Input rating history as a list of `(title, rating)` tuples where rating is an integer in `{1, 2, 3, 4, 5}`.
- **Preference Weight Transformation**:
  - `1` $\rightarrow -1.0$ (strongly disliked)
  - `2` $\rightarrow -0.5$ (disliked)
  - `3` $\rightarrow 0.0$ (neutral)
  - `4` $\rightarrow +0.5$ (liked)
  - `5` $\rightarrow +1.0$ (strongly liked)
- **User Profile Construction Formula**:
  $$\mathbf{u} = \frac{\sum_i w_i \cdot \mathbf{v}_i}{\sum_i |w_i|}$$
  Generates a single 5,000-dimensional preference vector $\mathbf{u}$.
- **Computational Efficiency**: Computes Cosine Similarity between single profile vector $\mathbf{u}$ `(1, 5000)` and all movie vectors `V` `(4803, 5000)`, running in $O(N)$ time (< 0.01 seconds).
- **Core Engine Features (`src/personalizer.py`)**:
  - `PersonalizedRecommender` class.
  - Validates ratings, rejects duplicates and non-integer/out-of-range ratings with `ValueError`.
  - Case-insensitive title resolution.
  - Strict exclusion of all movies present in the user's rating history.
  - Handles the net-weight zero `[5, 1]` preference case cleanly ($\sum |w_i| = 2.0 \neq 0$).

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
│   ├── 03_recommendation_engine.ipynb
│   └── 04_user_personalization.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── recommender.py
│   └── personalizer.py
│
├── docs/
│   └── reference-analysis.md
│
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_preprocessor.py
│   ├── test_recommender.py
│   └── test_personalizer.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── run.py
```

---

## 🛠️ Quick Start Guide

### 1. Prerequisites & Installation

Clone the repository and install requirements:

```bash
git clone https://github.com/Dp8453/Personalized-movie-recommendation-system.git
cd Personalized-movie-recommendation-system
pip install -r requirements.txt
```

### 2. Run Main Pipeline Verification

Execute data loading, preprocessing, single-movie lookup, and personalized user profile recommendation:

```bash
python run.py
```

### 3. Run Complete Unit Test Suite

Execute all 45 unit tests across Phases 1–4:

```bash
python -m unittest discover -s tests -v
```

### 4. Explore Personalization Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook notebooks/04_user_personalization.ipynb
```

---

## 📌 Implementation Roadmap

- [x] **Phase 1**: Project Foundation, Dataset Setup (ID-based merge), Modular Data Loader & Initial EDA
- [x] **Phase 2**: Data Preprocessing, JSON Feature Extraction, Entity Space Collapsing & Tags Construction
- [x] **Phase 3**: Vectorization (TF-IDF), Similarity Computation & Content-Based Recommendation Engine
- [x] **Phase 4**: User Preference Rating Model, Weighted User Profile Vector Construction & Content Personalization
- [ ] **Phase 5**: Offline Model Evaluation (Precision@K, Recall@K, NDCG@K)
- [ ] **Phase 6**: Web API Deployment (FastAPI) & Frontend UI (Streamlit)
