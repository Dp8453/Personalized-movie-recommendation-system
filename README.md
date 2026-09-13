# Personalized Movie Recommendation System

A content-based and collaborative-filtering movie recommendation engine built using movie metadata, Natural Language Processing (NLP) techniques, vectorization, user preference modeling, item-based collaborative filtering, similarity ranking, and offline evaluation metrics.

> [!IMPORTANT]
> **CURRENT PROJECT STATUS: PHASE 6 COMPLETED**
> - **Phase 1 (Completed)**: Project Foundation, Data Ingestion, ID-based Merging (`movies.id == credits.movie_id`), and Initial EDA.
> - **Phase 2 (Completed)**: Data Preprocessing, JSON Feature Extraction, Entity Space Collapsing, Overview Imputation, and Unified `tags` Construction.
> - **Phase 3 (Completed)**: TF-IDF Vectorization (`max_features=5000`, `stop_words='english'`), Cosine Similarity Matrix Modeling, Case-Insensitive Title Lookup, and Content-Based Recommendation Engine.
> - **Phase 4 (Completed)**: User Rating Preference Modeling, Weighted User Profile Construction ($\mathbf{u} = \frac{\sum w_i \mathbf{v}_i}{\sum |w_i|}$), Rated Movie Exclusion, and Content-Based Personalization Engine.
> - **Phase 5 (Completed)**: Offline Evaluation Framework (Precision@K, Recall@K, NDCG@K) using a deterministic held-out preference protocol.
> - **Phase 6 (Completed)**: Item-Based Collaborative Filtering (`ItemBasedCollaborativeRecommender`) using genuine user interaction data from **MovieLens latest-small** and leakage-safe temporal evaluation.
> - **Future Phases (Upcoming)**: Web API Service (FastAPI) and Frontend UI (Streamlit).
>
> **DATASET POLICY & DISCLAIMER**:
> 1. TMDB 5000 is a movie metadata dataset containing no multi-user interaction logs and is **NOT** used for collaborative filtering.
> 2. MovieLens latest-small (100,836 ratings across 610 users and 9,724 movies) provides genuine user behavioral interactions for Phase 6. Raw MovieLens files are kept locally in `data/raw/movielens/` and strictly excluded from Git tracking.
> 3. MovieLens provides genuine user interaction logs, but offline evaluation metrics on benchmark datasets do **NOT** guarantee real-world production performance.

---

## 🚀 End-to-End Project Architecture & Pipeline

```
  [ TMDB 5000 Metadata CSVs ]          [ MovieLens Interaction CSVs ]
       │                                     │
       ▼  ◄── PHASE 1 & 2                   ▼  ◄── PHASE 6
  [ Text Vectorization & Tags ]       [ Sparse Item x User Matrix R ]
       │                                     │
       ▼  ◄── PHASE 3 & 4                   ▼  ◄── PHASE 6
  [ Content User Profile Vector u ]   [ Item-Item Cosine Similarity S ]
       │                                     │
       ▼                                     ▼
  [ Content Personalization Engine ]  [ Collaborative Filtering Engine ]
       │                                     │
       └──────────────────┬──────────────────┘
                          │
                          ▼  ◄── PHASE 5 & 6
             [ Offline Evaluation Framework ]
             (Precision@K, Recall@K, NDCG@K)
```

---

## 🤝 Phase 6 — Item-Based Collaborative Filtering

### 1. Motivation & Architecture
While Phases 3–4 built a content-based recommendation system from metadata, Phase 6 adds behavioral collaborative filtering capability based on genuine user co-rating patterns in **MovieLens latest-small**.

- **Item-User Sparse Matrix**: Constructs a SciPy CSR sparse matrix $R$ of shape $(\text{num\_movies} \times \text{num\_users})$.
- **Item-Item Cosine Similarity**: Computes similarity matrix $S(i, j) = \frac{\mathbf{R}_i \cdot \mathbf{R}_j}{\|\mathbf{R}_i\| \|\mathbf{R}_j\|}$ with diagonal $S(i, i) = 0.0$ to exclude self-recommendation.
- **Personalized Candidate Scoring**:
  $$\text{score}(c) = \sum_{m \in \text{user\_pos\_movies}} S(m, c) \times \text{rating}(m)$$
- **Strict Exclusion**: Excludes all movies already rated by the target user.

### 2. Leakage-Safe Temporal Evaluation Protocol
1. **Chronological Split**: Ratings per user are sorted by `timestamp`. Earliest 80% form training history; latest 20% positive ratings ($\ge 4.0$) are held-out test targets.
2. **Model Fitting Constraint**: Model fitting occurs **STRICTLY on training ratings split**. Held-out test ratings are NEVER seen during model fitting.
3. **Data Leakage Safeguards**: Programmatically verifies that held-out test items are absent from user training histories and rated training items are never recommended.

---

## 📁 Repository Structure

```
Personalized-movie-recommendation-system/
│
├── data/
│   ├── raw/                  # Raw CSV files (git-ignored)
│   │   ├── tmdb_5000_movies.csv
│   │   ├── tmdb_5000_credits.csv
│   │   └── movielens/        # MovieLens latest-small CSVs
│   │       └── ml-latest-small/
│   │           ├── ratings.csv
│   │           ├── movies.csv
│   │           └── links.csv
│   └── processed/            # Cleaned data output (git-ignored)
│       └── clean_movies.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_recommendation_engine.ipynb
│   ├── 04_user_personalization.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── 06_collaborative_filtering.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── recommender.py
│   ├── personalizer.py
│   ├── evaluator.py
│   └── collaborative_filter.py
│
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_preprocessor.py
│   ├── test_recommender.py
│   ├── test_personalizer.py
│   ├── test_evaluator.py
│   └── test_collaborative_filter.py
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

### 2. Download Optional MovieLens Dataset

To run Phase 6 collaborative filtering on genuine user interaction data, download MovieLens latest-small:

```bash
python -c "
import urllib.request, ssl, zipfile, pathlib
d = pathlib.Path('data/raw/movielens')
d.mkdir(parents=True, exist_ok=True)
z = d / 'ml-latest-small.zip'
urllib.request.urlretrieve('https://files.grouplens.org/datasets/movielens/ml-latest-small.zip', z, context=ssl._create_unverified_context())
zipfile.ZipFile(z).extractall(d)
"
```

### 3. Run Main Pipeline Verification

Execute data loading, preprocessing, single-movie lookup, personalized user profile recommendation, evaluation, and collaborative filtering:

```bash
python run.py
```

### 4. Run Complete Unit Test Suite

Execute all 87 unit tests across Phases 1–6:

```bash
python -m unittest discover -s tests -v
```

### 5. Explore Collaborative Filtering Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook notebooks/06_collaborative_filtering.ipynb
```

---

## 📌 Implementation Roadmap

- [x] **Phase 1**: Project Foundation, Dataset Setup (ID-based merge), Modular Data Loader & Initial EDA
- [x] **Phase 2**: Data Preprocessing, JSON Feature Extraction, Entity Space Collapsing & Tags Construction
- [x] **Phase 3**: Vectorization (TF-IDF), Similarity Computation & Content-Based Recommendation Engine
- [x] **Phase 4**: User Preference Rating Model, Weighted User Profile Vector Construction & Content Personalization
- [x] **Phase 5**: Offline Model Evaluation (Precision@K, Recall@K, NDCG@K)
- [x] **Phase 6**: Item-Based Collaborative Filtering Engine & Temporal Evaluation
- [ ] **Phase 7**: Web API Deployment (FastAPI) & Frontend UI (Streamlit)
