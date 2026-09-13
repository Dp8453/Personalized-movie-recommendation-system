# Personalized Movie Recommendation System

A content-based and collaborative-filtering movie recommendation engine built using movie metadata, Natural Language Processing (NLP) techniques, vectorization, user preference modeling, item-based collaborative filtering, similarity ranking, and offline evaluation metrics.

> [!IMPORTANT]
> **CURRENT PROJECT STATUS: PHASE 7 COMPLETED**
> - **Phase 1 (Completed)**: Project Foundation, Data Ingestion, ID-based Merging (`movies.id == credits.movie_id`), and Initial EDA.
> - **Phase 2 (Completed)**: Data Preprocessing, JSON Feature Extraction, Entity Space Collapsing, Overview Imputation, and Unified `tags` Construction.
> - **Phase 3 (Completed)**: TF-IDF Vectorization (`max_features=5000`, `stop_words='english'`), Cosine Similarity Matrix Modeling, Case-Insensitive Title Lookup, and Content-Based Recommendation Engine.
> - **Phase 4 (Completed)**: User Rating Preference Modeling, Weighted User Profile Construction ($\mathbf{u} = \frac{\sum w_i \mathbf{v}_i}{\sum |w_i|}$), Rated Movie Exclusion, and Content-Based Personalization Engine.
> - **Phase 5 (Completed)**: Offline Evaluation Framework (Precision@K, Recall@K, NDCG@K) using a deterministic held-out preference protocol.
> - **Phase 6 (Completed)**: Item-Based Collaborative Filtering (`ItemBasedCollaborativeRecommender`) using genuine user interaction data from **MovieLens latest-small** and leakage-safe temporal evaluation.
> - **Phase 7 (Completed)**: Hybrid Movie Recommendation System (`HybridMovieRecommender`), candidate pool union ($N_{\text{cand}}=100$), Min-Max score normalization, title identity mapping alignment, and alpha ablation benchmarking.
> - **Future Phases (Upcoming)**: Web API Service (FastAPI) and Frontend UI (Streamlit).
>
> **DATASET POLICY & DISCLAIMER**:
> 1. TMDB 5000 is a movie metadata dataset containing no multi-user interaction logs and is **NOT** used for collaborative filtering.
> 2. MovieLens latest-small (100,836 ratings across 610 users and 9,724 movies) provides genuine user behavioral interactions for Phase 6 & 7. Raw MovieLens files are kept locally in `data/raw/movielens/` and strictly excluded from Git tracking.
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
                          ▼  ◄── PHASE 7 (Candidate Union & Min-Max Normalization)
             [ Hybrid Movie Recommender ]
             score = α * norm_content + (1 - α) * norm_cf
                          │
                          ▼  ◄── PHASE 5, 6 & 7
             [ Offline Evaluation Framework ]
             (Precision@K, Recall@K, NDCG@K)
```

---

## 🔀 Phase 7 — Hybrid Movie Recommendation System

### 1. Motivation & Hybrid Architecture
Phase 7 fuses Phase 4 personalized content-based recommendation and Phase 6 item-based collaborative filtering into `HybridMovieRecommender`.

- **Candidate Pool Union ($N_{\text{cand}}=100$)**: Unions top 100 candidates from Content engine and top 100 candidates from Collaborative Filtering engine, eliminating selection bottlenecks.
- **Min-Max Score Normalization**: Scales raw content similarity scores and raw CF predicted scores onto $[0.0, 1.0]$ per candidate pool:
  $$\text{norm\_score}(c) = \frac{\text{raw\_score}(c) - \min(S)}{\max(S) - \min(S)}$$
- **Weighted Hybrid Scoring**:
  $$\text{score}_{\text{hybrid}}(c) = \alpha \cdot \text{norm\_content\_score}(c) + (1 - \alpha) \cdot \text{norm\_cf\_score}(c)$$
- **Title Identity Alignment**: Deterministically maps MovieLens normalized titles to TMDB clean titles (2,812 out of 9,742 movies mapped, 28.86% coverage) for content profile matching while maintaining MovieLens `movieId` as primary recommendation identity.

### 2. Alpha Ablation Benchmark Results ($K=10$, 50 Evaluated Users)

| Model | Alpha ($\alpha$) | Precision@10 | Recall@10 | NDCG@10 |
| :--- | :---: | :---: | :---: | :---: |
| **Pure Item-Based CF** | 0.00 | **0.0800** | **0.0683** | **0.0954** |
| **CF-Dominant Hybrid** | 0.25 | 0.0620 | 0.0492 | 0.0729 |
| **Balanced Hybrid** | 0.50 | 0.0620 | 0.0485 | 0.0617 |
| **Content-Dominant Hybrid** | 0.75 | 0.0160 | 0.0311 | 0.0236 |
| **Pure Personalized Content** | 1.00 | 0.0080 | 0.0228 | 0.0157 |

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
│   ├── 06_collaborative_filtering.ipynb
│   └── 07_hybrid_recommendation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── recommender.py
│   ├── personalizer.py
│   ├── evaluator.py
│   ├── collaborative_filter.py
│   └── hybrid_recommender.py
│
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_preprocessor.py
│   ├── test_recommender.py
│   ├── test_personalizer.py
│   ├── test_evaluator.py
│   ├── test_collaborative_filter.py
│   └── test_hybrid_recommender.py
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

To run Phase 6 & 7 collaborative and hybrid recommendation on genuine user interaction data, download MovieLens latest-small:

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

Execute data loading, preprocessing, single-movie lookup, personalized user profile recommendation, evaluation, collaborative filtering, and hybrid recommendation:

```bash
python run.py
```

### 4. Run Complete Unit Test Suite

Execute all 106 unit tests across Phases 1–7:

```bash
python -m unittest discover -s tests -v
```

### 5. Explore Hybrid Recommendation Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook notebooks/07_hybrid_recommendation.ipynb
```

---

## 📌 Implementation Roadmap

- [x] **Phase 1**: Project Foundation, Dataset Setup (ID-based merge), Modular Data Loader & Initial EDA
- [x] **Phase 2**: Data Preprocessing, JSON Feature Extraction, Entity Space Collapsing & Tags Construction
- [x] **Phase 3**: Vectorization (TF-IDF), Similarity Computation & Content-Based Recommendation Engine
- [x] **Phase 4**: User Preference Rating Model, Weighted User Profile Vector Construction & Content Personalization
- [x] **Phase 5**: Offline Model Evaluation (Precision@K, Recall@K, NDCG@K)
- [x] **Phase 6**: Item-Based Collaborative Filtering Engine & Temporal Evaluation
- [x] **Phase 7**: Hybrid Movie Recommendation System, Candidate Pool Union, Min-Max Normalization & Ablation Evaluation
- [ ] **Phase 8**: Web API Deployment (FastAPI) & Frontend UI (Streamlit)
