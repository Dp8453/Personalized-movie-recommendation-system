# Personalized Movie Recommendation System

A content-based, item-based collaborative filtering, hybrid recommendation, and explainable AI engine built using TMDB 5000 metadata and MovieLens user interaction logs. Features NLP vectorization, preference modeling, sparse matrix similarity, temporal offline evaluation, evidence-based explainability, and a production REST API layer.

> [!IMPORTANT]
> **PROJECT STATUS: PHASE 10 COMPLETED (PROJECT FROZEN)**
> - **Phase 1 (Completed)**: Project Foundation, Data Ingestion, ID-based Merging (`movies.id == credits.movie_id`), and Initial EDA.
> - **Phase 2 (Completed)**: Data Preprocessing, JSON Feature Extraction, Entity Space Collapsing, Overview Imputation, and Unified `tags` Construction.
> - **Phase 3 (Completed)**: TF-IDF Vectorization (`max_features=5000`, `stop_words='english'`), Cosine Similarity Matrix Modeling, Case-Insensitive Title Lookup, and Content-Based Recommendation Engine.
> - **Phase 4 (Completed)**: User Rating Preference Modeling, Weighted User Profile Construction ($\mathbf{u} = \frac{\sum w_i \mathbf{v}_i}{\sum |w_i|}$), Rated Movie Exclusion, and Content-Based Personalization Engine.
> - **Phase 5 (Completed)**: Offline Evaluation Framework (Precision@K, Recall@K, NDCG@K) using a deterministic held-out preference protocol.
> - **Phase 6 (Completed)**: Item-Based Collaborative Filtering (`ItemBasedCollaborativeRecommender`) using genuine user interaction data from **MovieLens latest-small** and leakage-safe temporal evaluation.
> - **Phase 7 (Completed)**: Hybrid Movie Recommendation System (`HybridMovieRecommender`), candidate pool union ($N_{\text{cand}}=100$), Min-Max score normalization, title identity mapping alignment, and alpha ablation benchmarking.
> - **Phase 8 (Completed)**: Explainable Recommendation Analysis Layer (`RecommendationExplainer`), transparently decomposing model recommendations into factual content metadata overlaps, collaborative item-item rating contributions, and hybrid score weights with human-readable summary generation.
> - **Phase 9 (Completed)**: Production Recommendation Serving & API Layer (`src/api.py`), exposing Content, Personalized, Collaborative, Hybrid, and Explainable recommendations via lightweight REST endpoints (FastAPI, Pydantic, Uvicorn).
> - **Phase 10 (Completed)**: Final ML Evaluation, System Engineering Audit, Resume Readiness Documentation, and Project Freeze (`notebooks/10_final_project_audit.ipynb`).
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
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼  ◄── PHASE 8, 9 & 10
[ Offline Evaluation ]  [ Explainer ]  [ FastAPI REST Serving Layer ]  [ Final System Audit ]
(Precision, Recall, NDCG) (Evidence)   (/health, /recommend/*, /explain) (notebooks/10_*)
```

---

## 📊 Phase 10 — Consolidated ML Evaluation & Benchmark Audit

### 1. Unified Recommendation Performance Summary

| Paradigm | Evaluated Dataset | Metric @ K=10 | Score | Benchmark Protocol / Notes |
| :--- | :--- | :--- | :---: | :--- |
| **Content-Based Profile** | TMDB 5000 Synthetic History | **NDCG@5** | **0.8065** | Held-out preference profile test ($K=5$) |
| | | **Recall@5** | **0.7500** | Direct TF-IDF cosine matching against synthetic preferences |
| | | **Precision@5** | **0.3000** | Evaluated on 2 representative test user profiles |
| **Item-Based CF ($\alpha=0.00$)** | MovieLens latest-small | **NDCG@10** | **0.0954** | Leakage-safe temporal cutoff ($T \le \text{split}$), 50 test users |
| | | **Recall@10** | **0.0683** | Evaluated against future held-out user interactions |
| | | **Precision@10** | **0.0800** | Standard item-item cosine similarity ranking |
| **Hybrid ($\alpha=0.25$)** | MovieLens latest-small | **NDCG@10** | **0.0729** | CF-dominant candidate union fusion |
| **Hybrid ($\alpha=0.50$)** | MovieLens latest-small | **NDCG@10** | **0.0617** | Balanced content & collaborative fusion |
| **Hybrid ($\alpha=0.75$)** | MovieLens latest-small | **NDCG@10** | **0.0236** | Content-dominant candidate union fusion |
| **Content-Only Hybrid ($\alpha=1.00$)**| MovieLens latest-small | **NDCG@10** | **0.0157** | Candidate union pool weighted by content score only |

### 2. Strategic Insights & Model Audit
- **Collaborative Dominance**: Pure item-based collaborative filtering ($\alpha=0.00$) achieves the highest ranking efficiency ($NDCG@10 = 0.0954$) on actual user interaction data.
- **Title Identity Alignment**: 2,812 / 9,742 MovieLens movies (28.86% coverage) match TMDB metadata via exact normalized title mapping. Unmapped items remain available to CF recommendations without receiving invalid content scores.
- **Cold-Start Fallback**: Content-based profiles provide an item-side fallback for cold items with rich text metadata but sparse collaborative interactions. Pure user cold start requires initial rating history to build profile vector $\mathbf{u}$.

---

## 💼 System Engineering & Resume Capabilities

This repository demonstrates production-grade machine learning system engineering principles across the complete recommender lifecycle:

1. **Natural Language Processing & Vectorization**: TF-IDF feature extraction (`max_features=5000`), entity collapsing, overview text cleaning, and cosine similarity metric computation.
2. **User Profile Modeling**: Linear preference weighting ($\mathbf{u} = \frac{\sum w_i \mathbf{v}_i}{\sum |w_i|}$) mapping user ratings ($1 \dots 5 \to -1.0 \dots +1.0$) into sparse feature vector space.
3. **Item-Based Collaborative Filtering**: Memory-efficient SciPy CSR sparse matrix representation, rating mean-centering, pairwise item similarity computation, and leakage-safe temporal evaluation.
4. **Hybrid Scoring Architecture**: Union candidate pool selection ($N_{\text{cand}}=100$), Min-Max score normalization, title identity resolution, and parameter ablation.
5. **Deterministic Explainable AI**: Sub-graph evidence extraction isolating exact metadata tag matches, historical rating contributions, and component fusion weights without LLM hallucination or evaluation data leakage.
6. **API Serving & Pydantic Validation**: FastAPI REST backend featuring lifespan singleton initialization, strict custom validators (`field_validator`), structured error handlers, and zero-warning unit test suite (147 passing tests).

---

## ⚡ Phase 9 — Production Recommendation Serving & API Layer

### 1. API Architecture & Available Endpoints

FastAPI application (`src/api.py`) exposing recommendation endpoints:

| Endpoint | Method | Description | Example Query / Body |
| :--- | :---: | :--- | :--- |
| `/health` | `GET` | Service readiness & dataset status | `GET /health` |
| `/recommend/content` | `GET` | Single-movie content recommendations | `GET /recommend/content?title=Avatar&top_n=5` |
| `/recommend/personalized` | `POST` | User profile content recommendations | `{"history": [{"title": "Avatar", "rating": 5}], "top_n": 5}` |
| `/recommend/hybrid` | `POST` | Hybrid content & CF recommendations | `{"user_id": 1, "alpha": 0.5, "top_n": 5}` |
| `/recommend/explain` | `POST` | Recommendation evidence breakdown & summary | `{"user_id": 1, "target_movie_id": 2918, "alpha": 0.5}` |

### 2. Launching the Local API Server

```bash
python -m uvicorn src.api:app --reload
```

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
│   ├── 07_hybrid_recommendation.ipynb
│   ├── 08_recommendation_explainability.ipynb
│   ├── 09_api_serving_demo.ipynb
│   └── 10_final_project_audit.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── recommender.py
│   ├── personalizer.py
│   ├── evaluator.py
│   ├── collaborative_filter.py
│   ├── hybrid_recommender.py
│   ├── explanation.py
│   └── api.py
│
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_preprocessor.py
│   ├── test_recommender.py
│   ├── test_personalizer.py
│   ├── test_evaluator.py
│   ├── test_collaborative_filter.py
│   ├── test_hybrid_recommender.py
│   ├── test_explanation.py
│   └── test_api.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── run.py
```

---

## 🛠️ Quick Start Guide

### 1. Prerequisites & Installation

```bash
git clone https://github.com/Dp8453/Personalized-movie-recommendation-system.git
cd Personalized-movie-recommendation-system
pip install -r requirements.txt
```

### 2. Download Optional MovieLens Dataset

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

```bash
python run.py
```

### 4. Run Complete Unit Test Suite

Execute all 147 unit tests across Phases 1–9:

```bash
pytest
```

---

## 📌 Project Completion & Freeze Declaration

- [x] **Phase 1**: Project Foundation, Dataset Setup (ID-based merge), Modular Data Loader & Initial EDA
- [x] **Phase 2**: Data Preprocessing, JSON Feature Extraction, Entity Space Collapsing & Tags Construction
- [x] **Phase 3**: Vectorization (TF-IDF), Similarity Computation & Content-Based Recommendation Engine
- [x] **Phase 4**: User Preference Rating Model, Weighted User Profile Vector Construction & Content Personalization
- [x] **Phase 5**: Offline Model Evaluation (Precision@K, Recall@K, NDCG@K)
- [x] **Phase 6**: Item-Based Collaborative Filtering Engine & Temporal Evaluation
- [x] **Phase 7**: Hybrid Movie Recommendation System, Candidate Pool Union, Min-Max Normalization & Ablation Evaluation
- [x] **Phase 8**: Explainable Recommendation Analysis Layer (`RecommendationExplainer`) & Summary Generation
- [x] **Phase 9**: Production Recommendation Serving & API Layer (`src/api.py`)
- [x] **Phase 10**: Final ML Evaluation, Engineering Audit & Project Freeze (`notebooks/10_final_project_audit.ipynb`)

**THE PROJECT IS OFFICIALLY COMPLETE AND FROZEN.**

