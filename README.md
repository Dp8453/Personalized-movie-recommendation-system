# Personalized Movie Recommendation System

A content-based and collaborative-filtering movie recommendation engine built using movie metadata, Natural Language Processing (NLP) techniques, vectorization, user preference modeling, item-based collaborative filtering, similarity ranking, and offline evaluation metrics.

> [!IMPORTANT]
> **CURRENT PROJECT STATUS: PHASE 9 COMPLETED**
> - **Phase 1 (Completed)**: Project Foundation, Data Ingestion, ID-based Merging (`movies.id == credits.movie_id`), and Initial EDA.
> - **Phase 2 (Completed)**: Data Preprocessing, JSON Feature Extraction, Entity Space Collapsing, Overview Imputation, and Unified `tags` Construction.
> - **Phase 3 (Completed)**: TF-IDF Vectorization (`max_features=5000`, `stop_words='english'`), Cosine Similarity Matrix Modeling, Case-Insensitive Title Lookup, and Content-Based Recommendation Engine.
> - **Phase 4 (Completed)**: User Rating Preference Modeling, Weighted User Profile Construction ($\mathbf{u} = \frac{\sum w_i \mathbf{v}_i}{\sum |w_i|}$), Rated Movie Exclusion, and Content-Based Personalization Engine.
> - **Phase 5 (Completed)**: Offline Evaluation Framework (Precision@K, Recall@K, NDCG@K) using a deterministic held-out preference protocol.
> - **Phase 6 (Completed)**: Item-Based Collaborative Filtering (`ItemBasedCollaborativeRecommender`) using genuine user interaction data from **MovieLens latest-small** and leakage-safe temporal evaluation.
> - **Phase 7 (Completed)**: Hybrid Movie Recommendation System (`HybridMovieRecommender`), candidate pool union ($N_{\text{cand}}=100$), Min-Max score normalization, title identity mapping alignment, and alpha ablation benchmarking.
> - **Phase 8 (Completed)**: Explainable Recommendation Analysis Layer (`RecommendationExplainer`), transparently decomposing model recommendations into factual content metadata overlaps, collaborative item-item rating contributions, and hybrid score weights with human-readable summary generation.
> - **Phase 9 (Completed)**: Production Recommendation Serving & API Layer (`src/api.py`), exposing Content, Personalized, Collaborative, Hybrid, and Explainable recommendations via lightweight REST endpoints (FastAPI, Pydantic, Uvicorn).
> - **Future Phases (Upcoming)**: Frontend UI (Streamlit).
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
         ▼                ▼                ▼  ◄── PHASE 8 & 9
[ Offline Evaluation ]  [ Explainer ]  [ FastAPI REST Serving Layer ]
(Precision, Recall, NDCG) (Evidence)   (/health, /recommend/*, /explain)
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
- **Title Identity Alignment**: Deterministically maps MovieLens normalized titles to TMDB clean titles (2,812 out of 9,742 movies mapped, 28.86% coverage) for content profile matching while maintaining MovieLens `movieId` as primary recommendation identity. Unmapped items remain available to CF without receiving fabricated content scores.

### 2. Alpha Ablation Benchmark Results ($K=10$, 50 Evaluated Users)

| Model | Alpha ($\alpha$) | Precision@10 | Recall@10 | NDCG@10 |
| :--- | :---: | :---: | :---: | :---: |
| **CF-only hybrid scoring** | 0.00 | **0.0800** | **0.0683** | **0.0954** |
| **CF-dominant hybrid** | 0.25 | 0.0620 | 0.0492 | 0.0729 |
| **Balanced hybrid** | 0.50 | 0.0620 | 0.0485 | 0.0617 |
| **Content-dominant hybrid** | 0.75 | 0.0160 | 0.0311 | 0.0236 |
| **Content-only hybrid scoring** | 1.00 | 0.0080 | 0.0228 | 0.0157 |

> [!NOTE]
> **Endpoint Baseline Clarification**: $\alpha=0.00$ (CF-only) and $\alpha=1.00$ (Content-only) weight candidates selected from the candidate union pool ($N_{\text{cand}}=100$). They are conceptually distinct from the raw standalone Phase 6 CF and Phase 4 Content baselines.

### 3. Empirical Performance & Strategic Limitations
- **Benchmark Conclusion**: On the current MovieLens temporal benchmark, pure item-based collaborative filtering achieved the strongest ranking performance among the tested configurations ($NDCG@10 = 0.0954$). The hybrid system provides a flexible fusion architecture, but the current benchmark does not demonstrate an accuracy improvement over pure CF.
- **Cold-Start Interpretation**: Content-based features can provide an item-side fallback for movies with available metadata but limited or missing collaborative interaction history. True new-user cold start remains unresolved because personalized content profiles require user preferences.

---

## 💡 Phase 8 — Explainable Recommendation Analysis Layer

### 1. Explainability Layer Design & Motivation
Phase 8 introduces `RecommendationExplainer` in `src/explanation.py`, answering *"Why was this movie recommended?"* with factual model evidence rather than generic black-box assertions or synthesized text.

- **Content Evidence Breakdown**: Extracts specific metadata overlaps between the recommended item and movies positively rated ($\ge 4.0$) in the user's historical profile:
  - Shared genres, shared keywords, shared director, shared cast members.
  - Non-zero overlapping TF-IDF terms extracted directly from preprocessed text vectors.
- **Collaborative Evidence Breakdown**: Decomposes collaborative predictions by inspecting the top contributing rated movies in the user's history:
  - Individual rating weight $r(u, m)$, item-item similarity $S(m, c)$, and calculated contribution product $S(m, c) \cdot r(u, m)$.
- **Hybrid Score Decomposition**: Breaks down raw and normalized candidate scores, fusion parameter $\alpha$, weighted component contributions $\alpha \cdot \text{norm\_content}$ vs $(1-\alpha) \cdot \text{norm\_cf}$, and identifies the dominant recommendation branch.
- **Human-Readable Summary Generation**: Produces clear, concise, deterministic explanation strings highlighting primary recommendation drivers.

> [!IMPORTANT]
> **Data Leakage Safeguard**: Explanations strictly inspect historical training ratings ($T \le \text{split}$). Future held-out evaluation ratings are NEVER accessed as explanation evidence.

---

## ⚡ Phase 9 — Production Recommendation Serving & API Layer

### 1. API Architecture & Design
Phase 9 exposes the recommendation system via a lightweight **FastAPI** REST service (`src/api.py`). It reuses existing Phase 1–8 recommendation engines and explanations without modifying underlying algorithms.

- **Singleton Model Lifecycle**: Datasets and recommendation models are initialized **ONCE** during application startup (`lifespan` context manager) to avoid per-request model refitting or $O(N^2)$ matrix recalculations.
- **Strict Pydantic Validation**: Validates request parameters (non-empty titles, integer ratings between 1 and 5, `top_n > 0`, `alpha` in $[0.0, 1.0]$, duplicate title rejection, and explicit boolean rejection for numeric fields).
- **Clean JSON Serialization**: Converts NumPy scalars/arrays to native Python types, preventing serialization crashes.
- **Structured Error Responses**: Replaces raw stack traces with structured JSON error objects (`{"error": "...", "detail": "..."}`) returning HTTP status codes 400, 404, or 422.

### 2. Available Endpoints

| Endpoint | Method | Description | Example Query / Body |
| :--- | :---: | :--- | :--- |
| `/health` | `GET` | Service readiness & dataset status | `GET /health` |
| `/recommend/content` | `GET` | Single-movie content recommendations | `GET /recommend/content?title=Avatar&top_n=5` |
| `/recommend/personalized` | `POST` | User profile content recommendations | `{"history": [{"title": "Avatar", "rating": 5}], "top_n": 5}` |
| `/recommend/hybrid` | `POST` | Hybrid content & CF recommendations | `{"user_id": 1, "alpha": 0.5, "top_n": 5}` |
| `/recommend/explain` | `POST` | Recommendation evidence breakdown & summary | `{"user_id": 1, "target_movie_id": 2918, "alpha": 0.5}` |

### 3. Launching the Local API Server

To start the Uvicorn web server locally:

```bash
python -m uvicorn src.api:app --reload
```

> [!IMPORTANT]
> **API Serving Disclaimer**: This REST API layer provides an in-process serving demonstration via FastAPI. It does **NOT** include persistent user databases, user authentication/JWT, Redis caching, microservices, Docker, or cloud deployment.

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
│   └── 09_api_serving_demo.ipynb
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

Clone the repository and install requirements:

```bash
git clone https://github.com/Dp8453/Personalized-movie-recommendation-system.git
cd Personalized-movie-recommendation-system
pip install -r requirements.txt
```

### 2. Download Optional MovieLens Dataset

To run Phase 6, 7, 8 & 9 collaborative, hybrid, explainable, and API recommendation serving on genuine user interaction data, download MovieLens latest-small:

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

Execute data loading, preprocessing, single-movie lookup, personalized user profile recommendation, evaluation, collaborative filtering, hybrid recommendation, explainable analysis, and API serving verification:

```bash
python run.py
```

### 4. Run Complete Unit Test Suite

Execute all 147 unit tests across Phases 1–9:

```bash
python -m unittest discover -s tests -v
```

### 5. Launch REST API Server

Start the FastAPI application with Uvicorn:

```bash
python -m uvicorn src.api:app --reload
```

### 6. Explore API Serving Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook notebooks/09_api_serving_demo.ipynb
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
- [x] **Phase 8**: Explainable Recommendation Analysis Layer (`RecommendationExplainer`) & Summary Generation
- [x] **Phase 9**: Production Recommendation Serving & API Layer (`src/api.py`)
- [ ] **Future Phases (Upcoming)**: Frontend UI (Streamlit)
