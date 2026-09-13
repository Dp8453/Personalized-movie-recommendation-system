# Personalized Movie Recommendation System

A content-based movie recommendation engine built using movie metadata, Natural Language Processing (NLP) techniques, vectorization, user preference modeling, similarity ranking, and offline evaluation metrics.

> [!IMPORTANT]
> **CURRENT PROJECT STATUS: PHASE 5 COMPLETED**
> - **Phase 1 (Completed)**: Project Foundation, Data Ingestion, ID-based Merging (`movies.id == credits.movie_id`), and Initial EDA.
> - **Phase 2 (Completed)**: Data Preprocessing, JSON Feature Extraction, Entity Space Collapsing, Overview Imputation, and Unified `tags` Construction.
> - **Phase 3 (Completed)**: TF-IDF Vectorization (`max_features=5000`, `stop_words='english'`), Cosine Similarity Matrix Modeling, Case-Insensitive Title Lookup, and Content-Based Recommendation Engine.
> - **Phase 4 (Completed)**: User Rating Preference Modeling, Weighted User Profile Construction ($\mathbf{u} = \frac{\sum w_i \mathbf{v}_i}{\sum |w_i|}$), Rated Movie Exclusion, and Content-Based Personalization Engine.
> - **Phase 5 (Completed)**: Offline Evaluation Framework (Precision@K, Recall@K, NDCG@K) using a deterministic held-out preference protocol.
> - **Future Phases (Upcoming)**: Web API Service (FastAPI) and Frontend UI (Streamlit).
>
> **CRITICAL DATASET LIMITATION DISCLAIMER**:
> The TMDB 5000 Movie Dataset contains rich metadata (genres, keywords, cast, crew, overviews) but does **NOT** contain genuine multi-user rating histories or real user interaction logs. Therefore, the offline evaluation framework uses a **deterministic held-out preference protocol** based on thematic user scenarios. The resulting metric scores evaluate the recommendation engine's ability to retrieve held-out related items under controlled conditions and must **NOT** be interpreted as real-world production user performance.

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
       ▼  ◄── PHASE 5 (COMPLETED: Offline Evaluation Metrics in src/evaluator.py)
  [ Offline Evaluation Framework ]
       │  ├── Precision@K = (relevant in top-K) / K
       │  ├── Recall@K = (relevant in top-K) / |relevant|
       │  ├── NDCG@K = DCG@K / IDCG@K (Position Discounting)
       │  └── Data Leakage Check: set(user_history) & set(held_out_relevant) == empty
       │
       ▼  ◄── PHASE 6 (UPCOMING)
  [ Production Web Service (FastAPI) & Frontend UI (Streamlit) ]
```

---

## 📈 Phase 5 — Offline Recommendation Evaluation Framework

### 1. Evaluation Protocol & Ground-Truth Selection Rules
Because TMDB 5000 lacks real user logs, we evaluate recommendation quality using a **held-out preference protocol**:
1. **User History Construction**: Define user rating profiles containing positive preferences (ratings 4–5) and negative preferences (ratings 1–2).
2. **Held-Out Ground Truth**: Select target relevant movies based on explicit, reproducible rules (e.g. franchise sequels/prequels or shared director/universe connections absent from input history).
3. **Data Leakage Safeguard**: Programmatically enforce that `set(user_history_titles) & set(held_out_relevant_titles) == empty set`.
4. **Metric Calculation**: Compare generated recommendations against ground truth using Precision@K, Recall@K, and NDCG@K.

### 2. Metric Definitions & Formulas

- **Precision@K**:
  $$\text{Precision}@K = \frac{|\text{Top-}K \text{ Recommendations} \cap \text{Relevant Items}|}{K}$$

- **Recall@K**:
  $$\text{Recall}@K = \frac{|\text{Top-}K \text{ Recommendations} \cap \text{Relevant Items}|}{|\text{Relevant Items}|}$$

- **NDCG@K (Normalized Discounted Cumulative Gain)**:
  $$\text{DCG}@K = \sum_{i=1}^{\min(K, |rec|)} \frac{2^{rel_i} - 1}{\log_2(i + 1)}, \quad \text{IDCG}@K = \sum_{i=1}^{\min(K, |relevant|)} \frac{1}{\log_2(i + 1)}$$
  $$\text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$

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
│   ├── 04_user_personalization.ipynb
│   └── 05_model_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── recommender.py
│   ├── personalizer.py
│   └── evaluator.py
│
├── docs/
│   └── reference-analysis.md
│
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_preprocessor.py
│   ├── test_recommender.py
│   ├── test_personalizer.py
│   └── test_evaluator.py
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

Execute data loading, preprocessing, single-movie lookup, personalized user profile recommendation, and evaluation:

```bash
python run.py
```

### 3. Run Complete Unit Test Suite

Execute all 59 unit tests across Phases 1–5:

```bash
python -m unittest discover -s tests -v
```

### 4. Explore Model Evaluation Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook notebooks/05_model_evaluation.ipynb
```

---

## 📌 Implementation Roadmap

- [x] **Phase 1**: Project Foundation, Dataset Setup (ID-based merge), Modular Data Loader & Initial EDA
- [x] **Phase 2**: Data Preprocessing, JSON Feature Extraction, Entity Space Collapsing & Tags Construction
- [x] **Phase 3**: Vectorization (TF-IDF), Similarity Computation & Content-Based Recommendation Engine
- [x] **Phase 4**: User Preference Rating Model, Weighted User Profile Vector Construction & Content Personalization
- [x] **Phase 5**: Offline Model Evaluation (Precision@K, Recall@K, NDCG@K)
- [ ] **Phase 6**: Web API Deployment (FastAPI) & Frontend UI (Streamlit)
