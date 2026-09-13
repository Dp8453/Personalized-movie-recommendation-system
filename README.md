# Personalized Movie Recommendation System

A content-based movie recommendation engine built using movie metadata, Natural Language Processing (NLP) techniques, vectorization, and similarity modeling.

> [!IMPORTANT]
> **CURRENT PROJECT STATUS: PHASE 2 COMPLETED**
> - **Phase 1 (Completed)**: Project Foundation, Data Ingestion, ID-based Merging (`movies.id == credits.movie_id`), and Initial EDA.
> - **Phase 2 (Completed)**: Data Preprocessing, JSON Feature Extraction, Entity Space Collapsing, Overview Imputation, and Unified `tags` Construction.
> - **Future Phases (Upcoming)**: TF-IDF Vectorization, Cosine Similarity, Recommendation Engine, Personalization, Evaluation Metrics, FastAPI, and Streamlit.

---

## 🚀 End-to-End Project Architecture & Pipeline

```
  [ Dataset Ingestion & ID Merge ] (4,803 Unique Movies)
       │
       ▼  ◄── PHASE 1 (COMPLETED)
  [ Data Preprocessing & Feature Engineering ]
       │  ├── JSON Parsing (genres, keywords, cast, crew)
       │  ├── Top 3 Lead Cast Extraction
       │  ├── Director Extraction (crew job == 'Director')
       │  ├── Entity Space Collapsing ("Sam Worthington" -> "SamWorthington")
       │  ├── Plot Overview Null Imputation (NaN -> "")
       │  └── Unified Tags Construction (overview + genres + keywords + cast + director)
       │
       ▼  ◄── PHASE 2 (COMPLETED: Exported to data/processed/clean_movies.csv)
  [ Text Vectorization (TF-IDF) ]
       │
       ▼  ◄── PHASE 3 (UPCOMING)
  [ Similarity Modeling (Cosine Similarity) ]
       │
       ▼
  [ Content-Based Recommendation Engine ]
       │
       ▼
  [ Personalization & Ranking ]
       │
       ▼
  [ Offline Evaluation (Precision@K, Recall@K, NDCG@K) ]
       │
       ▼
  [ Production Web Service (FastAPI) ]
       │
       ▼
  [ User Interface (Streamlit) ]
```

---

## 📊 Phase 2 Feature Engineering Summary

- **Processed Output**: `data/processed/clean_movies.csv` (4,803 rows × 8 columns, Git-ignored).
- **Metadata Extraction Rules**:
  - `genres`: JSON list extracted into genre string tokens.
  - `keywords`: JSON list extracted into thematic keyword tokens.
  - `cast`: Top 3 lead actors extracted to restrict high-dimensional feature noise.
  - `director`: Extracted specifically from crew list where `job == 'Director'`.
  - `overview`: Null/NaN values replaced with empty strings (`""`) to prevent `"nan"` literal text contamination.
- **Entity Space Collapsing**: Multi-word names are collapsed into unified tokens (e.g. `"Sam Worthington"` -> `"SamWorthington"`) so TF-IDF treats full names as distinct single entities.
- **Unified `tags` Feature**: Combines normalized `overview` words + `genres` + `keywords` + `cast` + `director` into a single space-separated text string per movie.

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
│   └── 02_data_preprocessing.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   └── preprocessor.py
│
├── docs/
│   └── reference-analysis.md
│
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   └── test_preprocessor.py
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

Execute data loading and Phase 2 preprocessing:

```bash
python run.py
```

### 3. Run Unit Test Suite

Execute all 13 unit tests:

```bash
python -m unittest discover -s tests -v
```

### 4. Explore Notebooks

Launch Jupyter Notebook:

```bash
jupyter notebook notebooks/02_data_preprocessing.ipynb
```

---

## 📌 Implementation Roadmap

- [x] **Phase 1**: Project Foundation, Dataset Setup (ID-based merge), Modular Data Loader & Initial EDA
- [x] **Phase 2**: Data Preprocessing, JSON Feature Extraction, Entity Space Collapsing & Tags Construction
- [ ] **Phase 3**: Vectorization (TF-IDF), Similarity Computation & Content-Based Recommendation Engine
- [ ] **Phase 4**: User Personalization & Hybrid Ranking Logic
- [ ] **Phase 5**: Model Evaluation (Precision@K, Recall@K, NDCG@K)
- [ ] **Phase 6**: Web API Deployment (FastAPI) & Frontend UI (Streamlit)
