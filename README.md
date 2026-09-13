# Personalized Movie Recommendation System

A content-based movie recommendation engine built using movie metadata, Natural Language Processing (NLP) techniques, vectorization, and similarity modeling.

> [!IMPORTANT]
> **CURRENT PROJECT STATUS: PHASE 1 COMPLETED**
> Only Phase 1 (Project Foundation, Dataset Ingestion, and Initial Exploratory Data Analysis) is currently implemented. Later phases (Data Cleaning, TF-IDF, Cosine Similarity, Recommendation Engine, Personalization, Evaluation Metrics, FastAPI, and Streamlit) are planned and will be built step-by-step.

---

## 🚀 End-to-End Project Architecture & Pipeline

```
  [ Dataset ] (TMDB 5000 Movies & Credits)
       │
       ▼  ◄── PHASE 1 (COMPLETED)
  [ Data Cleaning & Validation ]
       │
       ▼  ◄── PHASE 2 (UPCOMING)
  [ Feature Engineering & Metadata Extraction ]
       │
       ▼
  [ Text Preprocessing & Tag Creation ]
       │
       ▼
  [ Vectorization (TF-IDF / Bag of Words) ]
       │
       ▼
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

## 📊 Phase 1 Implementation Summary

- **Dataset Source**: [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) (`tmdb_5000_movies.csv` & `tmdb_5000_credits.csv`).
- **Merged Dataset Size**: 4,809 movie records across 23 columns.
- **Key Recommendation Features Identified**: `title`, `overview`, `genres`, `keywords`, `cast`, `crew` (Director).
- **Modules Created**:
  - `src/data_loader.py`: Modular dataset loader with relative path resolution and error handling.
  - `notebooks/01_data_exploration.ipynb`: Comprehensive EDA notebook answering 7 key dataset analysis questions.
  - `docs/reference-analysis.md`: Structural and architectural analysis of reference recommendation implementation.
  - `run.py` & `tests/test_data_loader.py`: Phase 1 entry point and automated unit tests.

---

## 📁 Repository Structure

```
Personalized-movie-recommendation-system/
│
├── data/
│   ├── raw/
│   │   ├── tmdb_5000_movies.csv
│   │   └── tmdb_5000_credits.csv
│   └── processed/
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── src/
│   ├── __init__.py
│   └── data_loader.py
│
├── docs/
│   └── reference-analysis.md
│
├── tests/
│   ├── __init__.py
│   └── test_data_loader.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── run.py
```

---

## 🛠️ Quick Start Guide

### 1. Prerequisites & Installation

Clone the repository and install the Phase 1 dependencies:

```bash
git clone https://github.com/Dp8453/Personalized-movie-recommendation-system.git
cd Personalized-movie-recommendation-system
pip install -r requirements.txt
```

### 2. Verify Data Loader & Execution

Run the Phase 1 entry point script to verify dataset loading and summary statistics:

```bash
python run.py
```

### 3. Run Unit Tests

Execute the unit test suite:

```bash
python tests/test_data_loader.py
```

### 4. Explore EDA Notebook

Launch Jupyter Notebook to inspect the initial EDA:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

---

## 📌 Implementation Roadmap

- [x] **Phase 1**: Project Foundation, Dataset Setup, Modular Data Loader & Initial EDA
- [ ] **Phase 2**: Data Preprocessing, JSON Feature Extraction, Text Normalization & Tag Combination
- [ ] **Phase 3**: Vectorization (TF-IDF), Similarity Computation & Recommendation Engine
- [ ] **Phase 4**: User Personalization & Hybrid Ranking Logic
- [ ] **Phase 5**: Model Evaluation (Precision@K, Recall@K, NDCG@K)
- [ ] **Phase 6**: Web API Deployment (FastAPI) & Frontend UI (Streamlit)
