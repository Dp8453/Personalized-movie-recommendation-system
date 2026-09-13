# Reference Repository Analysis

**Reference Repository**: [AbhayVerma5/Movie-Recommendation-](https://github.com/AbhayVerma5/Movie-Recommendation-)

---

## 1. Reference Project Overview

The reference repository implements a Content-Based Movie Recommendation System. It uses movie metadata to calculate similarity between movies and recommend top-K similar titles based on a given movie query.

---

## 2. Dataset Used

The reference project utilizes the **TMDB 5000 Movie Dataset**, which consists of two primary CSV files:
1. `tmdb_5000_movies.csv`: Contains 4,803 movie records with attributes such as budget, genres, homepage, id, keywords, original_language, original_title, overview, popularity, production_companies, production_countries, release_date, revenue, runtime, spoken_languages, status, tagline, title, vote_average, and vote_count.
2. `tmdb_5000_credits.csv`: Contains 4,803 records mapping `movie_id` and `title` to `cast` (JSON string of actors) and `crew` (JSON string of production team members).

---

## 3. Key Technical Correction — Dataset Merge Key Analysis

In the reference implementation, the datasets were inner-joined on `title`. However, empirical inspection of the raw datasets reveals why **ID-based merging (`id` == `movie_id`) is strictly required**:

- `tmdb_5000_movies.csv` contains 4,803 rows and 4,800 unique titles (3 duplicate title pairs for remakes/re-releases, e.g., *Batman* [1966 vs 1989], *The Host*, *Out of the Blue*).
- Merging on `title` produces 4,809 rows due to Cartesian product joins on duplicate titles, creating invalid cross-matched records.
- Both CSVs contain a reliable integer primary key (`id` in `movies.csv` and `movie_id` in `credits.csv`). Merging on `movies.id` == `credits.movie_id` yields **exactly 4,803 unique, perfectly aligned movie records** with zero duplicate row inflation.

---

## 4. Important Movie Fields

From the dataset's 23 merged columns, the key metadata fields identified for recommendation are:

| Field Name | Description | Importance for Recommendation |
| :--- | :--- | :--- |
| `id` / `movie_id` | Unique TMDB movie integer ID | Stable primary key for 1-to-1 merging and lookups |
| `title` | Official title of the movie | Target identifier for querying recommendations |
| `overview` | Textual summary/synopsis of the movie plot | Provides core semantic text representation |
| `genres` | JSON string of genre names (e.g., Action, Sci-Fi) | Crucial categorical grouping for filtering/similarity |
| `keywords` | JSON string of plot keywords/themes | High-precision thematic matching |
| `cast` | JSON string of top actors | Matches user preferences for specific lead actors |
| `crew` | JSON string of crew members (Director, etc.) | Director matching provides strong stylistic affinity |

---

## 5. General Recommendation Approach

The general approach used in the reference implementation follows these steps:
1. **Data Ingestion & Merging**: Load `movies.csv` and `credits.csv`, and inner-join on primary keys `id` == `movie_id`.
2. **Feature Extraction**: Parse JSON-formatted strings (`ast.literal_eval`) for `genres`, `keywords`, top 3 items of `cast`, and director from `crew`.
3. **Text Normalization**: Strip spaces from multi-word tokens (e.g., `"Johnny Depp"` becomes `"JohnnyDepp"`) to prevent tag splitting.
4. **Tag Combination**: Concatenate `overview` words, `genres`, `keywords`, `cast`, and `crew` into a single unified `tags` string for each movie.
5. **Text Vectorization**: Convert `tags` text into numerical feature vectors using `CountVectorizer` or `TfidfVectorizer`.
6. **Similarity Calculation**: Compute pairwise Cosine Similarity matrix across all movie vectors.
7. **Query & Recommendation**: For a given movie title, retrieve its index, lookup top-5/10 highest cosine similarity scores, and return the corresponding movie titles.

---

## 6. What Ideas We Will Use

- **Dataset Selection**: Utilize the TMDB 5000 Movie & Credits dataset due to its rich metadata fields.
- **Primary Key Merging**: Merge strictly on `id` == `movie_id` to maintain exact 1-to-1 movie alignment (4,803 records).
- **Metadata Feature Selection**: Extract `overview`, `genres`, `keywords`, top actors from `cast`, and director from `crew`.
- **Space Stripping Strategy**: Convert multi-word entity names (e.g., `"Sam Worthington"`) into single tokens (`"SamWorthington"`) so TF-IDF treats full names as unified concepts.

---

## 7. What We Will Implement Ourselves

- **Clean Modular Architecture**: Modularize code into a clean `src/` package rather than relying on monolithic script files.
- **Robust Error Handling**: Provide clear exception handling in data loading (`src/data_loader.py`) for missing files and path resolution.
- **Systematic EDA & Analysis**: Create an interactive, reproducible Jupyter Notebook (`notebooks/01_data_exploration.ipynb`) with comprehensive statistics, null analysis, and feature evaluation.
- **Automated Testing**: Include unit tests (`tests/test_data_loader.py`) and a verification runner (`run.py`).
- **Raw Data Git Tracking Policy**: Exclude 45.7MB raw CSV files from Git history (`.gitignore`) to keep the repository lightweight, while providing seamless automated dataset download tools.
- **Phased Engineering**: Build step-by-step with clear interview-ready explanations for each NLP and ML choice (TF-IDF vs Bag of Words, Cosine Similarity vs Euclidean Distance, Personalization, and Evaluation metrics in future phases).
