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

The two datasets are merged on the `title` attribute, resulting in 4,809 merged movie rows.

---

## 3. Important Movie Fields

From the dataset's 23 merged columns, the key metadata fields identified for recommendation are:

| Field Name | Description | Importance for Recommendation |
| :--- | :--- | :--- |
| `title` | Official title of the movie | Target identifier for querying recommendations |
| `overview` | Textual summary/synopsis of the movie plot | Provides core semantic text representation |
| `genres` | JSON string of genre names (e.g., Action, Sci-Fi) | Crucial categorical grouping for filtering/similarity |
| `keywords` | JSON string of plot keywords/themes | High-precision thematic matching |
| `cast` | JSON string of top actors | Matches user preferences for specific lead actors |
| `crew` | JSON string of crew members (Director, etc.) | Director matching provides strong stylistic affinity |

---

## 4. General Recommendation Approach

The general approach used in the reference implementation follows these steps:
1. **Data Ingestion & Merging**: Load `movies.csv` and `credits.csv`, and inner-join on `title`.
2. **Feature Extraction**: Parse JSON-formatted strings (`ast.literal_eval`) for `genres`, `keywords`, top 3 items of `cast`, and director from `crew`.
3. **Text Normalization**: Strip spaces from multi-word tokens (e.g., `"Johnny Depp"` becomes `"JohnnyDepp"`) to prevent tag splitting.
4. **Tag Combination**: Concatenate `overview` words, `genres`, `keywords`, `cast`, and `crew` into a single unified `tags` string for each movie.
5. **Text Vectorization**: Convert `tags` text into numerical feature vectors using `CountVectorizer` or `TfidfVectorizer`.
6. **Similarity Calculation**: Compute pairwise Cosine Similarity matrix across all movie vectors.
7. **Query & Recommendation**: For a given movie title, retrieve its index, lookup top-5/10 highest cosine similarity scores, and return the corresponding movie titles.

---

## 5. What Ideas We Will Use

- **Dataset Selection**: Utilize the TMDB 5000 Movie & Credits dataset due to its rich metadata fields.
- **Metadata Feature Selection**: Extract `overview`, `genres`, `keywords`, top actors from `cast`, and director from `crew`.
- **Space Stripping Strategy**: Convert multi-word entity names (e.g., `"Sam Worthington"`) into single tokens (`"SamWorthington"`) so TF-IDF treats full names as unified concepts.

---

## 6. What We Will Implement Ourselves

- **Clean Modular Architecture**: Modularize code into a clean `src/` package rather than relying on monolithic script files.
- **Robust Error Handling**: Provide clear exception handling in data loading (`src/data_loader.py`) for missing files and path resolution.
- **Systematic EDA & Analysis**: Create an interactive, reproducible Jupyter Notebook (`notebooks/01_data_exploration.ipynb`) with comprehensive statistics, null analysis, and feature evaluation.
- **Automated Testing**: Include unit tests (`tests/test_data_loader.py`) and a verification runner (`run.py`).
- **Phased Engineering**: Build step-by-step with clear interview-ready explanations for each NLP and ML choice (TF-IDF vs Bag of Words, Cosine Similarity vs Euclidean Distance, Personalization, and Evaluation metrics in future phases).
