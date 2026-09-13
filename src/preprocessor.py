import ast
import os
import sys
from pathlib import Path
import pandas as pd


def safe_eval_json(val: str | float | None) -> list:
    """
    Safely parses JSON-like string metadata using ast.literal_eval.

    Handles NaN, None, empty strings, and malformed strings gracefully
    by returning an empty list instead of crashing.
    """
    if pd.isna(val) or val is None or not isinstance(val, str) or not val.strip():
        return []
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return parsed
        return []
    except (ValueError, SyntaxError):
        return []


def extract_genres(genres_str: str | float | None) -> list[str]:
    """Extracts a list of genre names from a JSON-like string."""
    items = safe_eval_json(genres_str)
    return [item["name"] for item in items if isinstance(item, dict) and "name" in item]


def extract_keywords(keywords_str: str | float | None) -> list[str]:
    """Extracts a list of keyword names from a JSON-like string."""
    items = safe_eval_json(keywords_str)
    return [item["name"] for item in items if isinstance(item, dict) and "name" in item]


def extract_top_cast(cast_str: str | float | None, top_n: int = 3) -> list[str]:
    """
    Extracts the top N cast member names from a JSON-like string.
    Top 3 lead actors are selected to prevent high-dimensional noise.
    """
    items = safe_eval_json(cast_str)
    names = []
    for item in items[:top_n]:
        if isinstance(item, dict) and "name" in item:
            names.append(item["name"])
    return names


def extract_director(crew_str: str | float | None) -> list[str]:
    """
    Extracts the Director's name from a crew JSON-like string.
    Returns a list with 1 element if found, or an empty list if not found.
    """
    items = safe_eval_json(crew_str)
    for item in items:
        if isinstance(item, dict) and item.get("job") == "Director" and "name" in item:
            return [item["name"]]
    return []


def collapse_spaces(names_list: list[str]) -> list[str]:
    """
    Collapses spaces within multi-word entity names (e.g. 'Johnny Depp' -> 'JohnnyDepp').

    Rationale:
    Joining first and last names prevents vectorizers (TF-IDF / CountVectorizer)
    from treating 'Johnny' and 'Depp' as independent tokens, ensuring full entity integrity.
    """
    return [name.replace(" ", "") for name in names_list]


def clean_overview(overview_val: str | float | None) -> list[str]:
    """
    Cleans the plot overview string.
    Handles NaN/missing values by returning an empty list of word tokens.
    """
    if pd.isna(overview_val) or overview_val is None or not isinstance(overview_val, str):
        return []
    overview_str = overview_val.strip()
    if not overview_str or overview_str.lower() in ("nan", "none", "null"):
        return []
    return overview_str.split()


def build_tags(
    overview_tokens: list[str],
    genre_tokens: list[str],
    keyword_tokens: list[str],
    cast_tokens: list[str],
    director_tokens: list[str],
) -> str:
    """
    Combines overview, genres, keywords, top cast, and director tokens into a single
    normalized, lower-cased space-separated tags string.
    """
    # Combine all token lists
    combined = (
        overview_tokens
        + collapse_spaces(genre_tokens)
        + collapse_spaces(keyword_tokens)
        + collapse_spaces(cast_tokens)
        + collapse_spaces(director_tokens)
    )

    # Convert all tokens to lower-case and strip whitespace
    cleaned_tokens = [
        tok.lower().strip()
        for tok in combined
        if tok and tok.lower() not in ("nan", "none", "null")
    ]

    return " ".join(cleaned_tokens)


def preprocess_data(df: pd.DataFrame, output_path: str | Path | None = None) -> pd.DataFrame:
    """
    Full Phase 2 preprocessing pipeline:
    1. Extract genres, keywords, top 3 cast, and director.
    2. Clean overview synopses.
    3. Construct unified 'tags' feature column.
    4. Save clean processed dataset to data/processed/clean_movies.csv (git-ignored).
    """
    print("Starting Phase 2 Feature Engineering & Preprocessing...", flush=True)
    processed_df = df.copy()

    # Extract metadata fields
    print("Extracting genres...", flush=True)
    processed_df["extracted_genres"] = processed_df["genres"].apply(extract_genres)

    print("Extracting keywords...", flush=True)
    processed_df["extracted_keywords"] = processed_df["keywords"].apply(extract_keywords)

    print("Extracting top 3 cast members...", flush=True)
    processed_df["extracted_cast"] = processed_df["cast"].apply(lambda x: extract_top_cast(x, top_n=3))

    print("Extracting director...", flush=True)
    processed_df["extracted_director"] = processed_df["crew"].apply(extract_director)

    print("Cleaning plot overviews...", flush=True)
    processed_df["cleaned_overview"] = processed_df["overview"].apply(clean_overview)

    print("Constructing 'tags' feature column...", flush=True)
    tags_list = [
        build_tags(ov, g, k, c, d)
        for ov, g, k, c, d in zip(
            processed_df["cleaned_overview"],
            processed_df["extracted_genres"],
            processed_df["extracted_keywords"],
            processed_df["extracted_cast"],
            processed_df["extracted_director"],
        )
    ]
    processed_df["tags"] = tags_list

    # Prepare final clean dataframe
    clean_cols = [
        "id",
        "title",
        "overview",
        "extracted_genres",
        "extracted_keywords",
        "extracted_cast",
        "extracted_director",
        "tags",
    ]

    # Rename columns for clarity in processed dataset
    clean_df = processed_df[clean_cols].rename(
        columns={
            "extracted_genres": "genres",
            "extracted_keywords": "keywords",
            "extracted_cast": "cast",
            "extracted_director": "director",
        }
    )

    # Determine default output path if not specified
    if output_path is None:
        project_root = Path(__file__).resolve().parent.parent
        output_dir = project_root / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "clean_movies.csv"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving processed dataset to: {output_path}", flush=True)
    clean_df.to_csv(output_path, index=False)
    print("Preprocessing completed successfully.", flush=True)

    return clean_df


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.data_loader import load_movies

    raw_df = load_movies()
    processed_df = preprocess_data(raw_df)
    print("\nPreview of Processed DataFrame:", flush=True)
    print(processed_df[["id", "title", "tags"]].head(), flush=True)
