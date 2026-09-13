import os
from pathlib import Path
import pandas as pd


def get_project_root() -> Path:
    """Returns the root path of the project directory."""
    return Path(__file__).resolve().parent.parent


def load_movies(data_dir: str | Path | None = None) -> pd.DataFrame:
    """
    Loads and merges the TMDB 5000 movies and credits datasets using primary key IDs.

    Merge Strategy:
    - Primary key in tmdb_5000_movies.csv: 'id'
    - Primary key in tmdb_5000_credits.csv: 'movie_id'
    - Merging on 'id' == 'movie_id' guarantees 1-to-1 matching (4,803 unique movies).
    - Avoids duplicate row creation caused by title collisions (e.g. remakes like Batman).

    Parameters
    ----------
    data_dir : str, Path, or None, default=None
        Path to the directory containing raw CSV files.
        If None, defaults to '<project_root>/data/raw'.

    Returns
    -------
    pd.DataFrame
        Merged pandas DataFrame containing 4,803 movie records across 23 columns.

    Raises
    ------
    FileNotFoundError
        If either tmdb_5000_movies.csv or tmdb_5000_credits.csv is missing.
    """
    if data_dir is None:
        data_dir = get_project_root() / "data" / "raw"
    else:
        data_dir = Path(data_dir)

    movies_path = data_dir / "tmdb_5000_movies.csv"
    credits_path = data_dir / "tmdb_5000_credits.csv"

    if not movies_path.exists():
        raise FileNotFoundError(
            f"Movies dataset not found at '{movies_path}'. "
            "Please ensure tmdb_5000_movies.csv is placed in the data/raw/ directory."
        )

    if not credits_path.exists():
        raise FileNotFoundError(
            f"Credits dataset not found at '{credits_path}'. "
            "Please ensure tmdb_5000_credits.csv is placed in the data/raw/ directory."
        )

    print(f"Loading movies dataset from: {movies_path}")
    movies_df = pd.read_csv(movies_path)

    print(f"Loading credits dataset from: {credits_path}")
    credits_df = pd.read_csv(credits_path)

    print("Merging datasets on primary key ('id' == 'movie_id')...")
    # Merge using primary keys 'id' and 'movie_id'
    merged_df = movies_df.merge(
        credits_df, left_on="id", right_on="movie_id", suffixes=("", "_credits")
    )

    # Remove redundant title_credits column if present
    if "title_credits" in merged_df.columns:
        merged_df = merged_df.drop(columns=["title_credits"])

    print(f"Successfully loaded and merged {len(merged_df)} unique movie records.")

    return merged_df


if __name__ == "__main__":
    df = load_movies()
    print("Preview of merged dataset:")
    print(df[["id", "movie_id", "title"]].head())
