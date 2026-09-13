import os
from pathlib import Path
import pandas as pd


def get_project_root() -> Path:
    """Returns the root path of the project directory."""
    return Path(__file__).resolve().parent.parent


def load_movies(data_dir: str | Path | None = None) -> pd.DataFrame:
    """
    Loads and merges the TMDB 5000 movies and credits datasets.

    Parameters
    ----------
    data_dir : str, Path, or None, default=None
        Path to the directory containing raw CSV files.
        If None, defaults to '<project_root>/data/raw'.

    Returns
    -------
    pd.DataFrame
        Merged pandas DataFrame containing movie details and credit information.

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

    print("Merging datasets on 'title'...")
    merged_df = movies_df.merge(credits_df, on="title")
    print(f"Successfully loaded and merged {len(merged_df)} movie records.")

    return merged_df


if __name__ == "__main__":
    df = load_movies()
    print("Preview of merged dataset:")
    print(df.head(2))
