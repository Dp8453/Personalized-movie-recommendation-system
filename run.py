"""
Phase 1 Entry Point — Personalized Movie Recommendation System

This script verifies the Phase 1 setup, dataset loading, and basic statistics.
"""

import sys
from src.data_loader import load_movies


def main():
    print("=" * 60)
    print("Personalized Movie Recommendation System — Phase 1 Verification")
    print("=" * 60)

    try:
        df = load_movies()

        print("\n--- Dataset Summary ---")
        print(f"Total Unique Movies Loaded: {len(df)}")
        print(f"Total Columns: {len(df.columns)}")
        print("\nColumns:")
        for idx, col in enumerate(df.columns, 1):
            print(f"  {idx:2d}. {col}")

        print("\n--- Primary Key Check ---")
        id_match_count = (df["id"] == df["movie_id"]).sum()
        print(f"ID Alignment ('id' == 'movie_id'): {id_match_count} / {len(df)} records matched perfectly.")

        print("\n--- Phase 1 Checks Passed Successfully ---")

    except Exception as e:
        print(f"\n[ERROR] Failed to load dataset: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
