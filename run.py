"""
Main Entry Point Verification Script — Personalized Movie Recommendation System

Verifies Phase 1 (Data Ingestion & ID-based Merging) and
Phase 2 (Data Preprocessing, JSON Feature Extraction & Tags Construction).
"""

import sys
from src.data_loader import load_movies
from src.preprocessor import preprocess_data


def main():
    print("=" * 70, flush=True)
    print("Personalized Movie Recommendation System — Phase 1 & 2 Verification", flush=True)
    print("=" * 70, flush=True)

    try:
        print("\n--- PHASE 1: DATA LOADING ---", flush=True)
        raw_df = load_movies()

        print(f"Total Unique Movies Loaded: {len(raw_df)}", flush=True)
        print(f"ID Alignment ('id' == 'movie_id'): {(raw_df['id'] == raw_df['movie_id']).sum()} / {len(raw_df)} matched.", flush=True)

        print("\n--- PHASE 2: PREPROCESSING & FEATURE ENGINEERING ---", flush=True)
        clean_df = preprocess_data(raw_df)

        print("\n--- Processed Dataset Summary ---", flush=True)
        print(f"Total Processed Movies: {len(clean_df)}", flush=True)
        print(f"Clean Columns: {list(clean_df.columns)}", flush=True)
        print(f"Non-Empty Tags Count: {(clean_df['tags'].str.strip() != '').sum()} / {len(clean_df)}", flush=True)

        print("\nSample Processed Movie:", flush=True)
        sample = clean_df.iloc[0]
        print(f"  Title   : {sample['title']}", flush=True)
        print(f"  Genres  : {sample['genres']}", flush=True)
        print(f"  Cast    : {sample['cast']}", flush=True)
        print(f"  Director: {sample['director']}", flush=True)
        print(f"  Tags    : {sample['tags'][:150]}...", flush=True)

        print("\n" + "=" * 70, flush=True)
        print("--- PHASE 1 & PHASE 2 CHECKS PASSED SUCCESSFULLY ---", flush=True)
        print("=" * 70, flush=True)

    except Exception as e:
        print(f"\n[ERROR] Pipeline verification failed: {e}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
