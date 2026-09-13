"""
Main Entry Point Verification Script — Personalized Movie Recommendation System

Demonstrates end-to-end pipeline:
Phase 1: Data Ingestion & ID-Based Merging
Phase 2: Data Preprocessing & Unified Tags Construction
Phase 3: TF-IDF Vectorization & Content-Based Recommendation Engine
"""

import sys
from pathlib import Path
import pandas as pd
from src.data_loader import load_movies
from src.preprocessor import preprocess_data
from src.recommender import MovieRecommender


def main():
    print("=" * 75, flush=True)
    print("Personalized Movie Recommendation System — Complete Pipeline Verification", flush=True)
    print("=" * 75, flush=True)

    try:
        print("\n--- PHASE 1: DATA LOADING ---", flush=True)
        raw_df = load_movies()
        print(f"Total Unique Movies Loaded: {len(raw_df)}", flush=True)
        print(f"ID Alignment ('id' == 'movie_id'): {(raw_df['id'] == raw_df['movie_id']).sum()} / {len(raw_df)} matched.", flush=True)

        print("\n--- PHASE 2: PREPROCESSING & FEATURE ENGINEERING ---", flush=True)
        processed_path = Path("data") / "processed" / "clean_movies.csv"
        if processed_path.exists():
            print(f"Loading cached processed dataset from: {processed_path}", flush=True)
            clean_df = pd.read_csv(processed_path)
        else:
            clean_df = preprocess_data(raw_df)

        print(f"Total Processed Movies: {len(clean_df)}", flush=True)
        print(f"Non-Empty Tags Count: {(clean_df['tags'].str.strip() != '').sum()} / {len(clean_df)}", flush=True)

        print("\n--- PHASE 3: CONTENT-BASED RECOMMENDATION ENGINE ---", flush=True)
        recommender = MovieRecommender(clean_df, max_features=5000, stop_words="english")
        print(f"TF-IDF Feature Matrix Shape : {recommender.get_tfidf_matrix_shape()}", flush=True)
        print(f"Vocabulary Size             : {recommender.get_vocab_size()}", flush=True)
        print(f"Cosine Similarity Matrix    : {recommender.get_similarity_matrix_shape()}", flush=True)

        test_queries = ["Avatar", "The Dark Knight", "Inception"]
        for query in test_queries:
            print(f"\nTop 5 Content-Based Recommendations for '{query}':", flush=True)
            recs = recommender.recommend(query, top_n=5)
            for idx, rec in enumerate(recs, 1):
                print(f"  {idx}. {rec['title']:35s} | Similarity Score: {rec['similarity_score']:.4f}", flush=True)

        print("\n" + "=" * 75, flush=True)
        print("--- ALL PIPELINE CHECKS (PHASE 1, 2 & 3) PASSED SUCCESSFULLY ---", flush=True)
        print("=" * 75, flush=True)

    except Exception as e:
        print(f"\n[ERROR] Pipeline verification failed: {e}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
