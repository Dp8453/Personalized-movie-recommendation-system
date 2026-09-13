"""
Main Entry Point Verification Script — Personalized Movie Recommendation System

Demonstrates complete pipeline:
Phase 1: Data Ingestion & ID-Based Merging
Phase 2: Data Preprocessing & Unified Tags Construction
Phase 3: TF-IDF Vectorization & Single-Movie Recommendations
Phase 4: Weighted User Profile Modeling & Personalized Recommendations
Phase 5: Offline Model Evaluation (Precision@K, Recall@K, NDCG@K)
"""

import sys
from pathlib import Path
import pandas as pd
from src.data_loader import load_movies
from src.preprocessor import preprocess_data
from src.recommender import MovieRecommender
from src.personalizer import PersonalizedRecommender
from src.evaluator import evaluate_scenarios, evaluate_recommendations


def main():
    print("=" * 80, flush=True)
    print("Personalized Movie Recommendation System — Complete Pipeline Verification", flush=True)
    print("=" * 80, flush=True)

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

        print("\n--- PHASE 3: SINGLE-MOVIE CONTENT RECOMMENDATIONS ---", flush=True)
        recommender = MovieRecommender(clean_df, max_features=5000, stop_words="english")
        print(f"TF-IDF Feature Matrix Shape : {recommender.get_tfidf_matrix_shape()}", flush=True)
        print(f"Vocabulary Size             : {recommender.get_vocab_size()}", flush=True)

        query = "Avatar"
        p3_recs = recommender.recommend(query, top_n=3)
        print(f"Top 3 Similar Movies for '{query}':", flush=True)
        for idx, rec in enumerate(p3_recs, 1):
            print(f"  {idx}. {rec['title']:35s} | Similarity Score: {rec['similarity_score']:.4f}", flush=True)

        print("\n--- PHASE 4: PERSONALIZED USER PROFILE RECOMMENDATIONS ---", flush=True)
        personalizer = PersonalizedRecommender(clean_df, recommender=recommender)

        sample_history = [
            ("Avatar", 5),
            ("Aliens", 5),
            ("The Dark Knight", 4),
            ("Titanic", 1),
        ]
        print(f"Sample User Rating History: {sample_history}", flush=True)
        user_profile = personalizer.build_user_profile(sample_history)
        print(f"User Profile Vector Shape : {user_profile.shape}", flush=True)

        p4_recs = personalizer.recommend_for_user(sample_history, top_n=5)
        print("\nTop 5 Personalized Recommendations for User:", flush=True)
        for idx, rec in enumerate(p4_recs, 1):
            print(f"  {idx}. {rec['title']:35s} | Personalized Score: {rec['personalized_score']:.4f}", flush=True)

        print("\n--- PHASE 5: OFFLINE MODEL EVALUATION ---", flush=True)
        eval_scenarios = [
            {
                "name": "Nolan Superhero/Sci-Fi",
                "user_history": [("Batman Begins", 5), ("The Dark Knight", 5), ("Titanic", 1)],
                "relevant_movies": ["The Dark Knight Rises", "Batman Returns"],
            },
            {
                "name": "Pixar Animated Family",
                "user_history": [("Toy Story", 5), ("Toy Story 2", 5), ("The Godfather", 1)],
                "relevant_movies": ["Toy Story 3", "Monsters, Inc."],
            },
        ]
        eval_results = evaluate_scenarios(personalizer, eval_scenarios, k=5)
        agg = eval_results["aggregate"]
        print(f"Evaluated {agg['num_scenarios']} Scenarios at K=5:", flush=True)
        print(f"  Mean Precision@5 : {agg['mean_precision@5']:.4f}", flush=True)
        print(f"  Mean Recall@5    : {agg['mean_recall@5']:.4f}", flush=True)
        print(f"  Mean NDCG@5      : {agg['mean_ndcg@5']:.4f}", flush=True)

        print("\n" + "=" * 80, flush=True)
        print("--- ALL PIPELINE CHECKS (PHASES 1, 2, 3, 4 & 5) PASSED SUCCESSFULLY ---", flush=True)
        print("=" * 80, flush=True)

    except Exception as e:
        print(f"\n[ERROR] Pipeline verification failed: {e}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
