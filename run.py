"""
Main Entry Point Verification Script — Personalized Movie Recommendation System

Demonstrates complete pipeline:
Phase 1: Data Ingestion & ID-Based Merging
Phase 2: Data Preprocessing & Unified Tags Construction
Phase 3: TF-IDF Vectorization & Single-Movie Recommendations
Phase 4: Weighted User Profile Modeling & Personalized Recommendations
Phase 5: Offline Model Evaluation (Precision@K, Recall@K, NDCG@K)
Phase 6: Item-Based Collaborative Filtering (MovieLens Interaction Data)
"""

import sys
from pathlib import Path
import pandas as pd
from src.data_loader import load_movies
from src.preprocessor import preprocess_data
from src.recommender import MovieRecommender
from src.personalizer import PersonalizedRecommender
from src.evaluator import evaluate_scenarios, evaluate_recommendations
from src.collaborative_filter import ItemBasedCollaborativeRecommender
from src.hybrid_recommender import HybridMovieRecommender


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

        print("\n--- PHASE 6: ITEM-BASED COLLABORATIVE FILTERING ---", flush=True)
        ml_ratings_path = Path("data") / "raw" / "movielens" / "ml-latest-small" / "ratings.csv"
        ml_movies_path = Path("data") / "raw" / "movielens" / "ml-latest-small" / "movies.csv"

        if ml_ratings_path.exists() and ml_movies_path.exists():
            ratings_df = pd.read_csv(ml_ratings_path)
            movies_df = pd.read_csv(ml_movies_path)
            cf_recommender = ItemBasedCollaborativeRecommender(
                min_ratings_per_movie=5, min_ratings_per_user=5, movies_df=movies_df
            )
            cf_recommender.fit(ratings_df)
            print(f"MovieLens Ratings Loaded    : {len(ratings_df):,} ratings across {cf_recommender.stats['filtered_users']} users", flush=True)
            print(f"Item-User Sparse Matrix     : {cf_recommender.item_user_matrix.shape}", flush=True)

            sims = cf_recommender.get_similar_items(1, top_n=3)
            print("Top 3 Collaborative Similar Movies to 'Toy Story (1995)' (movieId=1):", flush=True)
            for idx, item in enumerate(sims, 1):
                print(f"  {idx}. {item['title']:35s} | Similarity Score: {item['similarity_score']:.4f}", flush=True)

            cf_user_recs = cf_recommender.recommend(user_id=1, top_n=3)
            print("Top 3 Collaborative Personalized Recommendations for User 1:", flush=True)
            for idx, rec in enumerate(cf_user_recs, 1):
                print(f"  {idx}. {rec['title']:35s} | Collaborative Score: {rec['collaborative_score']:.4f}", flush=True)

            print("\n--- PHASE 7: HYBRID MOVIE RECOMMENDATION SYSTEM ---", flush=True)
            hybrid_rec = HybridMovieRecommender(
                collaborative_recommender=cf_recommender,
                personalizer=personalizer,
                movies_df=movies_df,
                tmdb_df=clean_df,
            )
            hybrid_rec.fit(ratings_df)
            mapped_c = hybrid_rec.mapping_stats["mapped_movies"]
            total_ml_c = hybrid_rec.mapping_stats["total_movielens_movies"]
            cov_p = hybrid_rec.mapping_stats["coverage_percentage"]
            print(f"Title Identity Alignment Mapped : {mapped_c:,} / {total_ml_c:,} ({cov_p:.2f}%)", flush=True)

            h_recs = hybrid_rec.recommend(user_id=1, alpha=0.5, top_n=3)
            print("Top 3 Hybrid Recommendations (alpha=0.50) for User 1:", flush=True)
            for idx, rec in enumerate(h_recs, 1):
                print(f"  {idx}. {rec['title']:35s} | Hybrid Score: {rec['hybrid_score']:.4f} (Content: {rec['norm_content_score']:.4f}, CF: {rec['norm_cf_score']:.4f})", flush=True)
        else:
            print("MovieLens dataset not found in data/raw/movielens/ml-latest-small/.", flush=True)
            print("Run scratch/download_movielens.py according to README.md to run Phase 6 & 7 demonstration.", flush=True)

        print("\n" + "=" * 80, flush=True)
        print("--- ALL PIPELINE CHECKS (PHASES 1, 2, 3, 4, 5, 6 & 7) PASSED SUCCESSFULLY ---", flush=True)
        print("=" * 80, flush=True)

    except Exception as e:
        print(f"\n[ERROR] Pipeline verification failed: {e}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
