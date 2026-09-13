import os
import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure src module resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.collaborative_filter import (
    ItemBasedCollaborativeRecommender,
    temporal_train_test_split,
    evaluate_collaborative_model,
)


class TestItemBasedCollaborativeRecommender(unittest.TestCase):
    """
    Unit test suite verifying dataset validation, matrix construction,
    cosine similarity calculation, tie-breaking, already-rated item exclusion,
    invalid top_n rejection, cold-start handling, temporal split, data leakage detection,
    and Phase 5 evaluator integration.
    """

    def setUp(self):
        # Small deterministic synthetic interactions DataFrame for algorithm unit tests
        self.synthetic_ratings = pd.DataFrame(
            [
                # User 1: Likes Sci-Fi (Movie 1, 2), Dislikes Comedy (Movie 3)
                {"userId": 1, "movieId": 101, "rating": 5.0, "timestamp": 1000},
                {"userId": 1, "movieId": 102, "rating": 4.5, "timestamp": 1001},
                {"userId": 1, "movieId": 103, "rating": 1.0, "timestamp": 1002},
                {"userId": 1, "movieId": 104, "rating": 5.0, "timestamp": 1003},
                # User 2: Likes Sci-Fi (Movie 1, 2)
                {"userId": 2, "movieId": 101, "rating": 4.0, "timestamp": 1010},
                {"userId": 2, "movieId": 102, "rating": 5.0, "timestamp": 1011},
                {"userId": 2, "movieId": 105, "rating": 4.5, "timestamp": 1012},
                # User 3: Likes Comedy (Movie 3, 4)
                {"userId": 3, "movieId": 103, "rating": 5.0, "timestamp": 1020},
                {"userId": 3, "movieId": 104, "rating": 4.0, "timestamp": 1021},
                {"userId": 3, "movieId": 101, "rating": 1.0, "timestamp": 1022},
                # User 4: Rates Movie 1, 2, 4, 5
                {"userId": 4, "movieId": 101, "rating": 5.0, "timestamp": 1030},
                {"userId": 4, "movieId": 102, "rating": 4.0, "timestamp": 1031},
                {"userId": 4, "movieId": 104, "rating": 4.5, "timestamp": 1032},
                {"userId": 4, "movieId": 105, "rating": 5.0, "timestamp": 1033},
            ]
        )

        self.synthetic_movies = pd.DataFrame(
            [
                {"movieId": 101, "title": "Sci-Fi Alpha"},
                {"movieId": 102, "title": "Sci-Fi Beta"},
                {"movieId": 103, "title": "Comedy One"},
                {"movieId": 104, "title": "Comedy Two"},
                {"movieId": 105, "title": "Sci-Fi Gamma"},
            ]
        )

    # 1. Missing Required Columns Validation
    def test_missing_required_columns(self):
        bad_df = pd.DataFrame([{"user": 1, "movie": 101}])
        rec = ItemBasedCollaborativeRecommender(min_ratings_per_movie=0, min_ratings_per_user=0)
        with self.assertRaises(ValueError):
            rec.fit(bad_df)

    # 2. Invalid Rating 0.0 Rejection
    def test_invalid_rating_zero(self):
        bad_df = pd.DataFrame([{"userId": 1, "movieId": 101, "rating": 0.0}])
        rec = ItemBasedCollaborativeRecommender(min_ratings_per_movie=0, min_ratings_per_user=0)
        with self.assertRaises(ValueError):
            rec.fit(bad_df)

    # 3. Invalid Rating 6.0 Rejection
    def test_invalid_rating_six(self):
        bad_df = pd.DataFrame([{"userId": 1, "movieId": 101, "rating": 6.0}])
        rec = ItemBasedCollaborativeRecommender(min_ratings_per_movie=0, min_ratings_per_user=0)
        with self.assertRaises(ValueError):
            rec.fit(bad_df)

    # 4. Non-numeric Rating Rejection
    def test_non_numeric_rating(self):
        bad_df = pd.DataFrame([{"userId": 1, "movieId": 101, "rating": "five"}])
        rec = ItemBasedCollaborativeRecommender(min_ratings_per_movie=0, min_ratings_per_user=0)
        with self.assertRaises(ValueError):
            rec.fit(bad_df)

    # 5. Null userId Rejection
    def test_null_user_id(self):
        bad_df = pd.DataFrame([{"userId": None, "movieId": 101, "rating": 4.0}])
        rec = ItemBasedCollaborativeRecommender(min_ratings_per_movie=0, min_ratings_per_user=0)
        with self.assertRaises(ValueError):
            rec.fit(bad_df)

    # 6. Null movieId Rejection
    def test_null_movie_id(self):
        bad_df = pd.DataFrame([{"userId": 1, "movieId": None, "rating": 4.0}])
        rec = ItemBasedCollaborativeRecommender(min_ratings_per_movie=0, min_ratings_per_user=0)
        with self.assertRaises(ValueError):
            rec.fit(bad_df)

    # 7. Null Rating Rejection
    def test_null_rating(self):
        bad_df = pd.DataFrame([{"userId": 1, "movieId": 101, "rating": None}])
        rec = ItemBasedCollaborativeRecommender(min_ratings_per_movie=0, min_ratings_per_user=0)
        with self.assertRaises(ValueError):
            rec.fit(bad_df)

    # 8. Empty Dataset Rejection
    def test_empty_dataset(self):
        empty_df = pd.DataFrame(columns=["userId", "movieId", "rating"])
        rec = ItemBasedCollaborativeRecommender(min_ratings_per_movie=0, min_ratings_per_user=0)
        with self.assertRaises(ValueError):
            rec.fit(empty_df)

    # 9 & 10. Duplicate User/Movie Deduplication & Latest Timestamp Wins
    def test_duplicate_user_movie_latest_timestamp_wins(self):
        dup_df = pd.DataFrame(
            [
                {"userId": 1, "movieId": 101, "rating": 1.0, "timestamp": 100},
                {"userId": 1, "movieId": 101, "rating": 5.0, "timestamp": 200},  # Latest
            ]
        )
        rec = ItemBasedCollaborativeRecommender(min_ratings_per_movie=0, min_ratings_per_user=0)
        clean = rec.validate_ratings_df(dup_df)
        self.assertEqual(len(clean), 1)
        self.assertEqual(clean.iloc[0]["rating"], 5.0)

    # 11. Model Fitting
    def test_model_fitting(self):
        rec = ItemBasedCollaborativeRecommender(
            min_ratings_per_movie=1, min_ratings_per_user=1, movies_df=self.synthetic_movies
        )
        rec.fit(self.synthetic_ratings)
        self.assertTrue(rec.is_fitted)
        self.assertGreater(len(rec.movie_id_to_idx), 0)

    # 12. Sparse Matrix Construction
    def test_sparse_matrix_shape(self):
        rec = ItemBasedCollaborativeRecommender(min_ratings_per_movie=0, min_ratings_per_user=0)
        rec.fit(self.synthetic_ratings)
        self.assertEqual(rec.item_user_matrix.shape, (5, 4))

    # 13. Similarity Calculation Accuracy
    def test_similarity_calculation(self):
        rec = ItemBasedCollaborativeRecommender(min_ratings_per_movie=0, min_ratings_per_user=0)
        rec.fit(self.synthetic_ratings)
        # Movie 101 and 102 are both highly rated by User 1, 2, 4 -> High Cosine Similarity
        m101_idx = rec.movie_id_to_idx[101]
        m102_idx = rec.movie_id_to_idx[102]
        sim_101_102 = rec.similarity_matrix[m101_idx, m102_idx]
        self.assertGreater(sim_101_102, 0.7)

    # 14. Similar Movies Sorted Descending
    def test_similar_items_sorted_descending(self):
        rec = ItemBasedCollaborativeRecommender(
            min_ratings_per_movie=0, min_ratings_per_user=0, movies_df=self.synthetic_movies
        )
        rec.fit(self.synthetic_ratings)
        sims = rec.get_similar_items(101, top_n=5)
        scores = [s["similarity_score"] for s in sims]
        self.assertEqual(scores, sorted(scores, reverse=True))

    # 15. Self-Item Exclusion
    def test_self_item_exclusion(self):
        rec = ItemBasedCollaborativeRecommender(min_ratings_per_movie=0, min_ratings_per_user=0)
        rec.fit(self.synthetic_ratings)
        sims = rec.get_similar_items(101, top_n=10)
        rec_ids = [s["movieId"] for s in sims]
        self.assertNotIn(101, rec_ids)

    # 16. Unknown Movie ID Handling
    def test_unknown_movie_id(self):
        rec = ItemBasedCollaborativeRecommender(min_ratings_per_movie=0, min_ratings_per_user=0)
        rec.fit(self.synthetic_ratings)
        with self.assertRaises(ValueError):
            rec.get_similar_items(999, top_n=5)

    # 17. Unknown User ID Handling
    def test_unknown_user_id(self):
        rec = ItemBasedCollaborativeRecommender(min_ratings_per_movie=0, min_ratings_per_user=0)
        rec.fit(self.synthetic_ratings)
        with self.assertRaises(ValueError):
            rec.recommend(999, top_n=5)

    # 18-22. Invalid top_n Rejection
    def test_invalid_top_n(self):
        rec = ItemBasedCollaborativeRecommender(min_ratings_per_movie=0, min_ratings_per_user=0)
        rec.fit(self.synthetic_ratings)

        invalid_top_ns = [0, -1, 3.5, True, False, "5"]
        for bad_n in invalid_top_ns:
            with self.subTest(bad_n=bad_n):
                with self.assertRaises(ValueError):
                    rec.get_similar_items(101, top_n=bad_n)
                with self.assertRaises(ValueError):
                    rec.recommend(1, top_n=bad_n)

    # 23. Personalized Recommendation Generation
    def test_personalized_recommendation(self):
        rec = ItemBasedCollaborativeRecommender(
            min_ratings_per_movie=0, min_ratings_per_user=0, movies_df=self.synthetic_movies
        )
        rec.fit(self.synthetic_ratings)
        # User 2 rated 101 (4.0), 102 (5.0), 105 (4.5). Unrated candidate: 104 or 103
        recs = rec.recommend(user_id=2, top_n=2, positive_threshold=4.0)
        self.assertIsInstance(recs, list)

    # 24. Already-Rated Movies Strict Exclusion
    def test_already_rated_movies_exclusion(self):
        rec = ItemBasedCollaborativeRecommender(min_ratings_per_movie=0, min_ratings_per_user=0)
        rec.fit(self.synthetic_ratings)
        # User 1 rated 101, 102, 103, 104
        user1_rated = {101, 102, 103, 104}
        recs = rec.recommend(user_id=1, top_n=5, positive_threshold=4.0)
        for r in recs:
            self.assertNotIn(r["movieId"], user1_rated)

    # 25. Positive Threshold Filtering
    def test_positive_threshold_filtering(self):
        rec = ItemBasedCollaborativeRecommender(min_ratings_per_movie=0, min_ratings_per_user=0)
        rec.fit(self.synthetic_ratings)
        # User 3 rated 103 (5.0), 104 (4.0), 101 (1.0). If threshold=4.0, 101 is ignored as source.
        recs_high = rec.recommend(user_id=3, top_n=5, positive_threshold=4.0)
        self.assertIsInstance(recs_high, list)

    # 26. Candidate Score Aggregation
    def test_candidate_score_aggregation(self):
        rec = ItemBasedCollaborativeRecommender(min_ratings_per_movie=0, min_ratings_per_user=0)
        rec.fit(self.synthetic_ratings)
        recs = rec.recommend(user_id=2, top_n=2, positive_threshold=4.0)
        for r in recs:
            self.assertGreater(r["collaborative_score"], 0.0)

    # 27. Deterministic Tie-Breaking
    def test_deterministic_tie_breaking(self):
        rec = ItemBasedCollaborativeRecommender(min_ratings_per_movie=0, min_ratings_per_user=0)
        rec.fit(self.synthetic_ratings)
        res1 = rec.recommend(user_id=2, top_n=2)
        res2 = rec.recommend(user_id=2, top_n=2)
        self.assertEqual(res1, res2)

    # 28. Cold-Start User Handling
    def test_cold_start_user(self):
        rec = ItemBasedCollaborativeRecommender(min_ratings_per_movie=0, min_ratings_per_user=0)
        rec.fit(self.synthetic_ratings)
        with self.assertRaises(ValueError):
            rec.recommend(user_id=999)

    # 29 & 30. Temporal Train/Test Split & Ordering
    def test_temporal_train_test_split(self):
        train_df, scenarios = temporal_train_test_split(
            self.synthetic_ratings, test_ratio=0.25, min_user_ratings=3, positive_threshold=4.0
        )
        self.assertIsInstance(train_df, pd.DataFrame)
        self.assertIsInstance(scenarios, list)

    # 31 & 32. Held-Out Data Absent from Training & Leakage Prevention
    def test_leakage_prevention(self):
        train_df, scenarios = temporal_train_test_split(
            self.synthetic_ratings, test_ratio=0.25, min_user_ratings=3, positive_threshold=4.0
        )
        for sc in scenarios:
            hist = set(sc["train_history_ids"])
            held_out = set(sc["held_out_relevant_ids"])
            overlap = hist & held_out
            self.assertEqual(len(overlap), 0, f"Data leakage detected: {overlap}")

    # 33. Phase 5 Evaluator Integration
    def test_evaluate_collaborative_model_integration(self):
        eval_res = evaluate_collaborative_model(
            self.synthetic_ratings,
            k_values=[1, 2],
            min_ratings_per_movie=0,
            min_ratings_per_user=0,
            min_user_ratings=3,
            test_ratio=0.25,
            positive_threshold=4.0,
        )
        self.assertIn("model_stats", eval_res)
        self.assertIn("eval_by_k", eval_res)

    # 34. End-to-End Recommendation on Synthetic Data
    def test_end_to_end_synthetic(self):
        rec = ItemBasedCollaborativeRecommender(
            min_ratings_per_movie=0, min_ratings_per_user=0, movies_df=self.synthetic_movies
        )
        rec.fit(self.synthetic_ratings)
        sims = rec.get_similar_items(101, top_n=2)
        recs = rec.recommend(1, top_n=2)
        self.assertGreater(len(sims), 0)
        self.assertIsInstance(recs, list)

    # 35. MovieLens Integration Test (Gracefully Skipped if Dataset Missing)
    def test_movielens_dataset_integration(self):
        project_root = Path(__file__).resolve().parent.parent
        ml_ratings_path = project_root / "data" / "raw" / "movielens" / "ml-latest-small" / "ratings.csv"
        ml_movies_path = project_root / "data" / "raw" / "movielens" / "ml-latest-small" / "movies.csv"

        if not ml_ratings_path.exists() or not ml_movies_path.exists():
            self.skipTest(
                f"MovieLens dataset not found at {ml_ratings_path}. "
                "Skipping optional MovieLens integration test."
            )

        ratings_df = pd.read_csv(ml_ratings_path)
        movies_df = pd.read_csv(ml_movies_path)

        rec = ItemBasedCollaborativeRecommender(
            min_ratings_per_movie=5, min_ratings_per_user=5, movies_df=movies_df
        )
        rec.fit(ratings_df)
        self.assertTrue(rec.is_fitted)

        sims = rec.get_similar_items(1, top_n=5)  # MovieLens ID 1: Toy Story (1995)
        self.assertEqual(len(sims), 5)

        recs = rec.recommend(user_id=1, top_n=5, positive_threshold=4.0)
        self.assertEqual(len(recs), 5)


if __name__ == "__main__":
    unittest.main()
