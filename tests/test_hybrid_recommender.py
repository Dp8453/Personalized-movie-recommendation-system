import os
import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure src module resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.collaborative_filter import ItemBasedCollaborativeRecommender
from src.personalizer import PersonalizedRecommender
from src.hybrid_recommender import (
    HybridMovieRecommender,
    validate_alpha,
    normalize_scores,
    evaluate_hybrid_model,
)


class TestHybridMovieRecommender(unittest.TestCase):
    """
    Unit test suite verifying alpha validation, score normalization, candidate pool union,
    pure CF (alpha=0), pure content (alpha=1), already-rated exclusion, cold-start handling,
    deterministic tie-breaking, data leakage prevention, and Phase 5 evaluator integration.
    """

    def setUp(self):
        # Small deterministic synthetic ratings DataFrame
        self.synthetic_ratings = pd.DataFrame(
            [
                {"userId": 1, "movieId": 101, "rating": 5.0, "timestamp": 1000},
                {"userId": 1, "movieId": 102, "rating": 4.5, "timestamp": 1001},
                {"userId": 1, "movieId": 103, "rating": 1.0, "timestamp": 1002},
                {"userId": 1, "movieId": 104, "rating": 5.0, "timestamp": 1003},
                {"userId": 2, "movieId": 101, "rating": 4.0, "timestamp": 1010},
                {"userId": 2, "movieId": 102, "rating": 5.0, "timestamp": 1011},
                {"userId": 2, "movieId": 105, "rating": 4.5, "timestamp": 1012},
                {"userId": 3, "movieId": 103, "rating": 5.0, "timestamp": 1020},
                {"userId": 3, "movieId": 104, "rating": 4.0, "timestamp": 1021},
                {"userId": 3, "movieId": 101, "rating": 1.0, "timestamp": 1022},
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

        self.cf_rec = ItemBasedCollaborativeRecommender(
            min_ratings_per_movie=0, min_ratings_per_user=0, movies_df=self.synthetic_movies
        )
        self.cf_rec.fit(self.synthetic_ratings)

    # 1. Initialization and Composition
    def test_initialization_and_composition(self):
        hybrid = HybridMovieRecommender(
            collaborative_recommender=self.cf_rec, movies_df=self.synthetic_movies
        )
        hybrid.fit(self.synthetic_ratings)
        self.assertTrue(hybrid.is_fitted)
        self.assertEqual(hybrid.cf_recommender, self.cf_rec)

    # 2. Valid Alpha Acceptance
    def test_valid_alpha(self):
        for val in [0.0, 0.25, 0.5, 0.75, 1.0, 0, 1]:
            self.assertEqual(validate_alpha(val), float(val))

    # 3-9. Invalid Alpha Rejection
    def test_invalid_alpha(self):
        bad_alphas = [-0.1, 1.1, True, False, "0.5", None, float("nan"), float("inf"), float("-inf")]
        for bad in bad_alphas:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    validate_alpha(bad)

    # 10. Invalid top_n Validation
    def test_invalid_top_n(self):
        hybrid = HybridMovieRecommender(collaborative_recommender=self.cf_rec)
        hybrid.fit(self.synthetic_ratings)

        for bad in [0, -1, 3.5, True, False]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    hybrid.recommend(user_id=1, top_n=bad)

    # 11. Invalid candidate_size Validation
    def test_invalid_candidate_size(self):
        hybrid = HybridMovieRecommender(collaborative_recommender=self.cf_rec)
        hybrid.fit(self.synthetic_ratings)

        for bad in [0, -1, 3.5, True]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    hybrid.get_hybrid_scores(user_id=1, candidate_size=bad)

    # 12. Unknown User Handling
    def test_unknown_user(self):
        hybrid = HybridMovieRecommender(collaborative_recommender=self.cf_rec)
        hybrid.fit(self.synthetic_ratings)

        with self.assertRaises(ValueError):
            hybrid.recommend(user_id=999)

    # 13. Cold-Start User Handling
    def test_cold_start_user(self):
        hybrid = HybridMovieRecommender(collaborative_recommender=self.cf_rec)
        hybrid.fit(self.synthetic_ratings)

        with self.assertRaises(ValueError):
            hybrid.get_hybrid_scores(user_id=999)

    # 14. Score Normalization Formula Accuracy
    def test_min_max_score_normalization(self):
        scores = {101: 2.0, 102: 4.0, 103: 6.0}
        norm = normalize_scores(scores)
        self.assertEqual(norm[101], 0.0)
        self.assertEqual(norm[102], 0.5)
        self.assertEqual(norm[103], 1.0)

    # 15. Score Normalization Equal Scores Handling
    def test_equal_scores_normalization(self):
        scores = {101: 5.0, 102: 5.0}
        norm = normalize_scores(scores)
        self.assertEqual(norm[101], 0.0)
        self.assertEqual(norm[102], 0.0)

    # 16. Score Normalization Single Candidate Handling
    def test_single_candidate_normalization(self):
        scores = {101: 4.5}
        norm = normalize_scores(scores)
        self.assertEqual(norm[101], 0.0)

    # 17. Empty Score Normalization
    def test_empty_scores_normalization(self):
        self.assertEqual(normalize_scores({}), {})

    # 18. Pure CF Behavior (alpha = 0.0)
    def test_pure_cf_alpha_zero(self):
        hybrid = HybridMovieRecommender(
            collaborative_recommender=self.cf_rec, movies_df=self.synthetic_movies
        )
        hybrid.fit(self.synthetic_ratings)
        recs = hybrid.recommend(user_id=2, alpha=0.0, top_n=2)
        for r in recs:
            self.assertAlmostEqual(r["hybrid_score"], r["norm_cf_score"], places=6)

    # 19. Pure Content Behavior (alpha = 1.0)
    def test_pure_content_alpha_one(self):
        hybrid = HybridMovieRecommender(
            collaborative_recommender=self.cf_rec, movies_df=self.synthetic_movies
        )
        hybrid.fit(self.synthetic_ratings)
        recs = hybrid.recommend(user_id=2, alpha=1.0, top_n=2)
        for r in recs:
            self.assertAlmostEqual(r["hybrid_score"], r["norm_content_score"], places=6)

    # 20. Already-Rated Movies Strict Exclusion
    def test_already_rated_exclusion(self):
        hybrid = HybridMovieRecommender(
            collaborative_recommender=self.cf_rec, movies_df=self.synthetic_movies
        )
        hybrid.fit(self.synthetic_ratings)
        # User 1 rated 101, 102, 103, 104
        user1_rated = {101, 102, 103, 104}
        recs = hybrid.recommend(user_id=1, top_n=5)
        for r in recs:
            self.assertNotIn(r["movieId"], user1_rated)

    # 21. Deterministic Tie-Breaking
    def test_deterministic_tie_breaking(self):
        hybrid = HybridMovieRecommender(
            collaborative_recommender=self.cf_rec, movies_df=self.synthetic_movies
        )
        hybrid.fit(self.synthetic_ratings)
        res1 = hybrid.recommend(user_id=2, alpha=0.5, top_n=2)
        res2 = hybrid.recommend(user_id=2, alpha=0.5, top_n=2)
        self.assertEqual(res1, res2)

    # 22. No Source Data Mutation
    def test_source_recommender_immutability(self):
        hybrid = HybridMovieRecommender(
            collaborative_recommender=self.cf_rec, movies_df=self.synthetic_movies
        )
        hybrid.fit(self.synthetic_ratings)
        matrix_before = self.cf_rec.similarity_matrix.copy()
        hybrid.recommend(user_id=2, alpha=0.5, top_n=2)
        np.testing.assert_array_equal(self.cf_rec.similarity_matrix, matrix_before)

    # 23. End-to-End Recommendation Generation
    def test_end_to_end_hybrid(self):
        hybrid = HybridMovieRecommender(
            collaborative_recommender=self.cf_rec, movies_df=self.synthetic_movies
        )
        hybrid.fit(self.synthetic_ratings)
        recs = hybrid.recommend(user_id=2, alpha=0.5, top_n=3)
        self.assertIsInstance(recs, list)

    # 24. Evaluate Hybrid Model Integration
    def test_evaluate_hybrid_model_integration(self):
        eval_res = evaluate_hybrid_model(
            ratings_df=self.synthetic_ratings,
            movies_df=self.synthetic_movies,
            alpha_values=[0.0, 0.5, 1.0],
            k_values=[1, 2],
            min_ratings_per_movie=0,
            min_ratings_per_user=0,
            min_user_ratings=3,
            test_ratio=0.25,
            max_eval_users=5,
        )
        self.assertIn("mapping_stats", eval_res)
        self.assertIn("eval_table", eval_res)

    # 25. MovieLens + TMDB Integration Test (Gracefully Skipped if Data Missing)
    def test_movielens_tmdb_integration(self):
        project_root = Path(__file__).resolve().parent.parent
        ml_ratings_path = project_root / "data" / "raw" / "movielens" / "ml-latest-small" / "ratings.csv"
        ml_movies_path = project_root / "data" / "raw" / "movielens" / "ml-latest-small" / "movies.csv"
        tmdb_path = project_root / "data" / "processed" / "clean_movies.csv"

        if not ml_ratings_path.exists() or not ml_movies_path.exists() or not tmdb_path.exists():
            self.skipTest("MovieLens or TMDB processed dataset missing. Skipping integration test.")

        ratings_df = pd.read_csv(ml_ratings_path)
        movies_df = pd.read_csv(ml_movies_path)
        tmdb_df = pd.read_csv(tmdb_path)

        cf_rec = ItemBasedCollaborativeRecommender(
            min_ratings_per_movie=5, min_ratings_per_user=5, movies_df=movies_df
        )
        cf_rec.fit(ratings_df)

        personalizer = PersonalizedRecommender(tmdb_df)
        hybrid = HybridMovieRecommender(
            collaborative_recommender=cf_rec,
            personalizer=personalizer,
            movies_df=movies_df,
            tmdb_df=tmdb_df,
        )
        hybrid.fit(ratings_df)

        self.assertGreater(hybrid.mapping_stats["mapped_movies"], 0)
        recs = hybrid.recommend(user_id=1, alpha=0.5, top_n=5)
        self.assertEqual(len(recs), 5)


if __name__ == "__main__":
    unittest.main()
