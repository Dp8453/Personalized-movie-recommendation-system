import os
import sys
import unittest
import pandas as pd

# Ensure src module resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluator import (
    extract_titles,
    validate_k,
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    evaluate_recommendations,
    evaluate_scenarios,
)
from src.personalizer import PersonalizedRecommender


class TestEvaluator(unittest.TestCase):
    """
    Unit test suite verifying exact mathematical correctness, input validation,
    edge-case handling, and data leakage detection for offline evaluation metrics.
    """

    def setUp(self):
        self.recommended_perfect = ["Avatar", "Aliens", "Titanic"]
        self.relevant_perfect = ["Avatar", "Aliens", "Titanic"]

    # -------------------------------------------------------------------------
    # 1. Perfect Recommendations
    # -------------------------------------------------------------------------
    def test_perfect_recommendations(self):
        rec = ["Avatar", "Aliens", "Titanic"]
        rel = ["Avatar", "Aliens", "Titanic"]
        k = 3

        p = precision_at_k(rec, rel, k)
        r = recall_at_k(rec, rel, k)
        n = ndcg_at_k(rec, rel, k)

        self.assertEqual(p, 1.0)
        self.assertEqual(r, 1.0)
        self.assertEqual(n, 1.0)

    # -------------------------------------------------------------------------
    # 2. No Relevant Recommendations
    # -------------------------------------------------------------------------
    def test_no_relevant_recommendations(self):
        rec = ["Inception", "Interstellar"]
        rel = ["Avatar", "Aliens"]
        k = 2

        p = precision_at_k(rec, rel, k)
        r = recall_at_k(rec, rel, k)
        n = ndcg_at_k(rec, rel, k)

        self.assertEqual(p, 0.0)
        self.assertEqual(r, 0.0)
        self.assertEqual(n, 0.0)

    # -------------------------------------------------------------------------
    # 3. Partial Precision
    # -------------------------------------------------------------------------
    def test_partial_precision(self):
        rec = ["Avatar", "Inception", "Interstellar", "The Prestige"]
        rel = ["Avatar"]
        k = 4

        p = precision_at_k(rec, rel, k)
        r = recall_at_k(rec, rel, k)

        # Hits = 1, K = 4 -> 1/4 = 0.25
        self.assertEqual(p, 0.25)
        # Hits = 1, Total Relevant = 1 -> 1/1 = 1.0
        self.assertEqual(r, 1.0)

    # -------------------------------------------------------------------------
    # 4. Partial Recall
    # -------------------------------------------------------------------------
    def test_partial_recall(self):
        rec = ["Avatar"]
        rel = ["Avatar", "Aliens"]
        k = 1

        p = precision_at_k(rec, rel, k)
        r = recall_at_k(rec, rel, k)

        # Hits = 1, K = 1 -> 1/1 = 1.0
        self.assertEqual(p, 1.0)
        # Hits = 1, Total Relevant = 2 -> 1/2 = 0.5
        self.assertEqual(r, 0.5)

    # -------------------------------------------------------------------------
    # 5. Position Sensitivity (Rank 1 vs Rank 3 for NDCG)
    # -------------------------------------------------------------------------
    def test_ndcg_rank_sensitivity(self):
        rel = ["Avatar"]
        k = 3

        rec_rank_1 = ["Avatar", "Inception", "Interstellar"]
        rec_rank_3 = ["Inception", "Interstellar", "Avatar"]

        ndcg_1 = ndcg_at_k(rec_rank_1, rel, k)
        ndcg_3 = ndcg_at_k(rec_rank_3, rel, k)

        # Rank 1: DCG = 1/log2(2) = 1.0, IDCG = 1.0 -> NDCG = 1.0
        # Rank 3: DCG = 1/log2(4) = 0.5, IDCG = 1.0 -> NDCG = 0.5
        self.assertEqual(ndcg_1, 1.0)
        self.assertEqual(ndcg_3, 0.5)
        self.assertGreater(ndcg_1, ndcg_3)

    # -------------------------------------------------------------------------
    # 6. K Smaller Than Recommendation List
    # -------------------------------------------------------------------------
    def test_k_smaller_than_recommendations(self):
        rec = ["Inception", "Interstellar", "Avatar"]
        rel = ["Avatar"]
        k = 2  # Avatar is at rank 3, cut off at K=2

        p = precision_at_k(rec, rel, k)
        r = recall_at_k(rec, rel, k)
        n = ndcg_at_k(rec, rel, k)

        self.assertEqual(p, 0.0)
        self.assertEqual(r, 0.0)
        self.assertEqual(n, 0.0)

    # -------------------------------------------------------------------------
    # 7. K Larger Than Recommendation List
    # -------------------------------------------------------------------------
    def test_k_larger_than_recommendations(self):
        rec = ["Avatar"]
        rel = ["Avatar"]
        k = 5

        p = precision_at_k(rec, rel, k)
        r = recall_at_k(rec, rel, k)
        n = ndcg_at_k(rec, rel, k)

        # Hits = 1, K = 5 -> 1/5 = 0.2
        self.assertEqual(p, 0.2)
        # Hits = 1, Total Relevant = 1 -> 1.0
        self.assertEqual(r, 1.0)
        # DCG = 1.0, IDCG = 1.0 -> 1.0
        self.assertEqual(n, 1.0)

    # -------------------------------------------------------------------------
    # 8. Empty Recommendations
    # -------------------------------------------------------------------------
    def test_empty_recommendations(self):
        rec = []
        rel = ["Avatar", "Aliens"]
        k = 3

        p = precision_at_k(rec, rel, k)
        r = recall_at_k(rec, rel, k)
        n = ndcg_at_k(rec, rel, k)

        self.assertEqual(p, 0.0)
        self.assertEqual(r, 0.0)
        self.assertEqual(n, 0.0)

    # -------------------------------------------------------------------------
    # 9. Empty Relevant Set
    # -------------------------------------------------------------------------
    def test_empty_relevant_set(self):
        rec = ["Avatar", "Aliens"]
        rel = []
        k = 2

        p = precision_at_k(rec, rel, k)
        r = recall_at_k(rec, rel, k)
        n = ndcg_at_k(rec, rel, k)

        self.assertEqual(p, 0.0)
        self.assertEqual(r, 0.0)
        self.assertEqual(n, 0.0)

    # -------------------------------------------------------------------------
    # 10. Invalid K Validation
    # -------------------------------------------------------------------------
    def test_invalid_k(self):
        rec = ["Avatar"]
        rel = ["Avatar"]

        invalid_ks = [0, -1, -5, 3.5, True, False, "5", None, [5]]
        for bad_k in invalid_ks:
            with self.subTest(bad_k=bad_k):
                with self.assertRaises(ValueError):
                    precision_at_k(rec, rel, bad_k)
                with self.assertRaises(ValueError):
                    recall_at_k(rec, rel, bad_k)
                with self.assertRaises(ValueError):
                    ndcg_at_k(rec, rel, bad_k)
                with self.assertRaises(ValueError):
                    validate_k(bad_k)

    # -------------------------------------------------------------------------
    # 11. Duplicate Recommendations Deduplication
    # -------------------------------------------------------------------------
    def test_duplicate_recommendations_deduplication(self):
        # Duplicates should be stripped while preserving first appearance
        rec = ["Avatar", "Avatar", "Inception"]
        rel = ["Avatar"]
        k = 2

        extracted = extract_titles(rec)
        self.assertEqual(extracted, ["Avatar", "Inception"])

        p = precision_at_k(rec, rel, k)
        r = recall_at_k(rec, rel, k)
        n = ndcg_at_k(rec, rel, k)

        # Hits = 1 in ['Avatar', 'Inception'], K = 2 -> 1/2 = 0.5
        self.assertEqual(p, 0.5)
        self.assertEqual(r, 1.0)
        self.assertEqual(n, 1.0)

    # -------------------------------------------------------------------------
    # 12. NDCG Bounds Guarantee [0, 1]
    # -------------------------------------------------------------------------
    def test_ndcg_bounds(self):
        test_cases = [
            (["Avatar", "Aliens"], ["Avatar"], 2),
            (["Inception", "Avatar"], ["Avatar", "Aliens"], 3),
            (["A", "B", "C", "D"], ["C", "D"], 4),
            ([], ["Avatar"], 5),
            (["Avatar"], [], 5),
        ]
        for rec, rel, k in test_cases:
            with self.subTest(rec=rec, rel=rel, k=k):
                n = ndcg_at_k(rec, rel, k)
                self.assertGreaterEqual(n, 0.0)
                self.assertLessEqual(n, 1.0)

    # -------------------------------------------------------------------------
    # 13. Data Leakage Detection in Scenarios
    # -------------------------------------------------------------------------
    def test_data_leakage_detection(self):
        # Held-out relevant movie "Avatar" is illegally included in user history
        leakage_scenario = [
            {
                "name": "Leakage_Test",
                "user_history": [("Avatar", 5), ("Aliens", 5)],
                "relevant_movies": ["Avatar", "The Terminator"],
            }
        ]

        class DummyRecommender:
            def recommend_for_user(self, history, top_n):
                return [{"title": "The Terminator"}]

        dummy_rec = DummyRecommender()
        with self.assertRaises(ValueError) as ctx:
            evaluate_scenarios(dummy_rec, leakage_scenario, k=2)

        self.assertIn("Data leakage detected", str(ctx.exception))
        self.assertIn("avatar", str(ctx.exception).lower())

    # -------------------------------------------------------------------------
    # 14. End-to-End Scenario Evaluation Integration
    # -------------------------------------------------------------------------
    def test_evaluate_scenarios_integration(self):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        processed_path = os.path.join(project_root, "data", "processed", "clean_movies.csv")

        if not os.path.exists(processed_path):
            self.skipTest("clean_movies.csv not available for integration test.")

        clean_df = pd.read_csv(processed_path)
        personalizer = PersonalizedRecommender(clean_df)

        scenarios = [
            {
                "name": "Batman_Cluster",
                "user_history": [("Batman Begins", 5), ("The Dark Knight", 5), ("Titanic", 1)],
                "relevant_movies": ["The Dark Knight Rises", "Batman Returns"],
            },
            {
                "name": "Pixar_Cluster",
                "user_history": [("Toy Story", 5), ("Toy Story 2", 5), ("The Godfather", 1)],
                "relevant_movies": ["Toy Story 3", "Monsters, Inc."],
            },
        ]

        res = evaluate_scenarios(personalizer, scenarios, k=5)

        self.assertIn("scenarios", res)
        self.assertIn("aggregate", res)
        self.assertEqual(len(res["scenarios"]), 2)

        agg = res["aggregate"]
        self.assertIn("mean_precision@5", agg)
        self.assertIn("mean_recall@5", agg)
        self.assertIn("mean_ndcg@5", agg)
        self.assertEqual(agg["k"], 5)
        self.assertEqual(agg["num_scenarios"], 2)


if __name__ == "__main__":
    unittest.main()
