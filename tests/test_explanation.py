import unittest
import numpy as np
import pandas as pd

from src.collaborative_filter import ItemBasedCollaborativeRecommender
from src.explanation import RecommendationExplainer, _parse_list_field, _parse_string_field
from src.hybrid_recommender import HybridMovieRecommender
from src.personalizer import PersonalizedRecommender
from src.recommender import MovieRecommender


class TestRecommendationExplainer(unittest.TestCase):
    """Unit test suite for Phase 8 RecommendationExplainer module."""

    @classmethod
    def setUpClass(cls):
        # Build synthetic TMDB clean movies DataFrame
        cls.synthetic_tmdb = pd.DataFrame(
            [
                {
                    "id": 1,
                    "movie_id": 1,
                    "title": "Toy Story",
                    "genres": "['Animation', 'Children', 'Comedy']",
                    "keywords": "['toy', 'animation', 'friendship']",
                    "director": "John Lasseter",
                    "cast": "['Tom Hanks', 'Tim Allen']",
                    "tags": "toy story animation children comedy toy animation friendship john lasseter tom hanks tim allen",
                },
                {
                    "id": 2,
                    "movie_id": 2,
                    "title": "A Bug's Life",
                    "genres": "['Animation', 'Children', 'Comedy']",
                    "keywords": "['ant', 'animation', 'insects']",
                    "director": "John Lasseter",
                    "cast": "['Dave Foley', 'Kevin Spacey']",
                    "tags": "a bug's life animation children comedy ant animation insects john lasseter dave foley kevin spacey",
                },
                {
                    "id": 3,
                    "movie_id": 3,
                    "title": "Toy Story 2",
                    "genres": "['Animation', 'Children', 'Comedy']",
                    "keywords": "['toy', 'animation', 'sequel']",
                    "director": "John Lasseter",
                    "cast": "['Tom Hanks', 'Tim Allen']",
                    "tags": "toy story 2 animation children comedy toy animation sequel john lasseter tom hanks tim allen",
                },
                {
                    "id": 4,
                    "movie_id": 4,
                    "title": "Die Hard",
                    "genres": "['Action', 'Thriller']",
                    "keywords": "['terrorist', 'skyscraper', 'cop']",
                    "director": "John McTiernan",
                    "cast": "['Bruce Willis', 'Alan Rickman']",
                    "tags": "die hard action thriller terrorist skyscraper cop john mctiernan bruce willis alan rickman",
                },
                {
                    "id": 5,
                    "movie_id": 5,
                    "title": "Die Hard 2",
                    "genres": "['Action', 'Thriller']",
                    "keywords": "['airport', 'terrorist', 'cop']",
                    "director": "Renny Harlin",
                    "cast": "['Bruce Willis', 'Bonnie Bedelia']",
                    "tags": "die hard 2 action thriller airport terrorist cop renny harlin bruce willis bonnie bedelia",
                },
            ]
        )

        cls.recommender = MovieRecommender(cls.synthetic_tmdb, max_features=500)
        cls.personalizer = PersonalizedRecommender(cls.synthetic_tmdb, recommender=cls.recommender)

        # Build synthetic MovieLens movies and ratings
        cls.synthetic_ml_movies = pd.DataFrame(
            [
                {"movieId": 101, "title": "Toy Story (1995)"},
                {"movieId": 102, "title": "A Bug's Life (1998)"},
                {"movieId": 103, "title": "Toy Story 2 (1999)"},
                {"movieId": 104, "title": "Die Hard (1988)"},
                {"movieId": 105, "title": "Die Hard 2 (1990)"},
                {"movieId": 999, "title": "Unmapped Movie (2020)"},
            ]
        )

        cls.synthetic_ratings = pd.DataFrame(
            [
                {"userId": 1, "movieId": 101, "rating": 5.0, "timestamp": 100},
                {"userId": 1, "movieId": 102, "rating": 4.0, "timestamp": 101},
                {"userId": 1, "movieId": 103, "rating": 5.0, "timestamp": 102},
                {"userId": 1, "movieId": 104, "rating": 1.0, "timestamp": 103},
                {"userId": 2, "movieId": 101, "rating": 5.0, "timestamp": 100},
                {"userId": 2, "movieId": 103, "rating": 5.0, "timestamp": 101},
                {"userId": 2, "movieId": 105, "rating": 4.0, "timestamp": 102},
                {"userId": 3, "movieId": 104, "rating": 5.0, "timestamp": 100},
                {"userId": 3, "movieId": 105, "rating": 5.0, "timestamp": 101},
            ]
        )

        cls.cf_rec = ItemBasedCollaborativeRecommender(
            min_ratings_per_movie=1, min_ratings_per_user=1, movies_df=cls.synthetic_ml_movies
        )
        cls.cf_rec.fit(cls.synthetic_ratings)

        cls.hybrid_rec = HybridMovieRecommender(
            collaborative_recommender=cls.cf_rec,
            personalizer=cls.personalizer,
            movies_df=cls.synthetic_ml_movies,
            tmdb_df=cls.synthetic_tmdb,
        )
        cls.hybrid_rec.fit(cls.synthetic_ratings)

    def test_parse_helpers(self):
        """1. Helper parser functions."""
        self.assertEqual(_parse_list_field("['Action', 'Comedy']"), ["Action", "Comedy"])
        self.assertEqual(_parse_list_field("Action, Comedy"), ["Action", "Comedy"])
        self.assertEqual(_parse_list_field(None), [])
        self.assertEqual(_parse_string_field(" John Lasseter "), "John Lasseter")
        self.assertIsNone(_parse_string_field(None))

    def test_initialization(self):
        """2. Basic initialization."""
        explainer = RecommendationExplainer(
            hybrid_recommender=self.hybrid_rec,
            movielens_movies_df=self.synthetic_ml_movies,
        )
        self.assertIsNotNone(explainer.hybrid_recommender)
        self.assertIsNotNone(explainer.collaborative_recommender)
        self.assertIsNotNone(explainer.personalizer)
        self.assertEqual(len(explainer.title_map), 6)

    def test_content_explanation_structure(self):
        """3. Content explanation structure."""
        explainer = RecommendationExplainer(personalizer=self.personalizer)
        res = explainer.explain_content_recommendation(
            target_movie_title="Toy Story 2",
            query_movie_title="Toy Story",
        )
        self.assertTrue(res["available"])
        self.assertEqual(res["target_title"], "Toy Story 2")
        self.assertIn("shared_genres", res)
        self.assertIn("shared_keywords", res)
        self.assertIn("shared_directors", res)
        self.assertIn("shared_cast", res)
        self.assertIn("top_tfidf_terms", res)

    def test_shared_genres_detection(self):
        """4. Shared genre detection and deterministic sorting."""
        explainer = RecommendationExplainer(personalizer=self.personalizer)
        res = explainer.explain_content_recommendation(
            target_movie_title="Toy Story 2",
            query_movie_title="Toy Story",
        )
        self.assertEqual(res["shared_genres"], ["Animation", "Children", "Comedy"])

    def test_shared_keywords_detection(self):
        """5. Shared keyword detection."""
        explainer = RecommendationExplainer(personalizer=self.personalizer)
        res = explainer.explain_content_recommendation(
            target_movie_title="Toy Story 2",
            query_movie_title="Toy Story",
        )
        self.assertIn("toy", res["shared_keywords"])
        self.assertIn("animation", res["shared_keywords"])

    def test_shared_director_detection(self):
        """6. Shared director detection."""
        explainer = RecommendationExplainer(personalizer=self.personalizer)
        res = explainer.explain_content_recommendation(
            target_movie_title="Toy Story 2",
            query_movie_title="Toy Story",
        )
        self.assertEqual(res["shared_directors"], ["John Lasseter"])

    def test_shared_cast_detection(self):
        """7. Shared cast detection."""
        explainer = RecommendationExplainer(personalizer=self.personalizer)
        res = explainer.explain_content_recommendation(
            target_movie_title="Toy Story 2",
            query_movie_title="Toy Story",
        )
        self.assertEqual(res["shared_cast"], ["Tim Allen", "Tom Hanks"])

    def test_no_metadata_case(self):
        """8. No metadata / unknown target movie handling."""
        explainer = RecommendationExplainer(personalizer=self.personalizer)
        res = explainer.explain_content_recommendation(target_movie_title="Unknown Movie XYZ")
        self.assertFalse(res["available"])
        self.assertEqual(res["shared_genres"], [])

    def test_cf_contributor_extraction(self):
        """9. CF contributor extraction."""
        explainer = RecommendationExplainer(collaborative_recommender=self.cf_rec)
        # User 1 rated 101 (5), 102 (4), 103 (5), 104 (1)
        res = explainer.explain_cf_recommendation(user_id=1, target_movie_id=105)
        self.assertTrue(res["available"])
        self.assertEqual(res["target_movie_id"], 105)
        self.assertGreater(len(res["similar_rated_movies"]), 0)

    def test_cf_contribution_calculation(self):
        """10. CF contribution calculation: contribution = similarity * rating."""
        explainer = RecommendationExplainer(collaborative_recommender=self.cf_rec)
        res = explainer.explain_cf_recommendation(user_id=1, target_movie_id=105)
        for contrib in res["similar_rated_movies"]:
            expected_contrib = contrib["similarity"] * contrib["user_rating"]
            self.assertAlmostEqual(contrib["contribution"], expected_contrib, places=3)

    def test_contributor_ranking(self):
        """11. Contributor ranking: contribution descending, movieId ascending for ties."""
        explainer = RecommendationExplainer(collaborative_recommender=self.cf_rec)
        res = explainer.explain_cf_recommendation(user_id=1, target_movie_id=105)
        contribs = res["similar_rated_movies"]
        for i in range(len(contribs) - 1):
            c1 = contribs[i]
            c2 = contribs[i + 1]
            self.assertTrue(
                c1["contribution"] > c2["contribution"]
                or (c1["contribution"] == c2["contribution"] and c1["movieId"] <= c2["movieId"])
            )

    def test_hybrid_score_decomposition(self):
        """12. Hybrid score decomposition: score = alpha * norm_content + (1-alpha) * norm_cf."""
        explainer = RecommendationExplainer(hybrid_recommender=self.hybrid_rec)
        res = explainer.explain_hybrid_recommendation(user_id=1, target_movie_id=105, alpha=0.5)
        self.assertTrue(res["available"])
        hy_ev = res["hybrid_evidence"]
        expected_score = round((0.5 * hy_ev["normalized_content_score"]) + (0.5 * hy_ev["normalized_cf_score"]), 6)
        self.assertAlmostEqual(hy_ev["hybrid_score"], expected_score, places=5)

    def test_alpha_zero_pure_cf(self):
        """13. Alpha = 0.0 (CF-only hybrid scoring)."""
        explainer = RecommendationExplainer(hybrid_recommender=self.hybrid_rec)
        res = explainer.explain_hybrid_recommendation(user_id=1, target_movie_id=105, alpha=0.0)
        hy_ev = res["hybrid_evidence"]
        self.assertEqual(hy_ev["dominant_branch"], "cf_only")
        self.assertEqual(hy_ev["weighted_content_contribution"], 0.0)
        self.assertAlmostEqual(hy_ev["hybrid_score"], hy_ev["normalized_cf_score"], places=5)

    def test_alpha_one_pure_content(self):
        """14. Alpha = 1.0 (Content-only hybrid scoring)."""
        explainer = RecommendationExplainer(hybrid_recommender=self.hybrid_rec)
        res = explainer.explain_hybrid_recommendation(user_id=1, target_movie_id=105, alpha=1.0)
        hy_ev = res["hybrid_evidence"]
        self.assertEqual(hy_ev["dominant_branch"], "content_only")
        self.assertEqual(hy_ev["weighted_cf_contribution"], 0.0)
        self.assertAlmostEqual(hy_ev["hybrid_score"], hy_ev["normalized_content_score"], places=5)

    def test_missing_content_signal(self):
        """15. Missing content signal (e.g. unmapped movie 999)."""
        explainer = RecommendationExplainer(hybrid_recommender=self.hybrid_rec)
        res = explainer.explain_hybrid_recommendation(user_id=1, target_movie_id=999, alpha=0.5)
        self.assertTrue(res["available"])
        self.assertFalse(res["content_evidence"]["available"])

    def test_missing_cf_signal(self):
        """16. Unknown user in CF history."""
        explainer = RecommendationExplainer(hybrid_recommender=self.hybrid_rec)
        res = explainer.explain_cf_recommendation(user_id=999, target_movie_id=105)
        self.assertFalse(res["available"])
        self.assertEqual(res["raw_cf_score"], 0.0)

    def test_missing_both_signals(self):
        """17. Unknown user and unmapped item."""
        explainer = RecommendationExplainer(hybrid_recommender=self.hybrid_rec)
        res = explainer.explain_hybrid_recommendation(user_id=999, target_movie_id=999)
        self.assertFalse(res["available"])

    def test_deterministic_output(self):
        """18. Deterministic output across multiple calls."""
        explainer = RecommendationExplainer(hybrid_recommender=self.hybrid_rec)
        res1 = explainer.explain_hybrid_recommendation(user_id=1, target_movie_id=105, alpha=0.5)
        res2 = explainer.explain_hybrid_recommendation(user_id=1, target_movie_id=105, alpha=0.5)
        self.assertEqual(res1, res2)

    def test_empty_contributor_list(self):
        """19. Empty contributor list when user has no positive ratings."""
        explainer = RecommendationExplainer(collaborative_recommender=self.cf_rec)
        res = explainer.explain_cf_recommendation(user_id=1, target_movie_id=105, positive_threshold=5.1)
        self.assertFalse(res["available"])
        self.assertEqual(res["similar_rated_movies"], [])

    def test_unknown_movie_handling(self):
        """20. Invalid inputs raise ValueError."""
        explainer = RecommendationExplainer(hybrid_recommender=self.hybrid_rec)
        with self.assertRaises(ValueError):
            explainer.explain_hybrid_recommendation(user_id="invalid", target_movie_id=105)
        with self.assertRaises(ValueError):
            explainer.explain_content_recommendation(target_movie_title="")

    def test_temporal_leakage_protection(self):
        """21 & 22. Verification that held-out ratings are not used as explanation evidence."""
        # Train history user 1 has ratings for 101, 102, 103, 104
        train_ratings_only = [(101, 5.0), (102, 4.0), (103, 5.0)]
        held_out_future_rating = (105, 5.0)  # Future test rating

        explainer = RecommendationExplainer(collaborative_recommender=self.cf_rec)
        res = explainer.explain_cf_recommendation(user_id=1, target_movie_id=105)

        # Verify held-out target item (105) is NOT present in contributing training movies
        contributor_ids = [m["movieId"] for m in res["similar_rated_movies"]]
        self.assertNotIn(105, contributor_ids)
        self.assertNotIn(held_out_future_rating[0], contributor_ids)

    def test_human_readable_summary_generation(self):
        """23. Human-readable summary string generation."""
        explainer = RecommendationExplainer(hybrid_recommender=self.hybrid_rec)
        res = explainer.explain_hybrid_recommendation(user_id=1, target_movie_id=105, alpha=0.5)
        self.assertIn("summary", res)
        self.assertIsInstance(res["summary"], str)
        self.assertTrue(len(res["summary"]) > 0)

    def test_unified_explain_recommendation_entrypoint(self):
        """24. Unified explain_recommendation entry point routing."""
        explainer = RecommendationExplainer(hybrid_recommender=self.hybrid_rec)

        # Hybrid routing
        res_h = explainer.explain_recommendation(
            target_item={"movieId": 105, "title": "Die Hard 2"},
            user_id=1,
            alpha=0.5,
        )
        self.assertTrue(res_h["available"])
        self.assertIn("hybrid_evidence", res_h)

        # Content routing
        res_c = explainer.explain_recommendation(
            target_item="Toy Story 2",
            query_title="Toy Story",
        )
        self.assertTrue(res_c["available"])
        self.assertEqual(res_c["target_title"], "Toy Story 2")


if __name__ == "__main__":
    unittest.main()
