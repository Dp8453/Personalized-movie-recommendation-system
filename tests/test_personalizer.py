import os
import sys
import unittest
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.personalizer import PersonalizedRecommender


class TestPersonalizedRecommender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Build a small, deterministic synthetic dataset for fast testing."""
        cls.synthetic_data = pd.DataFrame(
            [
                {
                    "id": 1,
                    "title": "Avatar",
                    "tags": "action adventure fantasy space alien jamescameron samworthington",
                },
                {
                    "id": 2,
                    "title": "Aliens",
                    "tags": "action sci-fi space alien sigourneyweaver jamescameron",
                },
                {
                    "id": 3,
                    "title": "The Dark Knight",
                    "tags": "action crime drama gotham batman christophernolan christianbale",
                },
                {
                    "id": 4,
                    "title": "Inception",
                    "tags": "action sci-fi thriller dream christophernolan leonardodicaprio",
                },
                {
                    "id": 5,
                    "title": "Titanic",
                    "tags": "drama romance ship ocean jamescameron leonardodicaprio katewinslet",
                },
                {
                    "id": 6,
                    "title": "Star Trek",
                    "tags": "action adventure sci-fi space alien starfleet spock",
                },
            ]
        )
        cls.personalizer = PersonalizedRecommender(cls.synthetic_data, max_features=100)

    def test_initialization(self):
        """Test successful personalizer initialization."""
        self.assertEqual(len(self.personalizer.movies_df), 6)
        self.assertIn("avatar", self.personalizer.title_to_index)

    def test_user_profile_vector_shape(self):
        """Test user profile vector shape."""
        history = [("Avatar", 5)]
        profile = self.personalizer.build_user_profile(history)
        self.assertIsInstance(profile, np.ndarray)
        self.assertEqual(profile.shape, (1, self.personalizer.tfidf_matrix.shape[1]))

    def test_positive_rating_contribution(self):
        """Test that positive ratings (5) produce positive preference weights."""
        entries = self.personalizer.validate_user_ratings([("Avatar", 5)])
        self.assertEqual(entries[0][3], 1.0)

    def test_negative_rating_contribution(self):
        """Test that negative ratings (1) produce negative preference weights."""
        entries = self.personalizer.validate_user_ratings([("Titanic", 1)])
        self.assertEqual(entries[0][3], -1.0)

    def test_neutral_rating_behavior(self):
        """Test neutral ratings (3) produce zero preference weight."""
        entries = self.personalizer.validate_user_ratings([("Titanic", 3)])
        self.assertEqual(entries[0][3], 0.0)

    def test_net_weight_zero_with_non_zero_sum_abs_weights_is_valid(self):
        """
        Test the [5, 1] case: ratings with +1.0 and -1.0 weights.
        Net weight = 0, but sum(abs(w_i)) = 2.0 != 0.
        The user profile MUST be valid!
        """
        history = [("Avatar", 5), ("Titanic", 1)]
        profile = self.personalizer.build_user_profile(history)
        self.assertIsInstance(profile, np.ndarray)
        self.assertFalse(np.all(profile == 0))

    def test_all_neutral_history_raises_value_error(self):
        """Test that history with only neutral ratings (sum(abs(w_i)) == 0) raises ValueError."""
        history = [("Avatar", 3), ("Titanic", 3)]
        with self.assertRaises(ValueError):
            self.personalizer.build_user_profile(history)

    def test_empty_user_history_raises_value_error(self):
        """Test empty user history raises ValueError."""
        with self.assertRaises(ValueError):
            self.personalizer.build_user_profile([])

    def test_invalid_rating_value_raises_value_error(self):
        """Test invalid ratings (floats, out-of-range ints, non-ints) raise ValueError."""
        invalid_ratings = [
            [("Avatar", 0)],
            [("Avatar", 6)],
            [("Avatar", -1)],
            [("Avatar", 4.5)],
            [("Avatar", "five")],
            [("Avatar", True)],
        ]
        for inv_history in invalid_ratings:
            with self.assertRaises(ValueError):
                self.personalizer.build_user_profile(inv_history)

    def test_duplicate_movie_titles_raise_value_error(self):
        """Test duplicate movie entries in rating history raise ValueError."""
        history = [("Avatar", 5), ("avatar", 4)]
        with self.assertRaises(ValueError):
            self.personalizer.build_user_profile(history)

    def test_unknown_movie_raises_value_error(self):
        """Test searching for an unknown movie title raises ValueError."""
        history = [("NonExistentMovie999", 5)]
        with self.assertRaises(ValueError):
            self.personalizer.build_user_profile(history)

    def test_case_insensitive_movie_title_lookup(self):
        """Test case-insensitive title resolution in rating history."""
        prof1 = self.personalizer.build_user_profile([("Avatar", 5)])
        prof2 = self.personalizer.build_user_profile([("AVATAR", 5)])
        prof3 = self.personalizer.build_user_profile([("avatar", 5)])

        np.testing.assert_array_almost_equal(prof1, prof2)
        np.testing.assert_array_almost_equal(prof1, prof3)

    def test_rated_movies_strictly_excluded(self):
        """Test that movies present in the user's history are strictly excluded from output."""
        history = [("Avatar", 5), ("Aliens", 5)]
        recs = self.personalizer.recommend_for_user(history, top_n=4)
        rec_titles = [r["title"] for r in recs]
        self.assertNotIn("Avatar", rec_titles)
        self.assertNotIn("Aliens", rec_titles)

    def test_top_n_count(self):
        """Test top-N recommendation count."""
        history = [("Avatar", 5)]
        recs = self.personalizer.recommend_for_user(history, top_n=3)
        self.assertEqual(len(recs), 3)

    def test_top_n_invalid_raises_value_error(self):
        """Test top_n <= 0 raises ValueError."""
        history = [("Avatar", 5)]
        with self.assertRaises(ValueError):
            self.personalizer.recommend_for_user(history, top_n=0)
        with self.assertRaises(ValueError):
            self.personalizer.recommend_for_user(history, top_n=-5)

    def test_top_n_larger_than_candidates_caps_gracefully(self):
        """Test top_n > unrated candidates returns all available candidates."""
        history = [("Avatar", 5)]  # 5 unrated candidates remaining
        recs = self.personalizer.recommend_for_user(history, top_n=100)
        self.assertEqual(len(recs), 5)

    def test_recommendations_sorted_descending(self):
        """Test personalized scores are sorted in descending order."""
        history = [("Avatar", 5), ("Aliens", 5)]
        recs = self.personalizer.recommend_for_user(history, top_n=3)
        scores = [r["personalized_score"] for r in recs]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_negative_preference_changes_ranking_and_scores(self):
        """Test that changing a movie rating from positive (5) to negative (1) alters ranking/scores."""
        history_pos = [("Avatar", 5), ("Titanic", 5)]
        history_neg = [("Avatar", 5), ("Titanic", 1)]

        recs_pos = self.personalizer.recommend_for_user(history_pos, top_n=3)
        recs_neg = self.personalizer.recommend_for_user(history_neg, top_n=3)

        scores_pos = [r["personalized_score"] for r in recs_pos]
        scores_neg = [r["personalized_score"] for r in recs_neg]

        self.assertNotEqual(scores_pos, scores_neg)

    def test_deterministic_output(self):
        """Test recommendation generation is 100% deterministic."""
        history = [("Avatar", 5), ("The Dark Knight", 4)]
        recs1 = self.personalizer.recommend_for_user(history, top_n=3)
        recs2 = self.personalizer.recommend_for_user(history, top_n=3)
        self.assertEqual(recs1, recs2)


if __name__ == "__main__":
    unittest.main()
