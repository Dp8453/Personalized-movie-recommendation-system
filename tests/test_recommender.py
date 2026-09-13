import os
import sys
import unittest
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.recommender import MovieRecommender


class TestMovieRecommender(unittest.TestCase):
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
                    "title": "Star Trek",
                    "tags": "action adventure sci-fi space alien starfleet spock",
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
            ]
        )
        cls.recommender = MovieRecommender(cls.synthetic_data, max_features=100)

    def test_recommender_initialization(self):
        """Test recommender initialization on dataset."""
        self.assertEqual(len(self.recommender.movies_df), 5)
        self.assertIn("Avatar".lower(), self.recommender.title_to_index)

    def test_missing_required_columns_raises_key_error(self):
        """Test that missing required columns raise KeyError."""
        invalid_df = pd.DataFrame([{"movie_id": 1, "name": "Test"}])
        with self.assertRaises(KeyError):
            MovieRecommender(invalid_df)

    def test_tfidf_matrix_shape(self):
        """Test TF-IDF matrix dimensions."""
        shape = self.recommender.get_tfidf_matrix_shape()
        self.assertEqual(shape[0], 5)
        self.assertGreater(shape[1], 0)

    def test_vocab_size(self):
        """Test vocabulary size extraction."""
        vocab_size = self.recommender.get_vocab_size()
        self.assertGreater(vocab_size, 0)
        self.assertLessEqual(vocab_size, 100)

    def test_similarity_matrix_shape(self):
        """Test Cosine Similarity matrix dimensions."""
        self.assertEqual(self.recommender.get_similarity_matrix_shape(), (5, 5))

    def test_valid_movie_recommendation(self):
        """Test recommendation generation for a valid movie."""
        recs = self.recommender.recommend("Avatar", top_n=2)
        self.assertIsInstance(recs, list)
        self.assertEqual(len(recs), 2)
        self.assertIn("title", recs[0])
        self.assertIn("similarity_score", recs[0])

    def test_case_insensitive_title_lookup(self):
        """Test case-insensitive title resolution."""
        recs_upper = self.recommender.recommend("AVATAR", top_n=2)
        recs_lower = self.recommender.recommend("avatar", top_n=2)
        recs_mixed = self.recommender.recommend("AvAtAr", top_n=2)

        self.assertEqual(recs_upper, recs_lower)
        self.assertEqual(recs_upper, recs_mixed)

    def test_query_movie_self_exclusion(self):
        """Test that the query movie itself is strictly excluded from recommendations."""
        query_title = "Avatar"
        recs = self.recommender.recommend(query_title, top_n=4)
        rec_titles = [r["title"] for r in recs]
        self.assertNotIn(query_title, rec_titles)

    def test_recommendations_sorted_descending(self):
        """Test that similarity scores are sorted in descending order."""
        recs = self.recommender.recommend("Avatar", top_n=3)
        scores = [r["similarity_score"] for r in recs]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_no_duplicate_recommendations(self):
        """Test that recommendations contain no duplicate titles."""
        recs = self.recommender.recommend("Avatar", top_n=3)
        rec_titles = [r["title"] for r in recs]
        self.assertEqual(len(rec_titles), len(set(rec_titles)))

    def test_similarity_scores_in_valid_range(self):
        """Test that similarity scores are bounded between 0.0 and 1.0."""
        recs = self.recommender.recommend("Avatar", top_n=4)
        for r in recs:
            score = r["similarity_score"]
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_invalid_title_raises_value_error(self):
        """Test that searching for a non-existent movie title raises ValueError."""
        with self.assertRaises(ValueError):
            self.recommender.recommend("NonExistentMovie123")

        with self.assertRaises(ValueError):
            self.recommender.recommend("")

    def test_top_n_bounds_handling(self):
        """Test top_n bounds handling (<= 0 and > total candidates)."""
        # top_n <= 0 should return empty list
        self.assertEqual(self.recommender.recommend("Avatar", top_n=0), [])
        self.assertEqual(self.recommender.recommend("Avatar", top_n=-5), [])

        # top_n > total candidate count should cap at total available candidates (4 candidates for dataset of 5)
        recs_large = self.recommender.recommend("Avatar", top_n=100)
        self.assertEqual(len(recs_large), 4)


if __name__ == "__main__":
    unittest.main()
