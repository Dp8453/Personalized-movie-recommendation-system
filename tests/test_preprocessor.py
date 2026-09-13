import os
import sys
import unittest
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessor import (
    safe_eval_json,
    extract_genres,
    extract_keywords,
    extract_top_cast,
    extract_director,
    collapse_spaces,
    clean_overview,
    build_tags,
    preprocess_data,
)


class TestPreprocessor(unittest.TestCase):
    def test_safe_eval_json_handles_edge_cases(self):
        """Test safe_eval_json on valid, invalid, empty, and NaN inputs."""
        self.assertEqual(safe_eval_json('[{"name": "Action"}]'), [{"name": "Action"}])
        self.assertEqual(safe_eval_json(None), [])
        self.assertEqual(safe_eval_json(float("nan")), [])
        self.assertEqual(safe_eval_json(""), [])
        self.assertEqual(safe_eval_json("invalid json string"), [])
        self.assertEqual(safe_eval_json("12345"), [])

    def test_extract_genres(self):
        """Test extraction of genre names."""
        genres_json = '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]'
        self.assertEqual(extract_genres(genres_json), ["Action", "Adventure"])
        self.assertEqual(extract_genres(None), [])

    def test_extract_keywords(self):
        """Test extraction of keyword names."""
        keywords_json = '[{"id": 1463, "name": "space war"}, {"id": 2964, "name": "future"}]'
        self.assertEqual(extract_keywords(keywords_json), ["space war", "future"])
        self.assertEqual(extract_keywords(None), [])

    def test_extract_top_cast(self):
        """Test extraction of top 3 cast members."""
        cast_json = (
            '[{"name": "Sam Worthington"}, {"name": "Zoe Saldana"}, '
            '{"name": "Sigourney Weaver"}, {"name": "Stephen Lang"}]'
        )
        # Should restrict to top 3 actors only
        extracted = extract_top_cast(cast_json, top_n=3)
        self.assertEqual(len(extracted), 3)
        self.assertEqual(extracted, ["Sam Worthington", "Zoe Saldana", "Sigourney Weaver"])

    def test_extract_director(self):
        """Test director extraction from crew list."""
        crew_json = (
            '[{"job": "Editor", "name": "Stephen E. Rivkin"}, '
            '{"job": "Director", "name": "James Cameron"}]'
        )
        self.assertEqual(extract_director(crew_json), ["James Cameron"])

        # Case when no director is present
        no_dir_crew = '[{"job": "Producer", "name": "Jon Landau"}]'
        self.assertEqual(extract_director(no_dir_crew), [])

    def test_collapse_spaces(self):
        """Test space collapsing for multi-word names."""
        names = ["Sam Worthington", "James Cameron", "Sci Fi"]
        collapsed = collapse_spaces(names)
        self.assertEqual(collapsed, ["SamWorthington", "JamesCameron", "SciFi"])

    def test_clean_overview(self):
        """Test cleaning plot overview and handling missing values."""
        overview = "A paraplegic Marine is dispatched to Pandora."
        self.assertEqual(clean_overview(overview), ["A", "paraplegic", "Marine", "is", "dispatched", "to", "Pandora."])
        self.assertEqual(clean_overview(None), [])
        self.assertEqual(clean_overview(float("nan")), [])
        self.assertEqual(clean_overview("nan"), [])

    def test_build_tags_and_no_nan_contamination(self):
        """Test building tags string and ensuring no 'nan' or 'None' literal strings."""
        overview_tokens = ["in", "space"]
        genres = ["Action", "Sci Fi"]
        keywords = ["space war"]
        cast = ["Sam Worthington"]
        director = ["James Cameron"]

        tags = build_tags(overview_tokens, genres, keywords, cast, director)
        self.assertIsInstance(tags, str)
        self.assertIn("in space action scifi spacewar samworthington jamescameron", tags)
        self.assertNotIn("nan", tags.split())
        self.assertNotIn("none", tags.split())

    def test_preprocess_data_synthetic(self):
        """Test complete preprocessing pipeline on a small synthetic DataFrame."""
        synthetic_df = pd.DataFrame(
            [
                {
                    "id": 1,
                    "title": "Test Movie",
                    "overview": "A test movie overview synopsis.",
                    "genres": '[{"name": "Action"}]',
                    "keywords": '[{"name": "test"}]',
                    "cast": '[{"name": "Actor One"}, {"name": "Actor Two"}]',
                    "crew": '[{"job": "Director", "name": "Director One"}]',
                }
            ]
        )

        # Temporary path for synthetic output
        temp_output = os.path.join(os.path.dirname(__file__), "temp_clean_movies.csv")
        try:
            processed = preprocess_data(synthetic_df, output_path=temp_output)
            self.assertIn("tags", processed.columns)
            self.assertEqual(len(processed), 1)
            self.assertEqual(processed["director"].iloc[0], ["Director One"])
            self.assertTrue(os.path.exists(temp_output))
        finally:
            if os.path.exists(temp_output):
                os.remove(temp_output)


if __name__ == "__main__":
    unittest.main()
