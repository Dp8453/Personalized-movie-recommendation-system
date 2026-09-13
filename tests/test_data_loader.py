import os
import sys
import unittest
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_movies


class TestDataLoader(unittest.TestCase):
    def test_load_movies_returns_dataframe(self):
        """Test that load_movies returns a pandas DataFrame."""
        df = load_movies()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

    def test_expected_columns_exist(self):
        """Test that critical columns exist in the loaded dataset."""
        df = load_movies()
        required_columns = [
            "movie_id",
            "title",
            "overview",
            "genres",
            "keywords",
            "cast",
            "crew",
            "vote_average",
            "vote_count",
        ]
        for col in required_columns:
            self.assertIn(col, df.columns, f"Required column '{col}' is missing.")

    def test_invalid_directory_raises_file_not_found(self):
        """Test that passing an invalid directory raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_movies(data_dir="/invalid/directory/path")


if __name__ == "__main__":
    unittest.main()
