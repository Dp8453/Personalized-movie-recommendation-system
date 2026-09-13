import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    """
    Classical Content-Based Movie Recommendation Engine using TF-IDF Vectorization
    and Cosine Similarity on movie metadata tags.
    """

    def __init__(
        self,
        movies_df: pd.DataFrame,
        max_features: int = 5000,
        stop_words: str | list[str] | None = "english",
    ):
        """
        Initializes the recommendation engine.

        Parameters
        ----------
        movies_df : pd.DataFrame
            DataFrame containing processed movie metadata with 'title' and 'tags' columns.
        max_features : int, default=5000
            Maximum number of top TF-IDF vocabulary features to retain.
        stop_words : str, list, or None, default='english'
            Stop words parameter for TfidfVectorizer.
        """
        if "title" not in movies_df.columns:
            raise KeyError("DataFrame must contain a 'title' column.")
        if "tags" not in movies_df.columns:
            raise KeyError("DataFrame must contain a 'tags' column.")

        # Preserve row indexing and store a clean copy
        self.movies_df = movies_df.reset_index(drop=True).copy()

        # Fill missing tags safely with empty string
        self.tags = self.movies_df["tags"].fillna("").astype(str)

        # Configure and fit TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=stop_words,
        )

        # Fit vectorizer and build TF-IDF sparse feature matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(self.tags)

        # Compute full Cosine Similarity matrix (shape: num_movies x num_movies)
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

        # Build robust case-insensitive title lookup dictionary
        self.title_to_index = {}
        for idx, title in enumerate(self.movies_df["title"]):
            clean_title = str(title).strip().lower()
            # Store first occurrence index if title appears multiple times
            if clean_title not in self.title_to_index:
                self.title_to_index[clean_title] = idx

    def recommend(self, movie_title: str, top_n: int = 5) -> list[dict[str, str | float]]:
        """
        Recommends top N similar movies for a given query movie title.

        Parameters
        ----------
        movie_title : str
            Title of the movie to generate recommendations for (case-insensitive).
        top_n : int, default=5
            Number of top similar movies to return.

        Returns
        -------
        list of dict
            List of recommendation dicts: [{'title': str, 'similarity_score': float}, ...]

        Raises
        ------
        ValueError
            If the movie title is not found in the dataset.
        """
        if not isinstance(movie_title, str) or not movie_title.strip():
            raise ValueError(f"Movie '{movie_title}' not found in dataset.")

        clean_title = movie_title.strip().lower()

        if clean_title not in self.title_to_index:
            raise ValueError(f"Movie '{movie_title}' not found in dataset.")

        movie_idx = self.title_to_index[clean_title]

        # Retrieve pairwise similarity scores for the query movie
        sim_scores = self.similarity_matrix[movie_idx]

        # Pair each movie index with its similarity score
        scored_movies = list(enumerate(sim_scores))

        # Exclude the query movie itself
        filtered_movies = [pair for pair in scored_movies if pair[0] != movie_idx]

        # Sort candidates descending by similarity score
        sorted_movies = sorted(filtered_movies, key=lambda x: x[1], reverse=True)

        # Handle top_n bounds
        if top_n <= 0:
            return []

        top_candidates = sorted_movies[:top_n]

        # Format output recommendations
        recommendations = []
        for idx, score in top_candidates:
            rec_title = self.movies_df.iloc[idx]["title"]
            recommendations.append(
                {
                    "title": rec_title,
                    "similarity_score": round(float(score), 4),
                }
            )

        return recommendations

    def get_tfidf_matrix_shape(self) -> tuple[int, int]:
        """Returns the shape of the TF-IDF feature matrix."""
        return self.tfidf_matrix.shape

    def get_vocab_size(self) -> int:
        """Returns the size of the TF-IDF vocabulary."""
        return len(self.vectorizer.vocabulary_)

    def get_similarity_matrix_shape(self) -> tuple[int, int]:
        """Returns the shape of the Cosine Similarity matrix."""
        return self.similarity_matrix.shape


if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    processed_path = project_root / "data" / "processed" / "clean_movies.csv"

    if processed_path.exists():
        print(f"Loading processed dataset from: {processed_path}", flush=True)
        clean_df = pd.read_csv(processed_path)
    else:
        sys.path.insert(0, str(project_root))
        from src.data_loader import load_movies
        from src.preprocessor import preprocess_data

        raw_df = load_movies()
        clean_df = preprocess_data(raw_df)

    recommender = MovieRecommender(clean_df)
    print(f"\nTF-IDF Matrix Shape: {recommender.get_tfidf_matrix_shape()}", flush=True)
    print(f"Vocabulary Size: {recommender.get_vocab_size()}", flush=True)
    print(f"Similarity Matrix Shape: {recommender.get_similarity_matrix_shape()}", flush=True)

    query = "Avatar"
    recs = recommender.recommend(query, top_n=5)
    print(f"\nTop 5 Recommendations for '{query}':", flush=True)
    for idx, rec in enumerate(recs, 1):
        print(f"  {idx}. {rec['title']} — Score: {rec['similarity_score']}", flush=True)
