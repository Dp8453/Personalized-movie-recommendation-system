import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure src module resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.recommender import MovieRecommender


class PersonalizedRecommender:
    """
    Personalized Content-Based Movie Recommendation Engine.

    Constructs a weighted User Preference Profile vector from a user's movie rating history:
        u = sum(w_i * v_i) / sum(abs(w_i))

    where w_i is the preference weight for rated movie i, and v_i is its TF-IDF vector.
    Compares the user profile vector against all candidate movie vectors via Cosine Similarity.
    """

    RATING_WEIGHT_MAP = {
        1: -1.0,  # Strongly disliked
        2: -0.5,  # Disliked
        3: 0.0,   # Neutral
        4: 0.5,   # Liked
        5: 1.0,   # Strongly liked
    }

    def __init__(
        self,
        movies_df: pd.DataFrame,
        recommender: MovieRecommender | None = None,
        max_features: int = 5000,
        stop_words: str | list[str] | None = "english",
    ):
        """
        Initializes the PersonalizedRecommender engine.

        Reuses or fits the Phase 3 TF-IDF feature space (TfidfVectorizer).
        """
        if recommender is not None:
            self.recommender = recommender
            self.movies_df = recommender.movies_df
            self.vectorizer = recommender.vectorizer
            self.tfidf_matrix = recommender.tfidf_matrix
            self.title_to_index = recommender.title_to_index
        else:
            self.recommender = MovieRecommender(
                movies_df=movies_df,
                max_features=max_features,
                stop_words=stop_words,
            )
            self.movies_df = self.recommender.movies_df
            self.vectorizer = self.recommender.vectorizer
            self.tfidf_matrix = self.recommender.tfidf_matrix
            self.title_to_index = self.recommender.title_to_index

    def validate_user_ratings(self, user_ratings: list[tuple[str, int]]) -> list[tuple[str, int, int, float]]:
        """
        Validates user rating history structure, rating values, duplicate titles, and movie existence.

        Returns
        -------
        list of tuple: [(title, rating, movie_idx, preference_weight), ...]
        """
        if not isinstance(user_ratings, (list, tuple)) or len(user_ratings) == 0:
            raise ValueError("User rating history cannot be empty.")

        seen_titles = set()
        validated_entries = []

        for item in user_ratings:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError(f"Invalid rating entry '{item}'. Expected (title, rating) tuple.")

            movie_title, rating = item

            # Validate title
            if not isinstance(movie_title, str) or not movie_title.strip():
                raise ValueError(f"Invalid movie title '{movie_title}'. Must be a non-empty string.")

            clean_title = movie_title.strip().lower()

            # Check duplicate movie titles in user history
            if clean_title in seen_titles:
                raise ValueError(f"Duplicate movie title '{movie_title}' found in user rating history.")
            seen_titles.add(clean_title)

            # Check title existence in dataset
            if clean_title not in self.title_to_index:
                raise ValueError(f"Movie '{movie_title}' not found in dataset.")

            movie_idx = self.title_to_index[clean_title]

            # Validate rating (must be an integer in 1..5, excluding booleans and floats)
            if isinstance(rating, bool) or not isinstance(rating, int) or rating not in self.RATING_WEIGHT_MAP:
                raise ValueError(
                    f"Invalid rating '{rating}' for movie '{movie_title}'. Ratings must be integers in 1..5."
                )

            pref_weight = self.RATING_WEIGHT_MAP[rating]
            canonical_title = self.movies_df.iloc[movie_idx]["title"]
            validated_entries.append((canonical_title, rating, movie_idx, pref_weight))

        return validated_entries

    def build_user_profile(self, user_ratings: list[tuple[str, int]]) -> np.ndarray:
        """
        Builds the weighted user preference profile vector:
            u = sum(w_i * v_i) / sum(abs(w_i))

        Raises
        ------
        ValueError
            If sum(abs(w_i)) == 0 (i.e. all ratings are neutral 3s).
        """
        validated_entries = self.validate_user_ratings(user_ratings)

        sum_abs_weights = sum(abs(entry[3]) for entry in validated_entries)

        if sum_abs_weights == 0.0:
            raise ValueError(
                "User profile cannot be built because no positive or negative preference signal was provided."
            )

        # Initialize zero vector matching TF-IDF feature dimension
        num_features = self.tfidf_matrix.shape[1]
        weighted_vector_sum = np.zeros((1, num_features), dtype=np.float64)

        for _, _, movie_idx, pref_weight in validated_entries:
            if pref_weight != 0.0:
                movie_vec = self.tfidf_matrix[movie_idx].toarray()
                weighted_vector_sum += pref_weight * movie_vec

        # Normalize by sum of absolute preference weights
        user_profile = weighted_vector_sum / sum_abs_weights
        return user_profile

    def recommend_for_user(
        self, user_ratings: list[tuple[str, int]], top_n: int = 5
    ) -> list[dict[str, str | float]]:
        """
        Generates personalized movie recommendations for a user based on their rating history.

        Parameters
        ----------
        user_ratings : list of tuple
            User's movie rating history, e.g. [("Avatar", 5), ("Titanic", 1)].
        top_n : int, default=5
            Number of top personalized recommendations to return.

        Returns
        -------
        list of dict
            Formatted recommendations: [{'title': str, 'personalized_score': float}, ...]
        """
        if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
            raise ValueError("top_n must be a positive integer > 0.")

        # Build user profile vector (shape: 1 x num_features)
        user_profile = self.build_user_profile(user_ratings)

        # Efficiently compute Cosine Similarity between single user profile vector and all movie vectors
        # user_profile: (1, num_features), tfidf_matrix: (num_movies, num_features) -> sim_scores: (1, num_movies)
        sim_scores = cosine_similarity(user_profile, self.tfidf_matrix)[0]

        # Extract rated movie indices for strict exclusion
        validated_entries = self.validate_user_ratings(user_ratings)
        rated_indices = {entry[2] for entry in validated_entries}

        # Pair candidates with score and exclude rated movies
        scored_candidates = [
            (idx, score) for idx, score in enumerate(sim_scores) if idx not in rated_indices
        ]

        # Sort candidate movies descending by personalized score
        sorted_candidates = sorted(scored_candidates, key=lambda x: x[1], reverse=True)

        # Cap recommendations at top_n or available candidate count
        top_candidates = sorted_candidates[:top_n]

        recommendations = []
        for idx, score in top_candidates:
            rec_title = self.movies_df.iloc[idx]["title"]
            recommendations.append(
                {
                    "title": rec_title,
                    "personalized_score": round(float(score), 4),
                }
            )

        return recommendations


if __name__ == "__main__":
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    processed_path = project_root / "data" / "processed" / "clean_movies.csv"

    if processed_path.exists():
        print(f"Loading processed dataset from: {processed_path}", flush=True)
        clean_df = pd.read_csv(processed_path)
    else:
        from src.data_loader import load_movies
        from src.preprocessor import preprocess_data

        raw_df = load_movies()
        clean_df = preprocess_data(raw_df)

    personalizer = PersonalizedRecommender(clean_df)

    sample_history = [
        ("Avatar", 5),
        ("Aliens", 5),
        ("The Dark Knight", 4),
        ("Titanic", 1),
    ]

    print(f"\nUser Rating History: {sample_history}", flush=True)
    recs = personalizer.recommend_for_user(sample_history, top_n=5)
    print("\nTop 5 Personalized Recommendations:", flush=True)
    for idx, rec in enumerate(recs, 1):
        print(f"  {idx}. {rec['title']:35s} | Personalized Score: {rec['personalized_score']}", flush=True)
