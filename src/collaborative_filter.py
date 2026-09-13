import math
from typing import Any, Sequence
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from src.evaluator import precision_at_k, recall_at_k, ndcg_at_k, evaluate_recommendations


class ItemBasedCollaborativeRecommender:
    """
    Item-Based Collaborative Filtering Recommendation Engine.

    Builds a sparse Item x User rating matrix from user-movie interaction histories
    and computes sparse Cosine Similarity between item rating vectors.
    Generates personalized recommendations by accumulating similarity-weighted
    ratings over a user's positive preference history.
    """

    def __init__(
        self,
        min_ratings_per_movie: int = 5,
        min_ratings_per_user: int = 5,
        movies_df: pd.DataFrame | None = None,
    ):
        """
        Initializes the ItemBasedCollaborativeRecommender engine.

        Parameters
        ----------
        min_ratings_per_movie : int, default=5
            Minimum number of ratings a movie must have to be retained during filtering.
        min_ratings_per_user : int, default=5
            Minimum number of ratings a user must have to be retained during filtering.
        movies_df : pd.DataFrame, optional
            Optional metadata DataFrame containing ['movieId', 'title'] for title resolution.
        """
        if isinstance(min_ratings_per_movie, bool) or not isinstance(min_ratings_per_movie, int) or min_ratings_per_movie < 0:
            raise ValueError("min_ratings_per_movie must be a non-negative integer >= 0.")
        if isinstance(min_ratings_per_user, bool) or not isinstance(min_ratings_per_user, int) or min_ratings_per_user < 0:
            raise ValueError("min_ratings_per_user must be a non-negative integer >= 0.")

        self.min_ratings_per_movie = min_ratings_per_movie
        self.min_ratings_per_user = min_ratings_per_user

        self.title_map: dict[int, str] = {}
        if movies_df is not None:
            self._build_title_map(movies_df)

        self.is_fitted = False
        self.movie_id_to_idx: dict[int, int] = {}
        self.idx_to_movie_id: dict[int, int] = {}
        self.user_id_to_idx: dict[int, int] = {}
        self.idx_to_user_id: dict[int, int] = {}

        self.item_user_matrix: csr_matrix | None = None
        self.similarity_matrix: np.ndarray | csr_matrix | None = None
        self.user_history: dict[int, dict[int, float]] = {}

        # Filtering statistics
        self.stats = {
            "raw_users": 0,
            "raw_movies": 0,
            "raw_ratings": 0,
            "filtered_users": 0,
            "filtered_movies": 0,
            "filtered_ratings": 0,
            "sparsity": 0.0,
        }

    def _build_title_map(self, movies_df: pd.DataFrame) -> None:
        if not isinstance(movies_df, pd.DataFrame):
            raise ValueError("movies_df must be a pandas DataFrame.")
        if "movieId" not in movies_df.columns or "title" not in movies_df.columns:
            raise ValueError("movies_df must contain 'movieId' and 'title' columns.")
        for _, row in movies_df.iterrows():
            m_id = row["movieId"]
            title = row["title"]
            if pd.notna(m_id) and pd.notna(title):
                self.title_map[int(m_id)] = str(title).strip()

    def validate_ratings_df(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates structure, non-nullity, numeric rating scale [0.5, 5.0], and deduplicates (latest timestamp wins).
        """
        if not isinstance(ratings_df, pd.DataFrame):
            raise ValueError("ratings_df must be a pandas DataFrame.")

        required_cols = {"userId", "movieId", "rating"}
        if not required_cols.issubset(ratings_df.columns):
            missing = required_cols - set(ratings_df.columns)
            raise ValueError(f"ratings_df missing required column(s): {missing}")

        if len(ratings_df) == 0:
            raise ValueError("ratings_df cannot be empty.")

        # Check null values
        if ratings_df[["userId", "movieId", "rating"]].isnull().any().any():
            raise ValueError("ratings_df contains null values in userId, movieId, or rating.")

        # Check numeric ratings
        for val in ratings_df["rating"]:
            if isinstance(val, bool) or not isinstance(val, (int, float, np.number)):
                raise ValueError(f"Invalid non-numeric rating value: '{val}'.")
            if val < 0.5 or val > 5.0:
                raise ValueError(f"Invalid rating value '{val}'. Ratings must be within scale [0.5, 5.0].")

        # Deterministic deduplication of (userId, movieId)
        df_copy = ratings_df.copy()
        df_copy["userId"] = df_copy["userId"].astype(int)
        df_copy["movieId"] = df_copy["movieId"].astype(int)
        df_copy["rating"] = df_copy["rating"].astype(float)

        if "timestamp" in df_copy.columns:
            # Sort by timestamp ascending so latest timestamp wins during drop_duplicates(keep='last')
            df_copy = df_copy.sort_values(by=["userId", "movieId", "timestamp"], ascending=[True, True, True])

        dedup_df = df_copy.drop_duplicates(subset=["userId", "movieId"], keep="last").reset_index(drop=True)
        return dedup_df

    def fit(self, ratings_df: pd.DataFrame) -> "ItemBasedCollaborativeRecommender":
        """
        Fits the item-based collaborative model on interaction DataFrame.
        """
        clean_ratings = self.validate_ratings_df(ratings_df)

        raw_users = clean_ratings["userId"].nunique()
        raw_movies = clean_ratings["movieId"].nunique()
        raw_ratings = len(clean_ratings)

        filtered_ratings = clean_ratings.copy()

        # Apply iterative minimum rating thresholds for movies and users
        if self.min_ratings_per_movie > 0:
            movie_counts = filtered_ratings["movieId"].value_counts()
            valid_movies = movie_counts[movie_counts >= self.min_ratings_per_movie].index
            filtered_ratings = filtered_ratings[filtered_ratings["movieId"].isin(valid_movies)]

        if self.min_ratings_per_user > 0:
            user_counts = filtered_ratings["userId"].value_counts()
            valid_users = user_counts[user_counts >= self.min_ratings_per_user].index
            filtered_ratings = filtered_ratings[filtered_ratings["userId"].isin(valid_users)]

        if len(filtered_ratings) == 0:
            raise ValueError("All ratings were excluded by min_ratings thresholds.")

        # Re-index unique sorted movie IDs and user IDs deterministically
        unique_movie_ids = sorted(filtered_ratings["movieId"].unique())
        unique_user_ids = sorted(filtered_ratings["userId"].unique())

        self.movie_id_to_idx = {m_id: idx for idx, m_id in enumerate(unique_movie_ids)}
        self.idx_to_movie_id = {idx: m_id for idx, m_id in enumerate(unique_movie_ids)}
        self.user_id_to_idx = {u_id: idx for idx, u_id in enumerate(unique_user_ids)}
        self.idx_to_user_id = {idx: u_id for idx, u_id in enumerate(unique_user_ids)}

        num_movies = len(unique_movie_ids)
        num_users = len(unique_user_ids)

        row_indices = [self.movie_id_to_idx[m_id] for m_id in filtered_ratings["movieId"]]
        col_indices = [self.user_id_to_idx[u_id] for u_id in filtered_ratings["userId"]]
        values = filtered_ratings["rating"].values.astype(np.float64)

        # Construct Item x User sparse matrix R (num_movies x num_users)
        self.item_user_matrix = csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(num_movies, num_users),
            dtype=np.float64,
        )

        # Compute Item-Item Cosine Similarity matrix S (num_movies x num_movies)
        self.similarity_matrix = cosine_similarity(self.item_user_matrix, dense_output=True)

        # Zero out diagonal to strictly exclude self-similarity
        np.fill_diagonal(self.similarity_matrix, 0.0)

        # Store complete user rating history for fast candidate filtering
        self.user_history = {}
        for u_id, group in filtered_ratings.groupby("userId"):
            self.user_history[int(u_id)] = {
                int(row["movieId"]): float(row["rating"]) for _, row in group.iterrows()
            }

        # Calculate statistics
        total_possible = num_movies * num_users
        sparsity = 1.0 - (len(filtered_ratings) / float(total_possible)) if total_possible > 0 else 0.0

        self.stats = {
            "raw_users": raw_users,
            "raw_movies": raw_movies,
            "raw_ratings": raw_ratings,
            "filtered_users": num_users,
            "filtered_movies": num_movies,
            "filtered_ratings": len(filtered_ratings),
            "sparsity": round(float(sparsity), 4),
        }

        self.is_fitted = True
        return self

    def get_similar_items(self, movie_id: int, top_n: int = 10) -> list[dict[str, Any]]:
        """
        Retrieves top-N most similar movies based on item-item collaborative cosine similarity.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling get_similar_items().")

        if isinstance(movie_id, bool) or not isinstance(movie_id, (int, np.integer)):
            raise ValueError(f"movie_id must be an integer. Got '{movie_id}'.")
        movie_id = int(movie_id)

        if movie_id not in self.movie_id_to_idx:
            raise ValueError(f"Movie ID '{movie_id}' not found in training dataset.")

        if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
            raise ValueError("top_n must be a positive integer > 0.")

        m_idx = self.movie_id_to_idx[movie_id]
        sim_scores = self.similarity_matrix[m_idx]

        candidates = []
        for cand_idx, score in enumerate(sim_scores):
            cand_movie_id = self.idx_to_movie_id[cand_idx]
            # Strictly exclude self-item
            if cand_movie_id == movie_id:
                continue
            if score > 0.0:
                candidates.append((cand_movie_id, float(score)))

        # Deterministic sorting: similarity score descending, movieId ascending
        sorted_candidates = sorted(candidates, key=lambda x: (-x[1], x[0]))
        top_candidates = sorted_candidates[:top_n]

        results = []
        for cand_m_id, score in top_candidates:
            title = self.title_map.get(cand_m_id, f"Movie_{cand_m_id}")
            results.append(
                {
                    "movieId": cand_m_id,
                    "title": title,
                    "similarity_score": round(score, 4),
                }
            )

        return results

    def recommend(
        self,
        user_id: int,
        top_n: int = 10,
        positive_threshold: float = 4.0,
    ) -> list[dict[str, Any]]:
        """
        Generates personalized collaborative recommendations for a target user.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling recommend().")

        if isinstance(user_id, bool) or not isinstance(user_id, (int, np.integer)):
            raise ValueError(f"user_id must be an integer. Got '{user_id}'.")
        user_id = int(user_id)

        if user_id not in self.user_history:
            raise ValueError(f"User ID '{user_id}' not found in training dataset (cold-start user).")

        if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
            raise ValueError("top_n must be a positive integer > 0.")

        if isinstance(positive_threshold, bool) or not isinstance(positive_threshold, (int, float)):
            raise ValueError("positive_threshold must be a numeric value.")

        user_ratings = self.user_history[user_id]
        rated_movie_ids = set(user_ratings.keys())

        # Select positively rated movies
        pos_rated_movies = {
            m_id: rating for m_id, rating in user_ratings.items() if rating >= positive_threshold
        }

        if not pos_rated_movies:
            return []

        # Vectorized candidate score accumulation: score = pos_ratings @ pos_sim_matrix
        pos_indices = []
        pos_ratings_list = []
        for pos_m_id, rating in pos_rated_movies.items():
            if pos_m_id in self.movie_id_to_idx:
                pos_indices.append(self.movie_id_to_idx[pos_m_id])
                pos_ratings_list.append(rating)

        if not pos_indices:
            return []

        pos_ratings_vec = np.array(pos_ratings_list, dtype=np.float64)
        pos_sim_matrix = self.similarity_matrix[pos_indices, :]
        raw_scores = pos_ratings_vec @ pos_sim_matrix

        # Zero out scores for all already-rated movies and non-positive candidates
        rated_indices = [
            self.movie_id_to_idx[m_id]
            for m_id in rated_movie_ids
            if m_id in self.movie_id_to_idx
        ]
        if rated_indices:
            raw_scores[rated_indices] = 0.0

        cand_indices = np.where(raw_scores > 0.0)[0]
        if len(cand_indices) == 0:
            return []

        candidates = [
            (self.idx_to_movie_id[idx], float(raw_scores[idx]))
            for idx in cand_indices
        ]

        # Sort candidates deterministically: collaborative score descending, movieId ascending
        sorted_candidates = sorted(candidates, key=lambda x: (-x[1], x[0]))
        top_candidates = sorted_candidates[:top_n]

        results = []
        for cand_m_id, score in top_candidates:
            title = self.title_map.get(cand_m_id, f"Movie_{cand_m_id}")
            results.append(
                {
                    "movieId": cand_m_id,
                    "title": title,
                    "collaborative_score": round(float(score), 4),
                }
            )

        return results


def temporal_train_test_split(
    ratings_df: pd.DataFrame,
    test_ratio: float = 0.2,
    min_user_ratings: int = 10,
    positive_threshold: float = 4.0,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Performs a deterministic, chronological temporal train/test split per user.

    Parameters
    ----------
    ratings_df : pd.DataFrame
        Complete user rating interaction DataFrame.
    test_ratio : float, default=0.2
        Proportion of latest user ratings reserved for held-out testing.
    min_user_ratings : int, default=10
        Minimum historical ratings a user must have to be eligible for evaluation.
    positive_threshold : float, default=4.0
        Minimum rating value for held-out test items to be considered relevant ground truth.

    Returns
    -------
    train_ratings_df : pd.DataFrame
        Training ratings DataFrame containing only historical training interactions.
    test_scenarios : list of dict
        Evaluation user scenarios with held-out positive target items.
    """
    if "timestamp" not in ratings_df.columns:
        raise ValueError("temporal_train_test_split requires a 'timestamp' column in ratings_df.")

    # Sort chronologically with secondary sort by movieId
    sorted_df = ratings_df.sort_values(
        by=["userId", "timestamp", "movieId"], ascending=[True, True, True]
    ).reset_index(drop=True)

    train_rows = []
    test_scenarios = []

    for u_id, group in sorted_df.groupby("userId"):
        n_total = len(group)
        if n_total < min_user_ratings:
            # Include all ratings of under-threshold users in training data to preserve matrix density
            train_rows.append(group)
            continue

        n_test = max(1, int(math.ceil(n_total * test_ratio)))
        n_train = n_total - n_test

        user_train = group.iloc[:n_train]
        user_test = group.iloc[n_train:]

        train_rows.append(user_train)

        # Held-out positive targets (rating >= positive_threshold)
        test_pos_movies = user_test[user_test["rating"] >= positive_threshold]["movieId"].tolist()

        if test_pos_movies:
            train_history_ids = user_train["movieId"].tolist()
            test_scenarios.append(
                {
                    "userId": int(u_id),
                    "train_history_ids": train_history_ids,
                    "held_out_relevant_ids": test_pos_movies,
                }
            )

    train_ratings_df = pd.concat(train_rows, ignore_index=True)
    return train_ratings_df, test_scenarios


def evaluate_collaborative_model(
    ratings_df: pd.DataFrame,
    k_values: Sequence[int] = (3, 5, 10),
    min_ratings_per_movie: int = 5,
    min_ratings_per_user: int = 5,
    min_user_ratings: int = 10,
    test_ratio: float = 0.2,
    positive_threshold: float = 4.0,
    max_eval_users: int | None = None,
    movies_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Evaluates the ItemBasedCollaborativeRecommender using a leakage-safe temporal train/test split.

    Model fitting is performed STRICTLY on the training split before recommendations are generated.
    """
    # 1. Chronological temporal split
    train_ratings_df, test_scenarios = temporal_train_test_split(
        ratings_df,
        test_ratio=test_ratio,
        min_user_ratings=min_user_ratings,
        positive_threshold=positive_threshold,
    )

    # 2. FIT MODEL STRICTLY ON TRAINING DATA
    recommender = ItemBasedCollaborativeRecommender(
        min_ratings_per_movie=min_ratings_per_movie,
        min_ratings_per_user=min_ratings_per_user,
        movies_df=movies_df,
    )
    recommender.fit(train_ratings_df)

    # Filter test scenarios to users present in the fitted model
    eligible_scenarios = [
        sc for sc in test_scenarios if sc["userId"] in recommender.user_id_to_idx
    ]

    if max_eval_users is not None and max_eval_users > 0:
        eligible_scenarios = eligible_scenarios[:max_eval_users]

    if not eligible_scenarios:
        raise ValueError("No eligible evaluation users remaining after training split and filtering.")

    results_by_k = {}

    for k in k_values:
        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        total_held_out_items = 0

        for sc in eligible_scenarios:
            u_id = sc["userId"]
            train_history_ids = sc["train_history_ids"]
            held_out_ids = sc["held_out_relevant_ids"]

            # PROGRAMMATIC LEAKAGE CHECKS
            # Check 1: Held-out items absent from user training history
            overlap = set(train_history_ids) & set(held_out_ids)
            if overlap:
                raise ValueError(
                    f"Data leakage detected for user {u_id}: held-out items {overlap} appear in training history."
                )

            # Generate top-K recommendations
            recs = recommender.recommend(u_id, top_n=k, positive_threshold=positive_threshold)
            rec_movie_ids = [r["movieId"] for r in recs]

            # Check 2: Rated training movies strictly excluded from recommendations
            rated_rec_overlap = set(train_history_ids) & set(rec_movie_ids)
            if rated_rec_overlap:
                raise ValueError(
                    f"Data leakage detected for user {u_id}: rated training movies {rated_rec_overlap} appeared in recommendations."
                )

            # Convert IDs to string representation for Phase 5 evaluator functions
            rec_str_list = [str(m_id) for m_id in rec_movie_ids]
            held_out_str_list = [str(m_id) for m_id in held_out_ids]

            p_k = precision_at_k(rec_str_list, held_out_str_list, k)
            r_k = recall_at_k(rec_str_list, held_out_str_list, k)
            n_k = ndcg_at_k(rec_str_list, held_out_str_list, k)

            precision_scores.append(p_k)
            recall_scores.append(r_k)
            ndcg_scores.append(n_k)
            total_held_out_items += len(held_out_ids)

        n_eval = len(eligible_scenarios)
        results_by_k[k] = {
            "k": k,
            "num_users": n_eval,
            "total_held_out_items": total_held_out_items,
            "mean_precision": round(float(np.mean(precision_scores)), 4),
            "mean_recall": round(float(np.mean(recall_scores)), 4),
            "mean_ndcg": round(float(np.mean(ndcg_scores)), 4),
        }

    return {
        "model_stats": recommender.stats,
        "eval_by_k": results_by_k,
        "num_evaluated_users": len(eligible_scenarios),
    }
