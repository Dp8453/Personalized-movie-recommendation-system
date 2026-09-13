import math
import re
from typing import Any, Sequence
import numpy as np
import pandas as pd

from src.evaluator import precision_at_k, recall_at_k, ndcg_at_k
from src.collaborative_filter import ItemBasedCollaborativeRecommender, temporal_train_test_split
from src.personalizer import PersonalizedRecommender


def validate_alpha(alpha: Any) -> float:
    """
    Validates that alpha is a finite numeric value in [0.0, 1.0].
    Explicitly rejects booleans (True/False), non-numeric strings, NaN, inf, and values outside [0.0, 1.0].
    """
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.number)):
        raise ValueError(f"alpha must be a numeric value in [0.0, 1.0]. Got '{alpha}' (type: {type(alpha).__name__}).")

    alpha_float = float(alpha)
    if math.isnan(alpha_float) or math.isinf(alpha_float):
        raise ValueError("alpha cannot be NaN or infinity.")

    if alpha_float < 0.0 or alpha_float > 1.0:
        raise ValueError(f"alpha must be in range [0.0, 1.0]. Got {alpha_float}.")

    return alpha_float


def normalize_scores(scores_dict: dict[int, float]) -> dict[int, float]:
    """
    Applies Min-Max score normalization to a dictionary of item scores:
        norm_score(c) = (score(c) - min_score) / (max_score - min_score)

    Returns 0.0 for all candidates if max_score == min_score, single candidate, or empty dict.
    Guarantees no NaN or infinity outputs.
    """
    if not isinstance(scores_dict, dict) or len(scores_dict) == 0:
        return {}

    values = list(scores_dict.values())
    min_val = min(values)
    max_val = max(values)

    val_range = max_val - min_val
    if val_range <= 0.0 or math.isnan(val_range) or math.isinf(val_range):
        return {k: 0.0 for k in scores_dict.keys()}

    normalized = {}
    for item_id, score in scores_dict.items():
        norm_val = (score - min_val) / val_range
        # Bound within [0.0, 1.0] to handle floating point precision
        bounded_val = min(1.0, max(0.0, float(norm_val)))
        normalized[item_id] = round(bounded_val, 6)

    return normalized


class HybridMovieRecommender:
    """
    Hybrid Movie Recommendation Engine.

    Composes Phase 4 Content Personalization and Phase 6 Item-Based Collaborative Filtering.
    Generates candidate pools by unioning top content and CF candidates, applies Min-Max
    score normalization, and computes weighted hybrid recommendation scores:
        HybridScore(c) = alpha * norm_content_score(c) + (1 - alpha) * norm_cf_score(c)
    """

    def __init__(
        self,
        collaborative_recommender: ItemBasedCollaborativeRecommender,
        personalizer: PersonalizedRecommender | None = None,
        movies_df: pd.DataFrame | None = None,
        tmdb_df: pd.DataFrame | None = None,
    ):
        """
        Initializes the HybridMovieRecommender.

        Parameters
        ----------
        collaborative_recommender : ItemBasedCollaborativeRecommender
            Phase 6 collaborative filtering engine.
        personalizer : PersonalizedRecommender, optional
            Phase 4 content-based personalization engine.
        movies_df : pd.DataFrame, optional
            MovieLens metadata DataFrame containing ['movieId', 'title'].
        tmdb_df : pd.DataFrame, optional
            TMDB metadata DataFrame containing clean title tags for content profile matching.
        """
        if not isinstance(collaborative_recommender, ItemBasedCollaborativeRecommender):
            raise ValueError("collaborative_recommender must be an instance of ItemBasedCollaborativeRecommender.")

        self.cf_recommender = collaborative_recommender
        self.personalizer = personalizer

        self.title_map: dict[int, str] = dict(self.cf_recommender.title_map)
        if movies_df is not None and "movieId" in movies_df.columns and "title" in movies_df.columns:
            for _, row in movies_df.iterrows():
                if pd.notna(row["movieId"]) and pd.notna(row["title"]):
                    self.title_map[int(row["movieId"])] = str(row["title"]).strip()

        # MovieLens movieId <-> TMDB clean title alignment map
        self.ml_id_to_clean_title: dict[int, str] = {}
        self.clean_title_to_ml_id: dict[str, int] = {}
        self.mapping_stats = {
            "total_movielens_movies": len(self.title_map),
            "mapped_movies": 0,
            "unmapped_movies": len(self.title_map),
            "coverage_percentage": 0.0,
        }

        if tmdb_df is not None and "title" in tmdb_df.columns:
            self._build_movie_identity_mapping(tmdb_df)

        self.is_fitted = False

    def _clean_title(self, raw_title: str) -> str:
        """Strips year in parentheses e.g. 'Toy Story (1995)' -> 'toy story'."""
        cleaned = re.sub(r"\s*\(\d{4}\)\s*$", "", str(raw_title))
        return cleaned.strip().lower()

    def _build_movie_identity_mapping(self, tmdb_df: pd.DataFrame) -> None:
        """
        Builds explicit, deterministic title-based mapping between MovieLens movieId and TMDB clean titles.
        """
        tmdb_clean_set = {str(t).strip().lower(): str(t).strip() for t in tmdb_df["title"] if pd.notna(t)}

        mapped_count = 0
        for ml_id, raw_title in self.title_map.items():
            norm_title = self._clean_title(raw_title)
            if norm_title in tmdb_clean_set:
                canonical_tmdb_title = tmdb_clean_set[norm_title]
                self.ml_id_to_clean_title[ml_id] = canonical_tmdb_title
                self.clean_title_to_ml_id[canonical_tmdb_title.lower()] = ml_id
                mapped_count += 1

        total = len(self.title_map)
        unmapped = total - mapped_count
        coverage = (mapped_count / float(total) * 100.0) if total > 0 else 0.0

        self.mapping_stats = {
            "total_movielens_movies": total,
            "mapped_movies": mapped_count,
            "unmapped_movies": unmapped,
            "coverage_percentage": round(coverage, 2),
        }

    def fit(self, ratings_df: pd.DataFrame) -> "HybridMovieRecommender":
        """
        Fits the underlying collaborative and content engines strictly on historical training ratings.
        """
        if not self.cf_recommender.is_fitted:
            self.cf_recommender.fit(ratings_df)

        self.is_fitted = True
        return self

    def _get_content_candidate_scores(self, user_id: int, top_n: int = 100) -> dict[int, float]:
        """
        Generates content-based candidate scores for MovieLens movies based on user's training history.
        """
        if user_id not in self.cf_recommender.user_history:
            return {}

        user_ratings = self.cf_recommender.user_history[user_id]
        if not user_ratings or self.personalizer is None:
            return {}

        # Map user's MovieLens ratings to (TMDB title, rating) tuples for Phase 4 personalizer
        mapped_history = []
        for m_id, rating in user_ratings.items():
            if m_id in self.ml_id_to_clean_title:
                clean_title = self.ml_id_to_clean_title[m_id]
                # Verify title exists in Phase 4 personalizer vocabulary
                if clean_title.lower() in self.personalizer.title_to_index:
                    mapped_history.append((clean_title, int(round(rating))))

        if not mapped_history:
            return {}

        # Generate content recommendations using Phase 4 personalizer
        try:
            content_recs = self.personalizer.recommend_for_user(mapped_history, top_n=top_n * 2)
        except Exception:
            return {}

        # Map returned TMDB content recommendations back to MovieLens movieIds
        content_scores = {}
        for rec in content_recs:
            rec_title = rec["title"].lower()
            if rec_title in self.clean_title_to_ml_id:
                ml_id = self.clean_title_to_ml_id[rec_title]
                content_scores[ml_id] = float(rec["personalized_score"])

        return content_scores

    def get_hybrid_scores(
        self,
        user_id: int,
        alpha: float = 0.5,
        candidate_size: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Generates hybrid recommendation scores over the union of content and CF candidate pools.
        """
        if not self.is_fitted:
            raise ValueError("Hybrid model must be fitted before generating scores.")

        valid_alpha = validate_alpha(alpha)

        if isinstance(user_id, bool) or not isinstance(user_id, (int, np.integer)):
            raise ValueError(f"user_id must be an integer. Got '{user_id}'.")
        user_id = int(user_id)

        if user_id not in self.cf_recommender.user_history:
            raise ValueError(f"User ID '{user_id}' not found in training dataset (cold-start user).")

        if isinstance(candidate_size, bool) or not isinstance(candidate_size, int) or candidate_size <= 0:
            raise ValueError("candidate_size must be a positive integer > 0.")

        user_rated_ids = set(self.cf_recommender.user_history[user_id].keys())

        # 1. Generate CF candidate pool
        cf_recs = self.cf_recommender.recommend(user_id, top_n=candidate_size, positive_threshold=4.0)
        cf_raw_scores = {r["movieId"]: float(r["collaborative_score"]) for r in cf_recs}

        # 2. Generate Content candidate pool
        content_raw_scores = self._get_content_candidate_scores(user_id, top_n=candidate_size)

        # 3. Candidate Pool Union: candidate_pool = content_candidates UNION cf_candidates
        candidate_pool = (set(cf_raw_scores.keys()) | set(content_raw_scores.keys())) - user_rated_ids

        if not candidate_pool:
            return []

        # 4. Extract raw candidate vectors (filling missing signals with 0.0)
        cand_content_raw = {m_id: content_raw_scores.get(m_id, 0.0) for m_id in candidate_pool}
        cand_cf_raw = {m_id: cf_raw_scores.get(m_id, 0.0) for m_id in candidate_pool}

        # 5. Min-Max Score Normalization
        norm_content = normalize_scores(cand_content_raw)
        norm_cf = normalize_scores(cand_cf_raw)

        # 6. Weighted Hybrid Scoring
        hybrid_candidates = []
        for m_id in candidate_pool:
            c_norm = norm_content.get(m_id, 0.0)
            cf_norm = norm_cf.get(m_id, 0.0)

            h_score = (valid_alpha * c_norm) + ((1.0 - valid_alpha) * cf_norm)
            title = self.title_map.get(m_id, f"Movie_{m_id}")

            hybrid_candidates.append(
                {
                    "movieId": m_id,
                    "title": title,
                    "hybrid_score": round(float(h_score), 6),
                    "norm_content_score": round(float(c_norm), 6),
                    "norm_cf_score": round(float(cf_norm), 6),
                }
            )

        # Deterministic sorting: hybrid_score descending, movieId ascending
        sorted_hybrid = sorted(hybrid_candidates, key=lambda x: (-x["hybrid_score"], x["movieId"]))
        return sorted_hybrid

    def recommend(
        self,
        user_id: int,
        alpha: float = 0.5,
        top_n: int = 10,
        candidate_size: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Generates top-N hybrid recommendations for a target user.
        """
        if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
            raise ValueError("top_n must be a positive integer > 0.")

        sorted_candidates = self.get_hybrid_scores(
            user_id=user_id,
            alpha=alpha,
            candidate_size=candidate_size,
        )

        # Cap at top_n
        top_recs = sorted_candidates[:top_n]
        return top_recs


def evaluate_hybrid_model(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    tmdb_df: pd.DataFrame | None = None,
    alpha_values: Sequence[float] = (0.0, 0.25, 0.50, 0.75, 1.0),
    k_values: Sequence[int] = (3, 5, 10),
    min_ratings_per_movie: int = 5,
    min_ratings_per_user: int = 5,
    min_user_ratings: int = 10,
    test_ratio: float = 0.2,
    positive_threshold: float = 4.0,
    candidate_size: int = 100,
    max_eval_users: int | None = 50,
) -> dict[str, Any]:
    """
    Evaluates the HybridMovieRecommender across multiple alpha values and K cutoffs
    using a leakage-safe temporal train/test split.
    """
    # 1. Chronological temporal train/test split
    train_ratings_df, test_scenarios = temporal_train_test_split(
        ratings_df,
        test_ratio=test_ratio,
        min_user_ratings=min_user_ratings,
        positive_threshold=positive_threshold,
    )

    # 2. FIT MODELS STRICTLY ON TRAINING DATA
    cf_recommender = ItemBasedCollaborativeRecommender(
        min_ratings_per_movie=min_ratings_per_movie,
        min_ratings_per_user=min_ratings_per_user,
        movies_df=movies_df,
    )

    personalizer = None
    if tmdb_df is not None:
        personalizer = PersonalizedRecommender(tmdb_df)

    hybrid_recommender = HybridMovieRecommender(
        collaborative_recommender=cf_recommender,
        personalizer=personalizer,
        movies_df=movies_df,
        tmdb_df=tmdb_df,
    )
    hybrid_recommender.fit(train_ratings_df)

    # Filter test scenarios to users present in training model
    eligible_scenarios = [
        sc for sc in test_scenarios if sc["userId"] in hybrid_recommender.cf_recommender.user_id_to_idx
    ]

    if max_eval_users is not None and max_eval_users > 0:
        eligible_scenarios = eligible_scenarios[:max_eval_users]

    if not eligible_scenarios:
        raise ValueError("No eligible evaluation users remaining after training split and filtering.")

    results_table = []

    for alpha in alpha_values:
        valid_alpha = validate_alpha(alpha)

        for k in k_values:
            precision_scores = []
            recall_scores = []
            ndcg_scores = []

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

                # Generate top-K hybrid recommendations
                recs = hybrid_recommender.recommend(
                    user_id=u_id,
                    alpha=valid_alpha,
                    top_n=k,
                    candidate_size=candidate_size,
                )
                rec_movie_ids = [r["movieId"] for r in recs]

                # Check 2: Rated training movies strictly excluded from recommendations
                rated_rec_overlap = set(train_history_ids) & set(rec_movie_ids)
                if rated_rec_overlap:
                    raise ValueError(
                        f"Data leakage detected for user {u_id}: rated training movies {rated_rec_overlap} appeared in recommendations."
                    )

                rec_str_list = [str(m_id) for m_id in rec_movie_ids]
                held_out_str_list = [str(m_id) for m_id in held_out_ids]

                p_k = precision_at_k(rec_str_list, held_out_str_list, k)
                r_k = recall_at_k(rec_str_list, held_out_str_list, k)
                n_k = ndcg_at_k(rec_str_list, held_out_str_list, k)

                precision_scores.append(p_k)
                recall_scores.append(r_k)
                ndcg_scores.append(n_k)

            n_eval = len(eligible_scenarios)
            results_table.append(
                {
                    "alpha": valid_alpha,
                    "k": k,
                    "num_users": n_eval,
                    "precision@k": round(float(np.mean(precision_scores)), 4),
                    "recall@k": round(float(np.mean(recall_scores)), 4),
                    "ndcg@k": round(float(np.mean(ndcg_scores)), 4),
                }
            )

    return {
        "mapping_stats": hybrid_recommender.mapping_stats,
        "eval_table": results_table,
        "num_evaluated_users": len(eligible_scenarios),
    }
