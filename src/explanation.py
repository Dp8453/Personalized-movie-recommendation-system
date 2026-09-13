import ast
import math
import re
from typing import Any, Sequence
import numpy as np
import pandas as pd

from src.collaborative_filter import ItemBasedCollaborativeRecommender
from src.hybrid_recommender import HybridMovieRecommender, normalize_scores, validate_alpha
from src.personalizer import PersonalizedRecommender
from src.recommender import MovieRecommender


def _parse_list_field(val: Any) -> list[str]:
    """
    Safely parses metadata fields that may be lists, JSON strings, Python literals,
    or comma-separated strings into a sorted list of clean strings.
    """
    if pd.isna(val) or val is None:
        return []

    if isinstance(val, (list, tuple, set)):
        items = [str(x).strip() for x in val if pd.notna(x) and str(x).strip()]
        return sorted(list(dict.fromkeys(items)))

    val_str = str(val).strip()
    if not val_str or val_str.lower() in ("nan", "none", "[]"):
        return []

    # Attempt ast.literal_eval or json parsing
    if val_str.startswith("[") and val_str.endswith("]"):
        try:
            parsed = ast.literal_eval(val_str)
            if isinstance(parsed, (list, tuple)):
                items = [str(x).strip() for x in parsed if pd.notna(x) and str(x).strip()]
                return sorted(list(dict.fromkeys(items)))
        except (ValueError, SyntaxError):
            pass

    # Fallback to comma-separated splitting
    items = [x.strip() for x in val_str.split(",") if x.strip()]
    return sorted(list(dict.fromkeys(items)))


def _parse_string_field(val: Any) -> str | None:
    """Safely parses single string metadata fields (e.g. director)."""
    if pd.isna(val) or val is None:
        return None
    val_str = str(val).strip()
    if not val_str or val_str.lower() in ("nan", "none"):
        return None
    return val_str


class RecommendationExplainer:
    """
    Explainable Recommendation Analysis Layer (Phase 8).

    Generates deterministic, evidence-based explanations for content-based,
    item-based collaborative filtering, and hybrid recommendations.
    Exposes actual metadata overlaps, collaborative item similarity contributions,
    and hybrid score decompositions without hallucinating unrepresented signals.
    """

    def __init__(
        self,
        hybrid_recommender: HybridMovieRecommender | None = None,
        collaborative_recommender: ItemBasedCollaborativeRecommender | None = None,
        personalizer: PersonalizedRecommender | None = None,
        recommender: MovieRecommender | None = None,
        movies_df: pd.DataFrame | None = None,
        movielens_movies_df: pd.DataFrame | None = None,
    ):
        """
        Initializes the RecommendationExplainer.

        Composes or references existing recommender modules and metadata DataFrames.
        """
        self.hybrid_recommender = hybrid_recommender
        self.collaborative_recommender = (
            collaborative_recommender
            if collaborative_recommender is not None
            else (hybrid_recommender.cf_recommender if hybrid_recommender is not None else None)
        )
        self.personalizer = (
            personalizer
            if personalizer is not None
            else (hybrid_recommender.personalizer if hybrid_recommender is not None else None)
        )
        self.recommender = (
            recommender
            if recommender is not None
            else (self.personalizer.recommender if self.personalizer is not None else None)
        )

        # Resolve TMDB movies DataFrame
        self.movies_df: pd.DataFrame | None = movies_df
        if self.movies_df is None and self.personalizer is not None:
            self.movies_df = self.personalizer.movies_df
        elif self.movies_df is None and self.recommender is not None:
            self.movies_df = self.recommender.movies_df

        # Resolve MovieLens movies DataFrame / title map
        self.movielens_movies_df: pd.DataFrame | None = movielens_movies_df
        self.title_map: dict[int, str] = {}
        if self.collaborative_recommender is not None:
            self.title_map.update(self.collaborative_recommender.title_map)
        if self.hybrid_recommender is not None:
            self.title_map.update(self.hybrid_recommender.title_map)

        # Build TMDB clean title index if movies_df available
        self.tmdb_title_to_idx: dict[str, int] = {}
        if self.movies_df is not None and "title" in self.movies_df.columns:
            for idx, title in enumerate(self.movies_df["title"]):
                clean_t = str(title).strip().lower()
                if clean_t not in self.tmdb_title_to_idx:
                    self.tmdb_title_to_idx[clean_t] = idx

    def _get_tmdb_movie_idx(self, movie_title: str) -> int | None:
        """Looks up index of movie in TMDB movies_df by title."""
        if not movie_title or self.movies_df is None:
            return None
        clean_t = str(movie_title).strip().lower()
        return self.tmdb_title_to_idx.get(clean_t, None)

    def _extract_movie_metadata(self, movie_idx: int) -> dict[str, Any]:
        """Extracts structured metadata for a movie in movies_df by index."""
        if self.movies_df is None or movie_idx < 0 or movie_idx >= len(self.movies_df):
            return {
                "title": "",
                "genres": [],
                "keywords": [],
                "director": None,
                "cast": [],
            }

        row = self.movies_df.iloc[movie_idx]
        title = str(row["title"]).strip() if "title" in row and pd.notna(row["title"]) else ""
        genres = _parse_list_field(row["genres"]) if "genres" in row else []
        keywords = _parse_list_field(row["keywords"]) if "keywords" in row else []
        director = _parse_string_field(row["director"]) if "director" in row else None
        cast = _parse_list_field(row["cast"]) if "cast" in row else []

        return {
            "title": title,
            "genres": genres,
            "keywords": keywords,
            "director": director,
            "cast": cast,
        }

    def explain_content_recommendation(
        self,
        target_movie_title: str,
        user_ratings: list[tuple[str, int]] | None = None,
        query_movie_title: str | None = None,
        top_n_terms: int = 5,
    ) -> dict[str, Any]:
        """
        Generates content-based explanation comparing target movie metadata
        against a query movie or user's positive preference history.
        """
        if isinstance(target_movie_title, bool) or not isinstance(target_movie_title, str) or not target_movie_title.strip():
            raise ValueError("target_movie_title must be a non-empty string.")

        target_idx = self._get_tmdb_movie_idx(target_movie_title)
        if target_idx is None:
            return {
                "available": False,
                "target_title": target_movie_title,
                "reason": "Target movie metadata not found in TMDB dataset.",
                "shared_genres": [],
                "shared_keywords": [],
                "shared_directors": [],
                "shared_cast": [],
                "top_tfidf_terms": [],
                "matching_sources": [],
            }

        target_meta = self._extract_movie_metadata(target_idx)
        canonical_target_title = target_meta["title"]

        # Collect source comparison targets (either from user_ratings positive movies or query_movie_title)
        comparison_sources: list[tuple[str, float]] = []

        if user_ratings is not None and len(user_ratings) > 0:
            # Extract positive user ratings (rating >= 4)
            for item in user_ratings:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    m_title, rating = item
                    if isinstance(rating, (int, float)) and rating >= 4:
                        comparison_sources.append((m_title, float(rating)))

        if query_movie_title is not None and query_movie_title.strip():
            comparison_sources.append((query_movie_title.strip(), 5.0))

        if not comparison_sources:
            return {
                "available": True,
                "target_title": canonical_target_title,
                "reason": "Target movie metadata available, but no positive preference sources provided for overlap comparison.",
                "shared_genres": target_meta["genres"],
                "shared_keywords": target_meta["keywords"],
                "shared_directors": [target_meta["director"]] if target_meta["director"] else [],
                "shared_cast": target_meta["cast"],
                "top_tfidf_terms": [],
                "matching_sources": [],
            }

        shared_genres_set: set[str] = set()
        shared_keywords_set: set[str] = set()
        shared_directors_set: set[str] = set()
        shared_cast_set: set[str] = set()
        matching_sources_list: list[dict[str, Any]] = []

        # Vectorizer term inspection setup
        vectorizer = None
        tfidf_matrix = None
        feature_names = None
        if self.recommender is not None:
            vectorizer = self.recommender.vectorizer
            tfidf_matrix = self.recommender.tfidf_matrix
            if vectorizer is not None and hasattr(vectorizer, "get_feature_names_out"):
                feature_names = vectorizer.get_feature_names_out()

        target_tfidf_vec = None
        if tfidf_matrix is not None and target_idx < tfidf_matrix.shape[0]:
            target_tfidf_vec = tfidf_matrix[target_idx].toarray().flatten()

        tfidf_term_scores: dict[str, float] = {}

        for src_title, src_rating in comparison_sources:
            src_idx = self._get_tmdb_movie_idx(src_title)
            if src_idx is None:
                continue

            src_meta = self._extract_movie_metadata(src_idx)
            canonical_src_title = src_meta["title"]

            common_g = sorted(list(set(target_meta["genres"]) & set(src_meta["genres"])))
            common_k = sorted(list(set(target_meta["keywords"]) & set(src_meta["keywords"])))
            common_d = [target_meta["director"]] if target_meta["director"] and target_meta["director"] == src_meta["director"] else []
            common_c = sorted(list(set(target_meta["cast"]) & set(src_meta["cast"])))

            shared_genres_set.update(common_g)
            shared_keywords_set.update(common_k)
            shared_directors_set.update(common_d)
            shared_cast_set.update(common_c)

            # Compute TF-IDF term overlap between target and source movie
            src_terms = []
            if target_tfidf_vec is not None and feature_names is not None and src_idx < tfidf_matrix.shape[0]:
                src_tfidf_vec = tfidf_matrix[src_idx].toarray().flatten()
                overlap_vec = target_tfidf_vec * src_tfidf_vec
                non_zero_indices = np.where(overlap_vec > 0.0)[0]
                if len(non_zero_indices) > 0:
                    term_pairs = [(feature_names[i], float(overlap_vec[i])) for i in non_zero_indices]
                    # Sort terms descending by contribution score, ascending by term string
                    sorted_terms = sorted(term_pairs, key=lambda x: (-x[1], x[0]))
                    for t_name, t_score in sorted_terms:
                        tfidf_term_scores[t_name] = max(tfidf_term_scores.get(t_name, 0.0), t_score)
                    src_terms = [t for t, _ in sorted_terms[:top_n_terms]]

            matching_sources_list.append(
                {
                    "source_title": canonical_src_title,
                    "user_rating": src_rating,
                    "shared_genres": common_g,
                    "shared_keywords": common_k,
                    "shared_director": common_d[0] if common_d else None,
                    "shared_cast": common_c,
                    "top_tfidf_terms": src_terms,
                }
            )

        # Sort top TF-IDF terms deterministically across all matching sources
        top_terms_sorted = []
        if tfidf_term_scores:
            sorted_global_terms = sorted(tfidf_term_scores.items(), key=lambda x: (-x[1], x[0]))
            top_terms_sorted = [t for t, _ in sorted_global_terms[:top_n_terms]]

        # Deterministic sorting for summary sets
        all_shared_genres = sorted(list(shared_genres_set))
        all_shared_keywords = sorted(list(shared_keywords_set))
        all_shared_directors = sorted(list(shared_directors_set))
        all_shared_cast = sorted(list(shared_cast_set))

        has_matches = (
            len(all_shared_genres) > 0
            or len(all_shared_keywords) > 0
            or len(all_shared_directors) > 0
            or len(all_shared_cast) > 0
            or len(top_terms_sorted) > 0
        )

        return {
            "available": True,
            "target_title": canonical_target_title,
            "has_overlaps": has_matches,
            "shared_genres": all_shared_genres,
            "shared_keywords": all_shared_keywords,
            "shared_directors": all_shared_directors,
            "shared_cast": all_shared_cast,
            "top_tfidf_terms": top_terms_sorted,
            "matching_sources": matching_sources_list,
        }

    def explain_cf_recommendation(
        self,
        user_id: int,
        target_movie_id: int,
        positive_threshold: float = 4.0,
        top_n_contributors: int = 5,
    ) -> dict[str, Any]:
        """
        Generates item-based collaborative filtering explanation identifying
        user's positively rated movies that contributed to target candidate CF score.
        """
        if isinstance(user_id, bool) or not isinstance(user_id, (int, np.integer)):
            raise ValueError(f"user_id must be an integer. Got '{user_id}'.")
        user_id = int(user_id)

        if isinstance(target_movie_id, bool) or not isinstance(target_movie_id, (int, np.integer)):
            raise ValueError(f"target_movie_id must be an integer. Got '{target_movie_id}'.")
        target_movie_id = int(target_movie_id)

        if isinstance(top_n_contributors, bool) or not isinstance(top_n_contributors, int) or top_n_contributors <= 0:
            raise ValueError("top_n_contributors must be a positive integer > 0.")

        target_title = self.title_map.get(target_movie_id, f"Movie_{target_movie_id}")

        if self.collaborative_recommender is None or not self.collaborative_recommender.is_fitted:
            return {
                "available": False,
                "user_id": user_id,
                "target_movie_id": target_movie_id,
                "target_title": target_title,
                "reason": "Collaborative filtering model is not fitted or unavailable.",
                "raw_cf_score": 0.0,
                "similar_rated_movies": [],
                "total_contributing_movies": 0,
            }

        cf = self.collaborative_recommender
        if user_id not in cf.user_history:
            return {
                "available": False,
                "user_id": user_id,
                "target_movie_id": target_movie_id,
                "target_title": target_title,
                "reason": f"User ID '{user_id}' not found in collaborative training history.",
                "raw_cf_score": 0.0,
                "similar_rated_movies": [],
                "total_contributing_movies": 0,
            }

        if target_movie_id not in cf.movie_id_to_idx:
            return {
                "available": False,
                "user_id": user_id,
                "target_movie_id": target_movie_id,
                "target_title": target_title,
                "reason": f"Movie ID '{target_movie_id}' not present in collaborative matrix.",
                "raw_cf_score": 0.0,
                "similar_rated_movies": [],
                "total_contributing_movies": 0,
            }

        cand_idx = cf.movie_id_to_idx[target_movie_id]
        user_ratings = cf.user_history[user_id]

        pos_rated = {m_id: rating for m_id, rating in user_ratings.items() if rating >= positive_threshold}
        if not pos_rated:
            return {
                "available": False,
                "user_id": user_id,
                "target_movie_id": target_movie_id,
                "target_title": target_title,
                "reason": f"User '{user_id}' has no positive ratings >= {positive_threshold}.",
                "raw_cf_score": 0.0,
                "similar_rated_movies": [],
                "total_contributing_movies": 0,
            }

        contributors = []
        total_cf_score = 0.0

        for pos_m_id, rating in pos_rated.items():
            if pos_m_id in cf.movie_id_to_idx:
                pos_idx = cf.movie_id_to_idx[pos_m_id]
                sim = float(cf.similarity_matrix[pos_idx, cand_idx])
                if sim > 0.0:
                    contrib = sim * float(rating)
                    total_cf_score += contrib
                    pos_title = self.title_map.get(pos_m_id, f"Movie_{pos_m_id}")
                    contributors.append(
                        {
                            "movieId": pos_m_id,
                            "title": pos_title,
                            "user_rating": float(rating),
                            "similarity": round(sim, 4),
                            "contribution": round(contrib, 4),
                        }
                    )

        # Deterministic sorting: contribution descending, movieId ascending
        sorted_contributors = sorted(contributors, key=lambda x: (-x["contribution"], x["movieId"]))
        top_contributors = sorted_contributors[:top_n_contributors]

        return {
            "available": len(contributors) > 0,
            "user_id": user_id,
            "target_movie_id": target_movie_id,
            "target_title": target_title,
            "raw_cf_score": round(float(total_cf_score), 4),
            "similar_rated_movies": top_contributors,
            "total_contributing_movies": len(contributors),
        }

    def explain_hybrid_recommendation(
        self,
        user_id: int,
        target_movie_id: int,
        alpha: float = 0.5,
        candidate_size: int = 100,
        positive_threshold: float = 4.0,
    ) -> dict[str, Any]:
        """
        Generates full hybrid recommendation explanation decomposing score contributions
        into normalized content and CF components, identifying dominant branch, and
        attaching metadata & collaborative evidence.
        """
        valid_alpha = validate_alpha(alpha)

        if isinstance(user_id, bool) or not isinstance(user_id, (int, np.integer)):
            raise ValueError(f"user_id must be an integer. Got '{user_id}'.")
        user_id = int(user_id)

        if isinstance(target_movie_id, bool) or not isinstance(target_movie_id, (int, np.integer)):
            raise ValueError(f"target_movie_id must be an integer. Got '{target_movie_id}'.")
        target_movie_id = int(target_movie_id)

        title = self.title_map.get(target_movie_id, f"Movie_{target_movie_id}")

        if self.hybrid_recommender is None or not self.hybrid_recommender.is_fitted:
            return {
                "available": False,
                "movieId": target_movie_id,
                "title": title,
                "user_id": user_id,
                "reason": "Hybrid recommendation engine is not fitted or unavailable.",
                "summary": "Explanation unavailable because hybrid model is not fitted.",
            }

        hybrid = self.hybrid_recommender

        if user_id not in hybrid.cf_recommender.user_history:
            return {
                "available": False,
                "movieId": target_movie_id,
                "title": title,
                "user_id": user_id,
                "reason": f"User ID '{user_id}' not found in training dataset (cold-start user).",
                "summary": f"Explanation unavailable: User ID '{user_id}' has no historical interaction data.",
            }

        # Obtain hybrid candidate pool scores
        hybrid_scores = hybrid.get_hybrid_scores(
            user_id=user_id,
            alpha=valid_alpha,
            candidate_size=candidate_size,
        )

        cand_info = next((c for c in hybrid_scores if c["movieId"] == target_movie_id), None)

        if cand_info is not None:
            norm_content_score = float(cand_info["norm_content_score"])
            norm_cf_score = float(cand_info["norm_cf_score"])
            hybrid_score = float(cand_info["hybrid_score"])
        else:
            # Fallback calculation if item not in candidate union top_n
            norm_content_score = 0.0
            norm_cf_score = 0.0
            hybrid_score = 0.0

        weighted_content_contrib = round(valid_alpha * norm_content_score, 6)
        weighted_cf_contrib = round((1.0 - valid_alpha) * norm_cf_score, 6)

        # Determine dominant branch
        if valid_alpha == 1.0:
            dominant_branch = "content_only"
        elif valid_alpha == 0.0:
            dominant_branch = "cf_only"
        elif weighted_content_contrib > weighted_cf_contrib + 1e-5:
            dominant_branch = "content"
        elif weighted_cf_contrib > weighted_content_contrib + 1e-5:
            dominant_branch = "collaborative"
        else:
            dominant_branch = "balanced"

        # Generate Content Evidence
        # Convert user's training history in MovieLens to TMDB clean titles
        user_ratings_tmdb: list[tuple[str, int]] = []
        user_hist = hybrid.cf_recommender.user_history.get(user_id, {})
        for m_id, r_val in user_hist.items():
            clean_t = hybrid.ml_id_to_clean_title.get(m_id, None)
            if clean_t is not None:
                user_ratings_tmdb.append((clean_t, int(round(r_val))))

        tmdb_title_for_target = hybrid.ml_id_to_clean_title.get(target_movie_id, title)
        content_ev = self.explain_content_recommendation(
            target_movie_title=tmdb_title_for_target,
            user_ratings=user_ratings_tmdb,
        )

        # Generate Collaborative Evidence
        cf_ev = self.explain_cf_recommendation(
            user_id=user_id,
            target_movie_id=target_movie_id,
            positive_threshold=positive_threshold,
        )

        # Build Human-Readable Explanation Summary
        summary_str = self._build_explanation_summary(
            target_title=title,
            dominant_branch=dominant_branch,
            alpha=valid_alpha,
            weighted_content_contrib=weighted_content_contrib,
            weighted_cf_contrib=weighted_cf_contrib,
            content_ev=content_ev,
            cf_ev=cf_ev,
        )

        return {
            "available": True,
            "movieId": target_movie_id,
            "title": title,
            "user_id": user_id,
            "content_evidence": content_ev,
            "cf_evidence": cf_ev,
            "hybrid_evidence": {
                "alpha": valid_alpha,
                "normalized_content_score": norm_content_score,
                "normalized_cf_score": norm_cf_score,
                "weighted_content_contribution": weighted_content_contrib,
                "weighted_cf_contribution": weighted_cf_contrib,
                "hybrid_score": hybrid_score,
                "dominant_branch": dominant_branch,
            },
            "summary": summary_str,
        }

    def _build_explanation_summary(
        self,
        target_title: str,
        dominant_branch: str,
        alpha: float,
        weighted_content_contrib: float,
        weighted_cf_contrib: float,
        content_ev: dict[str, Any],
        cf_ev: dict[str, Any],
    ) -> str:
        """
        Generates factual, human-readable summary string strictly from actual evidence.
        """
        parts = [f"'{target_title}' was recommended"]

        content_avail = content_ev.get("available", False) and content_ev.get("has_overlaps", False)
        cf_avail = cf_ev.get("available", False) and len(cf_ev.get("similar_rated_movies", [])) > 0

        if dominant_branch == "content_only":
            parts.append(f"by content-only hybrid scoring (alpha={alpha:.2f})")
            if content_avail:
                genres = ", ".join(content_ev["shared_genres"][:3])
                parts.append(f"due to shared metadata ({genres}) with movies you rated highly")
            else:
                parts.append("based on content profile similarity")
        elif dominant_branch == "cf_only":
            parts.append(f"by CF-only hybrid scoring (alpha={alpha:.2f})")
            if cf_avail:
                top_cf = cf_ev["similar_rated_movies"][0]["title"]
                parts.append(f"due to collaborative similarity to '{top_cf}'")
            else:
                parts.append("based on collaborative interaction similarity")
        elif dominant_branch == "content":
            parts.append(f"primarily by the content model (content weight: {weighted_content_contrib:.4f} vs CF weight: {weighted_cf_contrib:.4f})")
            if content_avail:
                genres = ", ".join(content_ev["shared_genres"][:3])
                parts.append(f"sharing {genres} characteristics with movies in your history")
            if cf_avail:
                top_cf = cf_ev["similar_rated_movies"][0]["title"]
                parts.append(f", supported by collaborative similarity to '{top_cf}'")
        elif dominant_branch == "collaborative":
            parts.append(f"primarily by the collaborative model (CF weight: {weighted_cf_contrib:.4f} vs content weight: {weighted_content_contrib:.4f})")
            if cf_avail:
                top_cf = cf_ev["similar_rated_movies"][0]["title"]
                parts.append(f"due to collaborative similarity to '{top_cf}'")
            if content_avail:
                genres = ", ".join(content_ev["shared_genres"][:3])
                parts.append(f", supported by content metadata ({genres})")
        else:  # balanced
            parts.append(f"with balanced contributions from content ({weighted_content_contrib:.4f}) and collaborative ({weighted_cf_contrib:.4f}) models")
            if cf_avail:
                top_cf = cf_ev["similar_rated_movies"][0]["title"]
                parts.append(f", featuring collaborative similarity to '{top_cf}'")

        if not content_avail and not cf_avail:
            parts.append(". Specific feature overlap details were limited.")
        else:
            parts.append(".")

        return " ".join(parts)

    def explain_recommendation(
        self,
        target_item: dict[str, Any] | int | str,
        user_id: int | None = None,
        user_ratings: list[tuple[str, int]] | None = None,
        query_title: str | None = None,
        alpha: float = 0.5,
        candidate_size: int = 100,
    ) -> dict[str, Any]:
        """
        Unified explanation entry point routing automatically to hybrid, CF, or content explanation.
        """
        target_movie_id = None
        target_movie_title = None

        if isinstance(target_item, dict):
            target_movie_id = target_item.get("movieId", None)
            target_movie_title = target_item.get("title", None)
        elif isinstance(target_item, (int, np.integer)):
            target_movie_id = int(target_item)
            target_movie_title = self.title_map.get(target_movie_id, None)
        elif isinstance(target_item, str):
            target_movie_title = target_item.strip()

        # Hybrid routing
        if self.hybrid_recommender is not None and user_id is not None and target_movie_id is not None:
            return self.explain_hybrid_recommendation(
                user_id=user_id,
                target_movie_id=target_movie_id,
                alpha=alpha,
                candidate_size=candidate_size,
            )

        # CF routing
        if self.collaborative_recommender is not None and user_id is not None and target_movie_id is not None:
            return self.explain_cf_recommendation(
                user_id=user_id,
                target_movie_id=target_movie_id,
            )

        # Content routing
        if target_movie_title is not None:
            return self.explain_content_recommendation(
                target_movie_title=target_movie_title,
                user_ratings=user_ratings,
                query_movie_title=query_title,
            )

        raise ValueError("Insufficient parameters to generate recommendation explanation.")
