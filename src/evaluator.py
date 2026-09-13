import math
import numpy as np
from typing import Any, Sequence


def validate_k(k: Any) -> int:
    """
    Validates that k is a positive integer > 0.
    Explicitly rejects booleans (True/False), floats, non-integers, and values <= 0.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer > 0. Got '{k}' (type: {type(k).__name__}).")
    return k


def extract_titles(items: Sequence[str | dict[str, Any]]) -> list[str]:
    """
    Extracts, normalizes, and deduplicates movie title strings from a sequence of strings
    or recommendation dictionaries (e.g. [{'title': 'Avatar', 'personalized_score': 0.85}, ...]).

    Preserves top-rank order while removing duplicate occurrences.
    """
    if not isinstance(items, (list, tuple)):
        return []

    extracted = []
    seen_normalized = set()

    for item in items:
        if isinstance(item, str):
            title = item.strip()
        elif isinstance(item, dict) and "title" in item and isinstance(item["title"], str):
            title = item["title"].strip()
        else:
            continue

        if not title:
            continue

        norm_title = title.lower()
        if norm_title not in seen_normalized:
            seen_normalized.add(norm_title)
            extracted.append(title)

    return extracted


def precision_at_k(
    recommended: Sequence[str | dict[str, Any]],
    relevant: Sequence[str],
    k: int,
) -> float:
    """
    Calculates Precision@K for recommendation results against a ground-truth relevant set.

    Precision@K = (number of relevant items in top-K recommendations) / K

    Parameters
    ----------
    recommended : Sequence[str | dict]
        Recommended items (list of titles or recommendation dicts).
    relevant : Sequence[str]
        Ground-truth set/list of relevant movie titles.
    k : int
        Number of top recommendations to evaluate (k > 0).

    Returns
    -------
    float
        Precision score in [0.0, 1.0]. Returns 0.0 if recommended or relevant is empty.
    """
    valid_k = validate_k(k)

    rec_titles = extract_titles(recommended)
    rel_set = {t.strip().lower() for t in extract_titles(relevant)}

    if not rec_titles or not rel_set:
        return 0.0

    rec_at_k = rec_titles[:valid_k]
    hits = sum(1 for title in rec_at_k if title.lower() in rel_set)

    precision = hits / float(valid_k)
    return round(float(precision), 4)


def recall_at_k(
    recommended: Sequence[str | dict[str, Any]],
    relevant: Sequence[str],
    k: int,
) -> float:
    """
    Calculates Recall@K for recommendation results against a ground-truth relevant set.

    Recall@K = (number of relevant items in top-K recommendations) / (total relevant items)

    Parameters
    ----------
    recommended : Sequence[str | dict]
        Recommended items (list of titles or recommendation dicts).
    relevant : Sequence[str]
        Ground-truth set/list of relevant movie titles.
    k : int
        Number of top recommendations to evaluate (k > 0).

    Returns
    -------
    float
        Recall score in [0.0, 1.0]. Returns 0.0 if recommended or relevant is empty.
    """
    valid_k = validate_k(k)

    rec_titles = extract_titles(recommended)
    rel_set = {t.strip().lower() for t in extract_titles(relevant)}

    if not rec_titles or not rel_set:
        return 0.0

    rec_at_k = rec_titles[:valid_k]
    hits = sum(1 for title in rec_at_k if title.lower() in rel_set)

    recall = hits / float(len(rel_set))
    return round(float(recall), 4)


def ndcg_at_k(
    recommended: Sequence[str | dict[str, Any]],
    relevant: Sequence[str],
    k: int,
) -> float:
    """
    Calculates Normalized Discounted Cumulative Gain at K (NDCG@K) with binary relevance.

    DCG@K = sum_{i=1}^{min(K, |rec|)} (2^{rel_i} - 1) / log2(i + 1)
    IDCG@K = sum_{i=1}^{min(K, |relevant|)} 1.0 / log2(i + 1)
    NDCG@K = DCG@K / IDCG@K

    Parameters
    ----------
    recommended : Sequence[str | dict]
        Recommended items (list of titles or recommendation dicts).
    relevant : Sequence[str]
        Ground-truth set/list of relevant movie titles.
    k : int
        Number of top recommendations to evaluate (k > 0).

    Returns
    -------
    float
        NDCG score in [0.0, 1.0]. Returns 0.0 if IDCG@K == 0 (e.g. empty relevant set).
    """
    valid_k = validate_k(k)

    rec_titles = extract_titles(recommended)
    rel_set = {t.strip().lower() for t in extract_titles(relevant)}

    if not rec_titles or not rel_set:
        return 0.0

    rec_at_k = rec_titles[:valid_k]

    # Calculate DCG@K
    dcg = 0.0
    for idx, title in enumerate(rec_at_k, start=1):
        if title.lower() in rel_set:
            dcg += 1.0 / math.log2(idx + 1)

    # Calculate IDCG@K (Ideal ranking puts min(K, |rel|) relevant items at ranks 1..min(K, |rel|))
    ideal_hits = min(valid_k, len(rel_set))
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))

    if idcg == 0.0:
        return 0.0

    ndcg = dcg / idcg
    # Guarantee bounds within [0.0, 1.0]
    bounded_ndcg = min(1.0, max(0.0, ndcg))
    return round(float(bounded_ndcg), 4)


def evaluate_recommendations(
    recommended: Sequence[str | dict[str, Any]],
    relevant: Sequence[str],
    k: int,
) -> dict[str, float | int]:
    """
    Evaluates a single recommendation list against a ground-truth relevant set.

    Returns
    -------
    dict
        {'precision@k': float, 'recall@k': float, 'ndcg@k': float, 'k': int}
    """
    valid_k = validate_k(k)
    p_k = precision_at_k(recommended, relevant, valid_k)
    r_k = recall_at_k(recommended, relevant, valid_k)
    n_k = ndcg_at_k(recommended, relevant, valid_k)

    return {
        f"precision@{valid_k}": p_k,
        f"recall@{valid_k}": r_k,
        f"ndcg@{valid_k}": n_k,
        "k": valid_k,
    }


def evaluate_scenarios(
    recommender: Any,
    scenarios: list[dict[str, Any]],
    k: int,
) -> dict[str, Any]:
    """
    Evaluates multiple synthetic user preference scenarios using the provided recommender.

    Programmatically verifies NO DATA LEAKAGE:
        set(history_titles) & set(relevant_titles) == empty set

    Parameters
    ----------
    recommender : PersonalizedRecommender
        Fitted personalized recommendation engine.
    scenarios : list of dict
        List of scenarios: [{'name': str, 'user_history': list, 'relevant_movies': list}, ...]
    k : int
        Recommendation cut-off rank (k > 0).

    Returns
    -------
    dict
        Contains detailed scenario results and mean aggregate metrics.
    """
    valid_k = validate_k(k)

    if not isinstance(scenarios, (list, tuple)) or len(scenarios) == 0:
        raise ValueError("Scenarios list cannot be empty.")

    results = []

    for idx, scenario in enumerate(scenarios, 1):
        if not isinstance(scenario, dict):
            raise ValueError(f"Scenario at index {idx} must be a dictionary.")

        name = scenario.get("name", f"Scenario_{idx}")
        user_history = scenario.get("user_history", [])
        relevant_movies = scenario.get("relevant_movies", [])

        # Programmatically check for data leakage
        history_titles = {
            item[0].strip().lower()
            for item in user_history
            if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str)
        }
        relevant_titles = {title.strip().lower() for title in extract_titles(relevant_movies)}

        leakage = history_titles & relevant_titles
        if leakage:
            raise ValueError(
                f"Data leakage detected in scenario '{name}': item(s) {leakage} appear in both user history and held-out relevant set."
            )

        # Generate top-K recommendations using PersonalizedRecommender
        recs = recommender.recommend_for_user(user_history, top_n=valid_k)
        metrics = evaluate_recommendations(recs, relevant_movies, valid_k)

        rec_title_list = extract_titles(recs)
        scenario_result = {
            "name": name,
            "user_history": user_history,
            "relevant_movies": relevant_movies,
            "recommendations": rec_title_list,
            f"precision@{valid_k}": metrics[f"precision@{valid_k}"],
            f"recall@{valid_k}": metrics[f"recall@{valid_k}"],
            f"ndcg@{valid_k}": metrics[f"ndcg@{valid_k}"],
        }
        results.append(scenario_result)

    # Compute aggregate mean metrics
    mean_p = sum(res[f"precision@{valid_k}"] for res in results) / len(results)
    mean_r = sum(res[f"recall@{valid_k}"] for res in results) / len(results)
    mean_n = sum(res[f"ndcg@{valid_k}"] for res in results) / len(results)

    aggregate = {
        f"mean_precision@{valid_k}": round(float(mean_p), 4),
        f"mean_recall@{valid_k}": round(float(mean_r), 4),
        f"mean_ndcg@{valid_k}": round(float(mean_n), 4),
        "k": valid_k,
        "num_scenarios": len(results),
    }

    return {
        "scenarios": results,
        "aggregate": aggregate,
    }
