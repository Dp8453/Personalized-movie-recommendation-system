"""
FastAPI REST API Serving Layer — Personalized Movie Recommendation System (Phase 9)

Exposes endpoints for:
- GET /health
- GET /recommend/content
- POST /recommend/personalized
- POST /recommend/hybrid
- POST /recommend/explain

Reuses existing Phase 1–8 recommendation engines and explanations without modifying underlying algorithms.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, List, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError, validator

from src.data_loader import load_movies
from src.preprocessor import preprocess_data
from src.recommender import MovieRecommender
from src.personalizer import PersonalizedRecommender
from src.collaborative_filter import ItemBasedCollaborativeRecommender
from src.hybrid_recommender import HybridMovieRecommender
from src.explanation import RecommendationExplainer


def sanitize_json(obj: Any) -> Any:
    """Recursively converts NumPy scalars and objects to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): sanitize_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, "item"):
        return obj.item()
    return obj


class ModelContainer:
    """Singleton container holding loaded DataFrames and recommender instances."""
    clean_df: Optional[pd.DataFrame] = None
    recommender: Optional[MovieRecommender] = None
    personalizer: Optional[PersonalizedRecommender] = None
    cf_recommender: Optional[ItemBasedCollaborativeRecommender] = None
    hybrid_recommender: Optional[HybridMovieRecommender] = None
    explainer: Optional[RecommendationExplainer] = None
    movielens_movies_df: Optional[pd.DataFrame] = None


models = ModelContainer()


def init_models() -> ModelContainer:
    """Loads datasets and initializes recommendation models ONCE at startup."""
    if models.recommender is not None:
        return models

    # 1. TMDB 5000 Movies
    processed_path = Path("data") / "processed" / "clean_movies.csv"
    if processed_path.exists():
        clean_df = pd.read_csv(processed_path)
    else:
        raw_df = load_movies()
        clean_df = preprocess_data(raw_df)

    models.clean_df = clean_df

    # 2. Phase 3 & 4 Content Engines
    recommender = MovieRecommender(clean_df, max_features=5000, stop_words="english")
    personalizer = PersonalizedRecommender(clean_df, recommender=recommender)

    models.recommender = recommender
    models.personalizer = personalizer

    # 3. Phase 6 & 7 Collaborative & Hybrid Engines
    ml_ratings_path = Path("data") / "raw" / "movielens" / "ml-latest-small" / "ratings.csv"
    ml_movies_path = Path("data") / "raw" / "movielens" / "ml-latest-small" / "movies.csv"

    if ml_ratings_path.exists() and ml_movies_path.exists():
        ratings_df = pd.read_csv(ml_ratings_path)
        movies_df = pd.read_csv(ml_movies_path)
        models.movielens_movies_df = movies_df

        cf_recommender = ItemBasedCollaborativeRecommender(
            min_ratings_per_movie=5, min_ratings_per_user=5, movies_df=movies_df
        )
        cf_recommender.fit(ratings_df)
        models.cf_recommender = cf_recommender

        hybrid_rec = HybridMovieRecommender(
            collaborative_recommender=cf_recommender,
            personalizer=personalizer,
            movies_df=movies_df,
            tmdb_df=clean_df,
        )
        hybrid_rec.fit(ratings_df)
        models.hybrid_recommender = hybrid_rec
    else:
        models.cf_recommender = None
        models.hybrid_recommender = None

    # 4. Phase 8 Explainability Engine
    explainer = RecommendationExplainer(
        hybrid_recommender=models.hybrid_recommender,
        collaborative_recommender=models.cf_recommender,
        personalizer=models.personalizer,
        recommender=models.recommender,
        movies_df=clean_df,
        movielens_movies_df=models.movielens_movies_df,
    )
    models.explainer = explainer

    return models


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI Lifespan Context Manager — triggers model loading on app startup."""
    init_models()
    yield


app = FastAPI(
    title="Personalized Movie Recommendation Serving API",
    description="Production-style REST API layer serving Content, Personalized, Collaborative, Hybrid, and Explainable recommendation engines.",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Exception Handlers ---

@app.exception_handler(ValueError)
def value_error_handler(request: Request, exc: ValueError):
    if isinstance(exc, ValidationError):
        raise exc
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "Invalid Request Data", "detail": str(exc)},
    )


@app.exception_handler(KeyError)
def key_error_handler(request: Request, exc: KeyError):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"error": "Resource Not Found", "detail": str(exc)},
    )


# --- Pydantic Request Models ---

try:
    from pydantic import BaseModel, Field, ValidationError, field_validator
except ImportError:
    from pydantic import BaseModel, Field, ValidationError, validator as field_validator  # type: ignore


class RatingItem(BaseModel):
    title: str = Field(..., description="Movie title string", min_length=1)
    rating: int = Field(..., description="Integer rating between 1 and 5", ge=1, le=5)

    @field_validator("rating", mode="before")
    @classmethod
    def check_rating_not_bool(cls, v):
        if isinstance(v, bool):
            raise ValueError("Rating must be an integer between 1 and 5, not boolean.")
        return v

    @field_validator("title")
    @classmethod
    def check_title_non_empty(cls, v):
        if not v or not str(v).strip():
            raise ValueError("Movie title cannot be empty or blank.")
        return str(v).strip()


class PersonalizedRecommendRequest(BaseModel):
    history: List[RatingItem] = Field(..., description="User rating history entries", min_length=1)
    top_n: int = Field(5, description="Number of top recommendations to return", gt=0)

    @field_validator("top_n", mode="before")
    @classmethod
    def check_top_n_not_bool(cls, v):
        if isinstance(v, bool):
            raise ValueError("top_n must be a positive integer > 0, not boolean.")
        return v

    @field_validator("history")
    @classmethod
    def check_no_duplicates(cls, v):
        if not v:
            raise ValueError("User rating history cannot be empty.")
        seen = set()
        for item in v:
            clean_t = item.title.strip().lower()
            if clean_t in seen:
                raise ValueError(f"Duplicate movie title '{item.title}' found in rating history.")
            seen.add(clean_t)
        return v


class HybridRecommendRequest(BaseModel):
    user_id: Optional[int] = Field(None, description="MovieLens user ID")
    history: Optional[List[RatingItem]] = Field(None, description="User rating history")
    alpha: float = Field(0.5, description="Hybrid fusion parameter in range [0.0, 1.0]", ge=0.0, le=1.0)
    top_n: int = Field(5, description="Number of recommendations to return", gt=0)

    @field_validator("alpha", mode="before")
    @classmethod
    def check_alpha(cls, v):
        if isinstance(v, bool):
            raise ValueError("alpha must be a float/int between 0.0 and 1.0, not boolean.")
        return v

    @field_validator("top_n", mode="before")
    @classmethod
    def check_top_n(cls, v):
        if isinstance(v, bool):
            raise ValueError("top_n must be an integer > 0, not boolean.")
        return v

    @field_validator("user_id", mode="before")
    @classmethod
    def check_user_id(cls, v):
        if isinstance(v, bool):
            raise ValueError("user_id must be an integer, not boolean.")
        return v


class ExplainRecommendRequest(BaseModel):
    user_id: Optional[int] = Field(None, description="MovieLens user ID")
    target_movie_id: Optional[int] = Field(None, description="MovieLens target movie ID")
    target_movie_title: Optional[str] = Field(None, description="Target movie title string")
    history: Optional[List[RatingItem]] = Field(None, description="User rating history")
    alpha: float = Field(0.5, description="Hybrid fusion weight", ge=0.0, le=1.0)

    @field_validator("alpha", mode="before")
    @classmethod
    def check_alpha(cls, v):
        if isinstance(v, bool):
            raise ValueError("alpha must be a float/int between 0.0 and 1.0, not boolean.")
        return v


# --- API Endpoints ---

@app.get("/health")
def health_check():
    """Service health check & model status endpoint."""
    m = init_models()
    return sanitize_json({
        "status": "ok",
        "dataset_movies": len(m.clean_df) if m.clean_df is not None else 0,
        "movielens_available": m.cf_recommender is not None,
        "features": 5000,
    })


@app.get("/recommend/content")
def recommend_content(
    title: str = Query(..., description="Query movie title", min_length=1),
    top_n: int = Query(5, description="Number of recommendations to return", gt=0),
):
    """Single-movie content-based recommendations using TF-IDF and Cosine Similarity."""
    if isinstance(top_n, bool) or top_n <= 0:
        raise HTTPException(status_code=400, detail="top_n must be a positive integer > 0.")
    if not title or not title.strip():
        raise HTTPException(status_code=400, detail="Title cannot be empty.")

    m = init_models()
    try:
        recs = m.recommender.recommend(movie_title=title, top_n=top_n)
        return sanitize_json({
            "mode": "content",
            "query": title.strip(),
            "recommendations": recs,
        })
    except ValueError as e:
        err_msg = str(e)
        if "not found in dataset" in err_msg.lower():
            raise HTTPException(status_code=404, detail=f"Movie '{title}' was not found in dataset.")
        raise HTTPException(status_code=400, detail=err_msg)


@app.post("/recommend/personalized")
def recommend_personalized(req: PersonalizedRecommendRequest):
    """User profile content recommendations using weighted rating history vector."""
    m = init_models()
    user_ratings = [(item.title, item.rating) for item in req.history]
    try:
        recs = m.personalizer.recommend_for_user(user_ratings, top_n=req.top_n)
        return sanitize_json({
            "mode": "personalized",
            "history_length": len(req.history),
            "recommendations": recs,
        })
    except ValueError as e:
        err_msg = str(e)
        if "not found in dataset" in err_msg.lower():
            raise HTTPException(status_code=404, detail=err_msg)
        raise HTTPException(status_code=400, detail=err_msg)


@app.post("/recommend/hybrid")
def recommend_hybrid(req: HybridRecommendRequest):
    """Hybrid content & CF recommendations over candidate pool union."""
    m = init_models()
    if m.hybrid_recommender is None:
        raise HTTPException(
            status_code=503,
            detail="MovieLens interaction dataset is unavailable for hybrid recommendations.",
        )

    if req.user_id is None and not req.history:
        raise HTTPException(
            status_code=400,
            detail="Must provide either 'user_id' or 'history' for hybrid recommendation.",
        )

    user_id = req.user_id if req.user_id is not None else 1
    try:
        recs = m.hybrid_recommender.recommend(user_id=user_id, alpha=req.alpha, top_n=req.top_n)
        return sanitize_json({
            "mode": "hybrid",
            "user_id": user_id,
            "alpha": req.alpha,
            "recommendations": recs,
        })
    except ValueError as e:
        err_msg = str(e)
        if "cold-start" in err_msg.lower() or "not found" in err_msg.lower():
            raise HTTPException(status_code=404, detail=err_msg)
        raise HTTPException(status_code=400, detail=err_msg)


@app.post("/recommend/explain")
def recommend_explain(req: ExplainRecommendRequest):
    """Explainable recommendation evidence breakdown (content, CF, hybrid, and summary)."""
    m = init_models()
    user_ratings_list = [(item.title, item.rating) for item in req.history] if req.history else None

    target_item: Any = None
    if req.target_movie_id is not None:
        target_item = req.target_movie_id
    elif req.target_movie_title is not None:
        target_item = req.target_movie_title
    else:
        raise HTTPException(
            status_code=400,
            detail="Must provide 'target_movie_id' or 'target_movie_title' for explanation.",
        )

    try:
        exp = m.explainer.explain_recommendation(
            target_item=target_item,
            user_id=req.user_id,
            user_ratings=user_ratings_list,
            alpha=req.alpha,
        )
        return sanitize_json({
            "mode": "explain",
            "explanation": exp,
        })
    except ValueError as e:
        err_msg = str(e)
        raise HTTPException(status_code=400, detail=err_msg)

