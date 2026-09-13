"""
Unit Tests for FastAPI REST API Serving Layer (Phase 9)

Tests GET /health, GET /recommend/content, POST /recommend/personalized,
POST /recommend/hybrid, and POST /recommend/explain endpoints, validation rules,
error handlers, and JSON response formatting using fastapi.testclient.TestClient.
"""

import unittest
from fastapi.testclient import TestClient
from src.api import app, init_models


class TestAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initializes model singletons and FastAPI TestClient once."""
        init_models()
        cls.client = TestClient(app)

    # 1. Health Check Endpoint
    def test_health_check_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("dataset_movies", data)
        self.assertIn("movielens_available", data)
        self.assertGreater(data["dataset_movies"], 0)

    # 2. Content Recommendation Success
    def test_recommend_content_success(self):
        response = self.client.get("/recommend/content?title=Avatar&top_n=3")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["mode"], "content")
        self.assertEqual(data["query"], "Avatar")
        self.assertEqual(len(data["recommendations"]), 3)
        for rec in data["recommendations"]:
            self.assertIn("title", rec)
            self.assertIn("similarity_score", rec)
            self.assertIsInstance(rec["similarity_score"], float)

    # 3. Content Recommendation Unknown Title Returns 404
    def test_recommend_content_unknown_title_404(self):
        response = self.client.get("/recommend/content?title=NonExistentMovieTitle99999&top_n=3")
        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("not found", data["detail"].lower())

    # 4. Content Recommendation Invalid top_n
    def test_recommend_content_invalid_top_n(self):
        response = self.client.get("/recommend/content?title=Avatar&top_n=0")
        self.assertIn(response.status_code, [400, 422])
        response_neg = self.client.get("/recommend/content?title=Avatar&top_n=-5")
        self.assertIn(response_neg.status_code, [400, 422])

    # 5. Content Recommendation Empty Title
    def test_recommend_content_empty_title(self):
        response = self.client.get("/recommend/content?title=&top_n=3")
        self.assertIn(response.status_code, [400, 422])

    # 6. Personalized Recommendation Success
    def test_recommend_personalized_success(self):
        payload = {
            "history": [
                {"title": "Avatar", "rating": 5},
                {"title": "Aliens", "rating": 4},
            ],
            "top_n": 4,
        }
        response = self.client.post("/recommend/personalized", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["mode"], "personalized")
        self.assertEqual(data["history_length"], 2)
        self.assertEqual(len(data["recommendations"]), 4)
        for rec in data["recommendations"]:
            self.assertIn("title", rec)
            self.assertIn("personalized_score", rec)
            self.assertIsInstance(rec["personalized_score"], float)

    # 7. Personalized Recommendation Empty History Returns 400/422
    def test_recommend_personalized_empty_history(self):
        payload = {"history": [], "top_n": 5}
        response = self.client.post("/recommend/personalized", json=payload)
        self.assertIn(response.status_code, [400, 422])

    # 8. Personalized Recommendation Invalid Rating (Out of Range or Boolean)
    def test_recommend_personalized_invalid_rating(self):
        payload_six = {
            "history": [{"title": "Avatar", "rating": 6}],
            "top_n": 5,
        }
        response_six = self.client.post("/recommend/personalized", json=payload_six)
        self.assertIn(response_six.status_code, [400, 422])

        payload_bool = {
            "history": [{"title": "Avatar", "rating": True}],
            "top_n": 5,
        }
        response_bool = self.client.post("/recommend/personalized", json=payload_bool)
        self.assertIn(response_bool.status_code, [400, 422])

    # 9. Personalized Recommendation Duplicate Titles Return 400
    def test_recommend_personalized_duplicate_titles(self):
        payload = {
            "history": [
                {"title": "Avatar", "rating": 5},
                {"title": "Avatar", "rating": 4},
            ],
            "top_n": 5,
        }
        response = self.client.post("/recommend/personalized", json=payload)
        self.assertIn(response.status_code, [400, 422])
        self.assertIn("Duplicate", str(response.json()))

    # 10. Personalized Recommendation Unknown Title Returns 404
    def test_recommend_personalized_unknown_title(self):
        payload = {
            "history": [{"title": "NonExistentMovie999", "rating": 5}],
            "top_n": 5,
        }
        response = self.client.post("/recommend/personalized", json=payload)
        self.assertEqual(response.status_code, 404)

    # 11. Hybrid Recommendation Success
    def test_recommend_hybrid_success(self):
        payload = {
            "user_id": 1,
            "alpha": 0.5,
            "top_n": 3,
        }
        response = self.client.post("/recommend/hybrid", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["mode"], "hybrid")
        self.assertEqual(data["user_id"], 1)
        self.assertEqual(data["alpha"], 0.5)
        self.assertEqual(len(data["recommendations"]), 3)
        for rec in data["recommendations"]:
            self.assertIn("movieId", rec)
            self.assertIn("title", rec)
            self.assertIn("hybrid_score", rec)

    # 12. Hybrid Recommendation Invalid Alpha
    def test_recommend_hybrid_invalid_alpha(self):
        payload_high = {"user_id": 1, "alpha": 1.5, "top_n": 5}
        response_high = self.client.post("/recommend/hybrid", json=payload_high)
        self.assertIn(response_high.status_code, [400, 422])

        payload_neg = {"user_id": 1, "alpha": -0.5, "top_n": 5}
        response_neg = self.client.post("/recommend/hybrid", json=payload_neg)
        self.assertIn(response_neg.status_code, [400, 422])

    # 13. Hybrid Recommendation Unknown User ID Returns 404
    def test_recommend_hybrid_unknown_user_id(self):
        payload = {"user_id": 999999, "alpha": 0.5, "top_n": 5}
        response = self.client.post("/recommend/hybrid", json=payload)
        self.assertEqual(response.status_code, 404)

    # 14. Hybrid Recommendation Missing Input Parameters
    def test_recommend_hybrid_missing_inputs(self):
        payload = {"alpha": 0.5, "top_n": 5}
        response = self.client.post("/recommend/hybrid", json=payload)
        self.assertEqual(response.status_code, 400)

    # 15. Explain Recommendation Success
    def test_recommend_explain_success(self):
        payload = {
            "user_id": 1,
            "target_movie_id": 2918,
            "alpha": 0.5,
        }
        response = self.client.post("/recommend/explain", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["mode"], "explain")
        exp = data["explanation"]
        self.assertIn("summary", exp)
        self.assertIn("hybrid_evidence", exp)

    # 16. Explain Recommendation Missing Target Item
    def test_recommend_explain_missing_target(self):
        payload = {"user_id": 1, "alpha": 0.5}
        response = self.client.post("/recommend/explain", json=payload)
        self.assertEqual(response.status_code, 400)

    # 17. Exception Handler Stack Trace Protection
    def test_exception_handler_no_stack_trace(self):
        response = self.client.get("/recommend/content?title=NonExistentTitleXYZ&top_n=5")
        self.assertEqual(response.status_code, 404)
        self.assertNotIn("Traceback", response.text)
        self.assertNotIn("File \"", response.text)

    # 18. Singleton Persistence Across Requests
    def test_singleton_persistence(self):
        m1 = init_models()
        m2 = init_models()
        self.assertIs(m1, m2)


if __name__ == "__main__":
    unittest.main()
