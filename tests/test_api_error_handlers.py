"""
tests/test_api_error_handlers.py
Tests for content-negotiated uniform error handlers in src/app.py.
"""

import unittest
from src.app import app


class TestApiErrorHandlers(unittest.TestCase):
    """Test suite for uniform JSON error responses on API endpoints and HTML on web pages."""

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_api_404_returns_json(self):
        """Requesting nonexistent /api/* endpoint returns structured JSON 404."""
        res = self.client.get("/api/unknown_route_xyz")
        self.assertEqual(res.status_code, 404)
        self.assertTrue(res.is_json)
        data = res.get_json()
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["error_code"], 404)
        self.assertIn("not found", data["message"].lower())

    def test_browser_404_returns_html(self):
        """Browser request (Accept: text/html) to non-API route returns HTML."""
        res = self.client.get("/nonexistent_page", headers={"Accept": "text/html,application/xhtml+xml"})
        self.assertEqual(res.status_code, 404)
        self.assertIn("text/html", res.content_type)
        self.assertIn("404", res.get_data(as_text=True))

    def test_api_405_method_not_allowed_returns_json(self):
        """GET request on POST-only /api/history/outcome returns structured JSON 405."""
        res = self.client.get("/api/history/outcome")
        self.assertEqual(res.status_code, 405)
        self.assertTrue(res.is_json)
        data = res.get_json()
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["error_code"], 405)
        self.assertIn("method", data["message"].lower())

    def test_api_400_malformed_json_returns_json(self):
        """Sending malformed JSON to an API endpoint returns structured JSON 400."""
        res = self.client.post(
            "/api/quote",
            data="{not_valid_json: 123",
            headers={"Content-Type": "application/json"}
        )
        self.assertEqual(res.status_code, 400)
        self.assertTrue(res.is_json)
        data = res.get_json()
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["error_code"], 400)

    def test_api_413_payload_too_large_returns_json(self):
        """Sending payload exceeding MAX_CONTENT_LENGTH (1MB) returns structured JSON 413."""
        # 1.2 MB oversized payload
        oversized = "x" * (1024 * 1024 + 200 * 1024)
        res = self.client.post(
            "/api/quote",
            data=oversized,
            headers={"Content-Type": "application/json"}
        )
        self.assertEqual(res.status_code, 413)
        self.assertTrue(res.is_json)
        data = res.get_json()
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["error_code"], 413)
        self.assertIn("payload", data["message"].lower())


if __name__ == "__main__":
    unittest.main()
