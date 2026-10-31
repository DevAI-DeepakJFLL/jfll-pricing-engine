"""
tests/test_rate_limiter.py
Unit tests for thread-safe, sliding-window rate limiting decorator in src/security.py.
"""

import time
import unittest
from flask import Flask, jsonify, request
from src.security import (
    rate_limit,
    reset_rate_limits,
    get_client_ip,
)


class TestRateLimiter(unittest.TestCase):
    """Test suite for rate limiting behavior, headers, IP resolution, and memory safety."""

    def setUp(self):
        reset_rate_limits()
        self.app = Flask(__name__)

        @self.app.route("/test-limited")
        @rate_limit(max_requests=3, window_seconds=2)
        def limited_route():
            return jsonify({"status": "ok", "message": "allowed"})

        @self.app.route("/test-ip")
        def ip_route():
            return jsonify({"ip": get_client_ip()})

        self.client = self.app.test_client()

    def tearDown(self):
        reset_rate_limits()

    def test_requests_within_limit_succeed(self):
        """Allows requests within the threshold (3 requests allowed)."""
        for i in range(3):
            res = self.client.get("/test-limited")
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.get_json()["status"], "ok")

    def test_requests_exceeding_limit_blocked_with_429(self):
        """Fourth request within 2 seconds must be blocked with HTTP 429 and Retry-After header."""
        for _ in range(3):
            res = self.client.get("/test-limited")
            self.assertEqual(res.status_code, 200)

        blocked = self.client.get("/test-limited")
        self.assertEqual(blocked.status_code, 429)
        data = blocked.get_json()
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["error_code"], 429)
        self.assertIn("Rate limit exceeded", data["message"])
        self.assertIn("Retry-After", blocked.headers)
        self.assertGreaterEqual(int(blocked.headers["Retry-After"]), 1)

    def test_rate_limit_resets_after_window_expires(self):
        """After window_seconds elapses, client can send requests again."""
        for _ in range(3):
            self.client.get("/test-limited")

        blocked = self.client.get("/test-limited")
        self.assertEqual(blocked.status_code, 429)

        # Wait for the 2-second window to expire
        time.sleep(2.1)

        allowed_again = self.client.get("/test-limited")
        self.assertEqual(allowed_again.status_code, 200)

    def test_get_client_ip_default_and_proxy(self):
        """Verifies remote_addr is used by default; X-Forwarded-For is only used when BEHIND_PROXY=1."""
        import os

        # Default: uses remote_addr, ignores spoofed X-Forwarded-For
        res = self.client.get("/test-ip", headers={"X-Forwarded-For": "203.0.113.195"})
        self.assertEqual(res.get_json()["ip"], "127.0.0.1")

        # When BEHIND_PROXY=1, uses client IP from X-Forwarded-For
        os.environ["BEHIND_PROXY"] = "1"
        try:
            res_proxy = self.client.get("/test-ip", headers={"X-Forwarded-For": "198.51.100.42, 10.0.0.1"})
            self.assertEqual(res_proxy.get_json()["ip"], "198.51.100.42")
        finally:
            os.environ.pop("BEHIND_PROXY", None)

    def test_rate_limit_disabled_via_environment_flag(self):
        """When RATE_LIMIT_ENABLED=0, rate limiting is bypassed."""
        import os

        os.environ["RATE_LIMIT_ENABLED"] = "0"
        try:
            for _ in range(6):
                res = self.client.get("/test-limited")
                self.assertEqual(res.status_code, 200)
        finally:
            os.environ.pop("RATE_LIMIT_ENABLED", None)


if __name__ == "__main__":
    unittest.main()
