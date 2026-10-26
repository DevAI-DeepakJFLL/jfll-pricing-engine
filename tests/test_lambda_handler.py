"""
tests/test_lambda_handler.py
Unit tests for AWS Lambda WSGI handler error sanitization (Finding 1 / CWE-209).
"""

import json
import unittest
from unittest.mock import patch, MagicMock

import lambda_handler


class TestLambdaHandlerSecurity(unittest.TestCase):
    """Test that lambda handler responses do not leak tracebacks, paths, or runtime environments."""

    def test_handler_initialization_failure_sanitized(self):
        """When wsgi_handler fails to initialize, response must not leak sys.version or tracebacks."""
        with patch.object(lambda_handler, "_wsgi_handler", None):
            response = lambda_handler.handler({}, {})
            self.assertEqual(response["statusCode"], 500)
            self.assertEqual(response["headers"]["Content-Type"], "application/json")

            body = json.loads(response["body"])
            self.assertIn("error", body)
            # Verify no sensitive leakage
            self.assertNotIn("traceback", body)
            self.assertNotIn("python_version", body)
            self.assertNotIn("executable", body)
            self.assertNotIn("exception", body)
            self.assertIn("Service Unavailable", body["error"])

    def test_handler_unhandled_invocation_exception_sanitized(self):
        """When wsgi_handler raises an unhandled exception, response must return generic error."""
        mock_handler = MagicMock(side_effect=RuntimeError("/var/secret/db_credentials.json missing"))
        with patch.object(lambda_handler, "_wsgi_handler", mock_handler):
            response = lambda_handler.handler({}, {})
            self.assertEqual(response["statusCode"], 500)
            body = json.loads(response["body"])
            self.assertEqual(body.get("error"), "Internal Server Error")
            # Verify internal paths or exception text are not leaked
            self.assertNotIn("/var/secret", response["body"])
            self.assertNotIn("db_credentials", response["body"])


if __name__ == "__main__":
    unittest.main()
