"""
tests/test_security_integrity.py
Security verification test suite for cryptographic integrity and access control.
Tests:
- SHA-256 model verification and tamper detection (CWE-502)
- Fail-closed artifact verification when checksums.json is absent or missing entry
- Admin API authorization guard (CWE-306 / CWE-862) with fail-closed behavior
"""

import os
import json
import tempfile
import unittest
from flask import Flask
from src.security import (
    compute_sha256,
    verify_artifact_integrity,
    SecurityIntegrityError,
    require_admin_auth
)


class TestSecurityIntegrity(unittest.TestCase):
    """Test cryptographic model integrity checks and access control guards."""

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.artifact_path = os.path.join(self.test_dir.name, "sample_model.joblib")
        with open(self.artifact_path, "wb") as f:
            f.write(b"SAMPLE_MODEL_BINARY_PAYLOAD_TEST_DATA")

        self.valid_hash = compute_sha256(self.artifact_path)
        self.checksums_path = os.path.join(self.test_dir.name, "checksums.json")
        with open(self.checksums_path, "w", encoding="utf-8") as f:
            json.dump({"sample_model.joblib": self.valid_hash}, f)

    def tearDown(self):
        self.test_dir.cleanup()
        os.environ.pop("ADMIN_API_KEY", None)
        os.environ.pop("ALLOW_INSECURE_DEV", None)
        os.environ.pop("ALLOW_UNVERIFIED_ARTIFACTS", None)
        os.environ.pop("APP_ENV", None)

    def test_compute_sha256_deterministic(self):
        """Verifies SHA-256 calculation is deterministic and 64 hex characters."""
        digest1 = compute_sha256(self.artifact_path)
        digest2 = compute_sha256(self.artifact_path)
        self.assertEqual(digest1, digest2)
        self.assertEqual(len(digest1), 64)

    def test_verify_artifact_integrity_success(self):
        """Verifies clean artifact passes integrity check."""
        is_valid = verify_artifact_integrity(self.artifact_path, self.checksums_path)
        self.assertTrue(is_valid)

    def test_verify_artifact_integrity_tamper_detection(self):
        """Verifies tampered/corrupted artifact raises SecurityIntegrityError."""
        with open(self.artifact_path, "ab") as f:
            f.write(b"_MALICIOUS_INJECTION")

        with self.assertRaises(SecurityIntegrityError) as ctx:
            verify_artifact_integrity(self.artifact_path, self.checksums_path)
        self.assertIn("Cryptographic integrity violation", str(ctx.exception))

    def test_verify_artifact_integrity_missing_checksums_fails_closed(self):
        """Verifies missing checksums.json raises SecurityIntegrityError unless ALLOW_UNVERIFIED_ARTIFACTS=1."""
        nonexistent_checksums = os.path.join(self.test_dir.name, "nonexistent_checksums.json")
        os.environ.pop("ALLOW_UNVERIFIED_ARTIFACTS", None)

        with self.assertRaises(SecurityIntegrityError) as ctx:
            verify_artifact_integrity(self.artifact_path, nonexistent_checksums)
        self.assertIn("Checksum registry missing", str(ctx.exception))

        # With ALLOW_UNVERIFIED_ARTIFACTS=1, passes
        os.environ["ALLOW_UNVERIFIED_ARTIFACTS"] = "1"
        self.assertTrue(verify_artifact_integrity(self.artifact_path, nonexistent_checksums))

    def test_verify_artifact_integrity_missing_entry_fails_closed(self):
        """Verifies missing artifact entry in checksums.json raises SecurityIntegrityError unless ALLOW_UNVERIFIED_ARTIFACTS=1."""
        empty_checksums = os.path.join(self.test_dir.name, "empty_checksums.json")
        with open(empty_checksums, "w") as f:
            json.dump({}, f)
        os.environ.pop("ALLOW_UNVERIFIED_ARTIFACTS", None)

        with self.assertRaises(SecurityIntegrityError) as ctx:
            verify_artifact_integrity(self.artifact_path, empty_checksums)
        self.assertIn("not found in checksum registry", str(ctx.exception))

        os.environ["ALLOW_UNVERIFIED_ARTIFACTS"] = "1"
        self.assertTrue(verify_artifact_integrity(self.artifact_path, empty_checksums))

    def test_verify_production_checksums_match_real_artifacts(self):
        """Verifies all model artifacts listed in models/checksums.json match their actual hashes."""
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        checksums_file = os.path.join(repo_root, "models", "checksums.json")

        if not os.path.exists(checksums_file):
            self.skipTest("models/checksums.json not found")

        with open(checksums_file, "r") as f:
            checksums = json.load(f)

        for fname, expected_hash in checksums.items():
            fpath = os.path.join(repo_root, "models", fname)
            if os.path.exists(fpath):
                actual_hash = compute_sha256(fpath)
                self.assertEqual(
                    actual_hash,
                    expected_hash,
                    f"Model checksum mismatch for {fname}: actual={actual_hash}, expected={expected_hash}"
                )

    def test_require_admin_auth_fail_closed_and_headers(self):
        """Verifies require_admin_auth validates API keys, rejects query params, and fails closed."""
        app = Flask("test_auth_app")

        @app.route("/admin/test")
        @require_admin_auth
        def admin_test():
            return {"status": "ok"}

        client = app.test_client()

        # 1. When ADMIN_API_KEY is unset and ALLOW_INSECURE_DEV is not set -> 401
        os.environ.pop("ADMIN_API_KEY", None)
        os.environ.pop("ALLOW_INSECURE_DEV", None)
        resp = client.get("/admin/test")
        self.assertEqual(resp.status_code, 401)
        self.assertIn("unconfigured", resp.get_json()["message"])

        # 2. When ADMIN_API_KEY is unset and APP_ENV=production -> 503
        os.environ["APP_ENV"] = "production"
        resp = client.get("/admin/test")
        self.assertEqual(resp.status_code, 503)
        self.assertIn("production", resp.get_json()["message"])
        os.environ.pop("APP_ENV", None)

        # 3. When ADMIN_API_KEY is unset but ALLOW_INSECURE_DEV=1 -> 200
        os.environ["ALLOW_INSECURE_DEV"] = "1"
        resp = client.get("/admin/test")
        self.assertEqual(resp.status_code, 200)
        os.environ.pop("ALLOW_INSECURE_DEV", None)

        # 4. When ADMIN_API_KEY is configured
        os.environ["ADMIN_API_KEY"] = "secret_pass_token_xyz_123"
        try:
            # Missing key -> 401
            resp = client.get("/admin/test")
            self.assertEqual(resp.status_code, 401)

            # Incorrect key -> 401
            resp = client.get("/admin/test", headers={"X-API-Key": "wrong_token"})
            self.assertEqual(resp.status_code, 401)

            # Query param api_key must be strictly REJECTED (CWE-598) -> 401
            resp = client.get("/admin/test?api_key=secret_pass_token_xyz_123")
            self.assertEqual(resp.status_code, 401)
            self.assertIn("query parameters", resp.get_json()["message"])

            # Correct key via X-API-Key -> 200
            resp = client.get("/admin/test", headers={"X-API-Key": "secret_pass_token_xyz_123"})
            self.assertEqual(resp.status_code, 200)

            # Correct key via Authorization Bearer -> 200
            resp = client.get("/admin/test", headers={"Authorization": "Bearer secret_pass_token_xyz_123"})
            self.assertEqual(resp.status_code, 200)
        finally:
            os.environ.pop("ADMIN_API_KEY", None)


if __name__ == "__main__":
    unittest.main()
