"""
src/security.py
Security utilities for Air Export Pricing Engine.
Includes:
- SHA-256 cryptographic model verification to defend against untrusted deserialization (CWE-502).
  Enforces fail-closed behavior if checksums.json is missing or unverified unless ALLOW_UNVERIFIED_ARTIFACTS=1.
- Admin API authorization decorator for sensitive routes (CWE-306 / CWE-862).
  Fails closed when unconfigured (HTTP 503 in prod, HTTP 401 in dev unless ALLOW_INSECURE_DEV=1).
  Rejects query parameter tokens (header-only: X-API-Key or Authorization Bearer).
"""

import os
import json
import hmac
import hashlib
import logging
from functools import wraps
from typing import Optional
from flask import request, jsonify

logger = logging.getLogger("pricing_security")


class SecurityIntegrityError(RuntimeError):
    """Raised when an artifact fails cryptographic integrity checks."""


def compute_sha256(filepath: str) -> str:
    """Computes SHA-256 hex digest for a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_artifact_integrity(filepath: str, checksums_path: Optional[str] = None) -> bool:
    """
    Verifies that a model artifact matches its known SHA-256 digest in checksums.json.
    Fails closed: raises SecurityIntegrityError if checksums.json is missing or does
    not contain the artifact, unless ALLOW_UNVERIFIED_ARTIFACTS=1.
    """
    if not os.path.exists(filepath):
        return False

    allow_unverified = os.environ.get("ALLOW_UNVERIFIED_ARTIFACTS") == "1"

    if checksums_path is None:
        model_dir = os.path.dirname(os.path.abspath(filepath))
        checksums_path = os.path.join(model_dir, "checksums.json")

    if not os.path.exists(checksums_path):
        if allow_unverified:
            logger.warning(
                f"Checksum registry not found at {checksums_path}, but ALLOW_UNVERIFIED_ARTIFACTS=1. "
                "Allowing unverified artifact."
            )
            return True
        raise SecurityIntegrityError(
            f"Cryptographic integrity violation: Checksum registry missing ({checksums_path}). "
            "Refusing to load unverified artifact in fail-closed mode. "
            "Set ALLOW_UNVERIFIED_ARTIFACTS=1 to override in development."
        )

    try:
        with open(checksums_path, "r", encoding="utf-8") as f:
            checksums = json.load(f)
    except Exception as e:
        if allow_unverified:
            logger.warning(f"Could not read checksum registry {checksums_path}: {e}")
            return True
        raise SecurityIntegrityError(
            f"Cryptographic integrity violation: Failed to parse checksum registry {checksums_path}: {e}"
        )

    fname = os.path.basename(filepath)
    expected = checksums.get(fname)
    if not expected:
        if allow_unverified:
            logger.warning(
                f"No checksum entry for {fname} in {checksums_path}, allowing because ALLOW_UNVERIFIED_ARTIFACTS=1."
            )
            return True
        raise SecurityIntegrityError(
            f"Cryptographic integrity violation (CWE-502): Artifact {fname} not found in checksum registry ({checksums_path})."
        )

    actual = compute_sha256(filepath)
    if actual != expected:
        raise SecurityIntegrityError(
            f"Cryptographic integrity violation (CWE-502): Artifact {fname} hash mismatch! "
            f"Expected {expected}, got {actual}."
        )

    logger.info(f"Verified cryptographic SHA-256 integrity for {fname} [{actual[:12]}...]")
    return True


def require_admin_auth(f):
    """
    Guards administrative or sensitive export endpoints (CWE-306 / CWE-862).
    Fail-closed authorization guard:
    - If ADMIN_API_KEY is not set:
      - In production (APP_ENV=production or FLASK_ENV=production): returns HTTP 503 Service Unavailable.
      - In development/test: returns HTTP 401 Unauthorized unless ALLOW_INSECURE_DEV=1 is explicitly set.
    - If ADMIN_API_KEY is set:
      - Only accepts 'X-API-Key' header or 'Authorization: Bearer <key>' header.
      - Query parameter 'api_key' is strictly rejected to prevent URI token leakage in logs.
      - Uses constant-time comparison (hmac.compare_digest).
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        active_key = os.environ.get("ADMIN_API_KEY")
        is_production = os.environ.get("APP_ENV") == "production" or os.environ.get("FLASK_ENV") == "production"

        if not active_key:
            if is_production:
                logger.error(f"Access denied to {request.path}: ADMIN_API_KEY is unconfigured in production.")
                return jsonify({
                    "status": "error",
                    "message": "Admin API key unconfigured in production environment."
                }), 503

            if os.environ.get("ALLOW_INSECURE_DEV") == "1":
                logger.warning(f"Admin endpoint {request.path} accessed without ADMIN_API_KEY (ALLOW_INSECURE_DEV=1).")
                return f(*args, **kwargs)

            logger.warning(
                f"Unauthorized access to {request.path}: ADMIN_API_KEY is not configured. "
                "Set ADMIN_API_KEY or set ALLOW_INSECURE_DEV=1 for local testing."
            )
            return jsonify({
                "status": "error",
                "message": "Admin API key unconfigured. Set ADMIN_API_KEY or set ALLOW_INSECURE_DEV=1."
            }), 401

        # Strictly reject query parameters to prevent token leakage in HTTP access logs
        if "api_key" in request.args:
            logger.warning(f"Rejected query parameter authentication attempt on {request.path} from IP {request.remote_addr}")
            return jsonify({
                "status": "error",
                "message": "Unauthorized: API keys must be provided via X-API-Key or Authorization Bearer header, not query parameters."
            }), 401

        auth_header = request.headers.get("Authorization", "").strip()
        bearer_token = auth_header[7:].strip() if auth_header.startswith("Bearer ") else ""
        provided_key = request.headers.get("X-API-Key") or bearer_token

        if not provided_key or not hmac.compare_digest(str(provided_key), str(active_key)):
            logger.warning(f"Unauthorized access attempt to admin endpoint {request.path} from IP {request.remote_addr}")
            return jsonify({
                "status": "error",
                "message": "Unauthorized: Valid X-API-Key or Bearer token required."
            }), 401

        return f(*args, **kwargs)
    return decorated
