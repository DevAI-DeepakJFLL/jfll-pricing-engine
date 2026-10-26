# Security Hardening & Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remediate all vulnerabilities identified in the security audit across container packaging, API authentication, model deserialization integrity, and CI/CD pipelines with zero functional regressions.

**Architecture:** Implement defense-in-depth security: strict `.dockerignore` data boundary enforcement, unprivileged container execution via non-root `appuser`, configurable header-based API key authentication (`X-API-Key`) for sensitive endpoints, and cryptographic SHA-256 verification of ML model bundles prior to deserialization.

**Tech Stack:** Python 3.11, Flask 3.1, Docker, hashlib, Pytest, Bandit AST Security Scanner.

**Spec:** Security Audit Report (2026-10-26) covering CWE-200, CWE-306, CWE-250, CWE-502, and CWE-862.

## Global Constraints

- Never break existing 299 test cases in `pytest tests/`.
- Ensure backwards compatibility for local and development runs when `ADMIN_API_KEY` is not set.
- Do not alter mathematical prediction logic in `src/quantile_model.py` or `src/engine_v3.py`.
- Preserve UI visual styling and templates in `src/templates/`.
- Maintain clean, minimal Git commits.

---

### Task 1: Prevent Sensitive Data Leakage in Docker Builds (CWE-200 / CWE-312)

**Files:**
- Modify: `.dockerignore`

**Interfaces:**
- Excludes: `data/raw/*`, `data/processed/*`, `data/*.db*`, `*.xlsx`, `*.csv`, `*.parquet`

- [ ] **Step 1: Inspect current `.dockerignore`**
- [ ] **Step 2: Add strict data exclusion patterns matching `.gitignore`**
- [ ] **Step 3: Verify with simulated docker build context / file scan**
- [ ] **Step 4: Commit change: `chore(docker): exclude confidential spreadsheets and database files from docker build context`**

---

### Task 2: Implement Non-Root User Runtime in Container (CWE-250 / CIS Benchmark 4.1)

**Files:**
- Modify: `Dockerfile`

**Interfaces:**
- System: Linux user `appuser` (UID 10001, GID 10001) with home directory `/app`
- Permissions: `/app` directory read/write ownership granted to `appuser`

- [ ] **Step 1: Add non-root `appuser` creation in `Dockerfile`**
- [ ] **Step 2: Ensure `/app` and SQLite `/tmp` permissions are properly configured**
- [ ] **Step 3: Set `USER appuser` prior to container entrypoint**
- [ ] **Step 4: Commit change: `sec(docker): run container entrypoint under non-privileged appuser`**

---

### Task 3: Implement Authentication Guard for Admin Endpoints (CWE-306 / CWE-862)

**Files:**
- Modify: `src/app.py:925-985`
- Test: `tests/test_api_quote.py`

**Interfaces:**
- Decorator: `require_admin_auth(f)`
- Header: `X-API-Key`
- Query Param: `?api_key=`
- Environment: `ADMIN_API_KEY` (if set, enforce; if unset, allow local development with audit warning)

- [ ] **Step 1: Write unit test in `tests/test_api_quote.py` testing `require_admin_auth` with valid, invalid, and unset API key**
- [ ] **Step 2: Implement `require_admin_auth` decorator in `src/app.py`**
- [ ] **Step 3: Attach `@require_admin_auth` to `/api/history/clear` and `/api/history/export`**
- [ ] **Step 4: Run unit tests to verify both existing test suite and new security tests pass**
- [ ] **Step 5: Commit change: `sec(api): add configurable API key authentication guard to history clear and export endpoints`**

---

### Task 4: Add Cryptographic SHA-256 Model Integrity Verification (CWE-502)

**Files:**
- Create: `models/checksums.json`
- Modify: `src/engine_v3.py`
- Modify: `src/app.py`
- Test: `tests/test_security_integrity.py`

**Interfaces:**
- Function: `verify_artifact_sha256(filepath: str, expected_hash: str) -> bool`
- Manifest: JSON mapping model filenames to their SHA-256 digests

- [ ] **Step 1: Compute SHA-256 checksums for production model bundles and save to `models/checksums.json`**
- [ ] **Step 2: Implement `verify_artifact_sha256` helper in `src/engine_v3.py` and `src/app.py`**
- [ ] **Step 3: Validate model checksum on load, logging security verification status**
- [ ] **Step 4: Write unit test in `tests/test_security_integrity.py` verifying tamper detection**
- [ ] **Step 5: Commit change: `sec(engine): enforce SHA-256 cryptographic verification on model bundle deserialization`**

---

### Task 5: Eliminate S3 CI/CD Accidental Deletion Hazard

**Files:**
- Modify: `.github/workflows/deploy.yml`

**Interfaces:**
- CI Workflow: `aws s3 sync`

- [ ] **Step 1: Review line 87 of `.github/workflows/deploy.yml`**
- [ ] **Step 2: Remove hazardous `--delete` flag from `data/processed/` sync on fresh runners**
- [ ] **Step 3: Commit change: `fix(ci): remove destructive --delete flag from processed dataset S3 sync`**

---

### Task 6: Full Pytest & Bandit Verification Gate

**Files:**
- Verification: Pytest suite (300+ tests)
- Verification: Bandit AST security scanner

- [ ] **Step 1: Run Bandit AST scan to verify 0 High / 0 Medium security vulnerabilities**
- [ ] **Step 2: Run `verify_vocab_alignment.py` (ensure 36/36 pass)**
- [ ] **Step 3: Run full `pytest tests/` test suite (ensure 100% pass)**
- [ ] **Step 4: Final git status check and verification report to user**
