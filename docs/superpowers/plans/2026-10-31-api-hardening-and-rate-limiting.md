# API Hardening & Quotation Rate Limiting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement uniform JSON error handlers for all HTTP error states (400, 404, 405, 413, 429, 500) and thread-safe, sliding-window quotation rate limiting on `/api/quote` to prevent scraping and memory exhaustion.

**Architecture:** Extend `src/app.py` with content-negotiating error handlers that return structured JSON for API calls and client requests. Build a thread-safe, memory-bounded, zero-dependency sliding-window rate limiter in `src/security.py` and apply it to `/api/quote` with standard `Retry-After` headers and configurable thresholds.

**Tech Stack:** Python 3.11, Flask 3.1.3, Werkzeug, standard library (`threading`, `collections`, `time`, `math`).

**Spec:** Recommendations 1 & 2 from Senior Security Audit Report (`SECURITY_AUDIT_REPORT.md`).

## Global Constraints

- Zero external package additions: use Python standard library (`threading`, `collections.defaultdict`, `time`, `math`) and existing Flask/Werkzeug utilities.
- Maintain documentation and docstring integrity across modified files.
- Thread safety: all stateful shared structures must be synchronized with `threading.Lock`.
- Memory bounds: rate-limiting stores must prune expired timestamps to prevent unbounded memory growth (CWE-400).
- Backward compatibility: ensure all 345 existing pytest tests pass without regression.

---

### Task 1: Rate Limiter Module & Unit Tests (TDD)

**Files:**
- Modify: `src/security.py`
- Create/Modify: `tests/test_rate_limiter.py`

**Interfaces:**
- Consumes: `flask.request`, `time.time`, `threading.Lock`.
- Produces: `rate_limit(max_requests=60, window_seconds=60, by="ip")` decorator, `reset_rate_limits()` test helper, `get_client_ip()`.

- [ ] **Step 1: Write the failing unit tests for rate limiter**
  Create `tests/test_rate_limiter.py` testing:
  - Allowing requests within the limit (e.g. 3 requests in 60s).
  - Blocking requests exceeding the limit with HTTP 429 and `Retry-After` header.
  - IP resolution with and without `X-Forwarded-For`.
  - Disabling rate limits via `RATE_LIMIT_ENABLED=0`.
  - Memory cleanup/pruning of expired entries.

- [ ] **Step 2: Run test suite to verify tests fail**
  `PYTHONPATH=. pytest tests/test_rate_limiter.py -v` (expect import or assertion error).

- [ ] **Step 3: Implement sliding-window rate limiter in `src/security.py`**
  Implement thread-safe, memory-bounded rate limiter decorator `rate_limit` with sliding window, automatic purge, and `Retry-After` header.

- [ ] **Step 4: Run tests to verify they pass**
  `PYTHONPATH=. pytest tests/test_rate_limiter.py -v` (all tests pass).

---

### Task 2: Uniform API Error Handlers & Error Handling Tests (TDD)

**Files:**
- Modify: `src/app.py`
- Create/Modify: `tests/test_api_error_handlers.py`

**Interfaces:**
- Consumes: Flask application instance `app`, `werkzeug.exceptions.HTTPException`.
- Produces: Global error handlers for 400, 404, 405, 413, 429, 500 returning consistent JSON payloads for API routes (`/api/*`) or `Accept: application/json`.

- [ ] **Step 1: Write failing tests for error handlers**
  Create `tests/test_api_error_handlers.py` testing:
  - 400 on malformed JSON payload to `/api/quote`.
  - 404 on nonexistent API route `/api/nonexistent`.
  - 405 on invalid method `GET /api/history/outcome`.
  - 413 on oversized payload exceeding 1MB (`MAX_CONTENT_LENGTH`).
  - Standard HTML preserved for browser navigation to non-existent page `/does_not_exist`.

- [ ] **Step 2: Run test suite to verify tests fail**
  `PYTHONPATH=. pytest tests/test_api_error_handlers.py -v` (expect HTML responses instead of JSON).

- [ ] **Step 3: Implement uniform error handlers in `src/app.py`**
  Add content-negotiated error handlers for 400, 404, 405, 413, 429, and 500 in `src/app.py`.

- [ ] **Step 4: Run tests to verify they pass**
  `PYTHONPATH=. pytest tests/test_api_error_handlers.py -v` (all tests pass).

---

### Task 3: Attach Rate Limiting to `/api/quote` & End-to-End Integration

**Files:**
- Modify: `src/app.py`
- Modify: `tests/test_api_quote.py`

**Interfaces:**
- Consumes: `rate_limit` from `src.security`.
- Produces: Guarded `/api/quote` endpoint with 60 requests/minute default rate limit.

- [ ] **Step 1: Add rate limiting decorator to `/api/quote` in `src/app.py`**
  Apply `@rate_limit(max_requests=60, window_seconds=60)` to `api_quote()`.

- [ ] **Step 2: Verify existing quote tests pass**
  `PYTHONPATH=. pytest tests/test_api_quote.py tests/test_api_v3_quote.py -v`.

- [ ] **Step 3: Add integration test for burst limit on `/api/quote`**
  Verify that repeated burst inquiries trigger HTTP 429 with JSON payload and Retry-After header.

---

### Task 4: Full Security Scan & Zero-Regression Verification

**Files:**
- Run: Bandit scanner on `src/`.
- Run: Full pytest suite.

- [ ] **Step 1: Run Bandit SAST security scan**
  `.venv/bin/bandit -r src/` (verify 0 vulnerabilities).

- [ ] **Step 2: Run complete test suite**
  `PYTHONPATH=. pytest tests/` (verify all 350+ tests pass).
