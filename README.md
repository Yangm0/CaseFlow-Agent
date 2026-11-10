# CaseFlow Agent V1-A

CaseFlow is a small, reviewable payment-incident investigation agent. It connects a FastAPI request to read-only PostgreSQL queries, Chinese BM25 retrieval, and an explicit LangGraph tool loop. It is intended as an interviewable engineering project, not a production payment control plane.

## Current local verification status

Verified on Windows with Python 3.13.2 on 2026-11-10:

- Installed `.[dev]` into the existing `.venv` through the active v2rayN proxy (`127.0.0.1:10808`). FastAPI, LangGraph, SQLAlchemy, psycopg, jieba, and pytest all import successfully.
- Docker Desktop 4.99.0 is running with Docker Engine 29.9.3, Compose 5.7.0, and its WSL 2 backend. PostgreSQL 17.11 is healthy on `127.0.0.1:5432`.
- The separate `caseflow` and `caseflow_test` databases each contain 49,600 payment attempts, 18 documents, 7 changes, 4 incidents, and 12 metadata rows. Re-seeding `caseflow` leaves those counts unchanged. The application role can select but cannot insert, update, or delete; the seed role can create objects in the project schema.
- Pytest collected 8 tests and all 8 passed, including the PostgreSQL integration checks and offline FastAPI end-to-end investigation. No tests were skipped.
- A real offline API request returned HTTP 200 and `completed`. It called `query_payment_metrics`, `query_change_events`, and `search_knowledge` in that order. PostgreSQL returned 22/50 failed attempts (44%) versus 10/330 (3.03%) across the seven prior matching windows. The report included related change and incident records, five BM25 passages, and a low-confidence route/retry hypothesis with source-checked quotations; it states that the evidence does not prove causation.
- This is offline engineering acceptance using the deterministic fake model. No paid or live LLM request was made, and it is not a live model quality evaluation.

## V1-A scope

- PostgreSQL tables for synthetic payment attempts, service changes, incidents, documents, and dataset metadata.
- Parameterized SQLAlchemy aggregate tools with bounded time windows, allow-listed grouping dimensions, statement timeouts, result caps, and database read-only permissions.
- Chinese tokenization with `jieba` and a real `rank_bm25` index built from documents read from PostgreSQL.
- A single LangGraph `StateGraph` that loops through model decisions, tool execution, and structured synthesis under hard model/tool call caps, elapsed-time checks between nodes, and per-query PostgreSQL timeouts.
- `offline_test` uses a deterministic rule-based fake model for tests and demos. It is not a real LLM and is labeled as such.
- `live` uses an OpenAI-compatible Function Calling API through `langchain-openai`; credentials and endpoint settings come from environment variables.
- FastAPI endpoints: `GET /health` and `POST /api/v1/investigations`.

V1-A deliberately excludes writes, auto-remediation, pgvector, MCP, Redis, multi-agent coordination, web UI, and persistent checkpoints.

## Architecture and request path

```text
POST /api/v1/investigations
  -> LangGraph decision node
  -> one or more of the three read-only tools
  -> tool results returned to the decision node
  -> structured synthesis
  -> server-built statistical facts + validated citations
```

The API report keeps statistical facts, retrieved document passages, service changes/incidents, tentative hypotheses, uncertainties, recommendations, tool traces, and operation status in separate fields. Statistical values are copied from tool output by the server. A hypothesis is retained only when its evidence ID exists and its quoted text exactly matches retrieved source content. This checks source linkage and quote fidelity; human review is still needed to judge whether a hypothesis over-interprets a quote.

## Data definition

The generator is in `src/caseflow/data/synthetic.py`; it has no dependency on API calls or external data.

- Fixed seed: `20250121`.
- Local business timezone: `Asia/Shanghai`; generated aware timestamps are stored as PostgreSQL `timestamptz`.
- Data interval: `[2025-01-01 00:00:00+08:00, 2025-02-01 00:00:00+08:00)`; `as_of` is the exclusive end.
- Payment attempts: 31 days × 4 regions × 400 records = 49,600 deterministic rows. `pending` rows are present to exercise the denominator rule.
- Documents: 18 Chinese operations manuals/runbook passages. Changes and incident rows include related and unrelated examples from multiple regions.
- A Beijing anomaly is injected for `[2025-01-21 09:00, 12:00)`: the terminal failure probability is raised from 2.5% to 40%. The ground-truth label and expected related IDs are isolated in `evals/ground_truth.json`; that file is never imported into the agent, tool prompt, or database.
- Valid attempts are rows with status `succeeded` or `failed`. Failure rate is `failed_attempts / valid_attempts`; a zero denominator yields `null`.
- Target intervals are left-closed/right-open. For windows up to 24 hours, the baseline is the seven prior same-clock daily windows. For longer windows, the baseline is the immediately preceding equal-duration interval.

Seed/import is additive and idempotent. It creates missing CaseFlow tables and inserts deterministic IDs with `ON CONFLICT DO NOTHING`; it contains no delete, truncate, drop, reset, or volume-removal operation. It accepts database names `caseflow` and `*_test` only. The test workflow uses a separately provisioned `caseflow_test` database.

## Setup on Windows PowerShell

Requirements: Python 3.13, Docker Compose v2 for PostgreSQL, and PowerShell. The project does not change the global Python installation.

```powershell
Copy-Item .env.example .env
python --version
python -m venv .venv
New-Item -ItemType Directory -Path .tmp -Force | Out-Null
$env:TEMP = (Resolve-Path .tmp).Path
$env:TMP = $env:TEMP
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

The sample `.env` values are for local development only. Change passwords before exposing PostgreSQL beyond loopback. Compose provisions a `caseflow` development database and an isolated `caseflow_test` database, with separate `caseflow_app` (read-only) and `caseflow_seed` (schema/seed) roles.

```powershell
docker compose up -d postgres
python scripts/seed_database.py
```

The script prints row counts. Re-running it is safe and does not delete existing records. It does not attempt to install or configure Docker/PostgreSQL.

## Offline run

Keep `CASEFLOW_MODE=offline_test` in `.env`:

```powershell
python -m uvicorn caseflow.api:app --app-dir src --reload
```

In another PowerShell window:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
$body = @{ question = '北京 2025-01-21 09:00 到 12:00 支付失败率是否异常？请查一下变更和文档。' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/v1/investigations -ContentType 'application/json' -Body $body
```

This exercises real SQL and BM25 against the local PostgreSQL instance, but the offline decision and synthesis are driven by a deterministic fake model. It must not be described as a real LLM result. The offline parser supports the documented demo query style and falls back to the synthetic Beijing scenario when no explicit date/window is present.

## Live model run

Set `CASEFLOW_MODE=live`, `OPENAI_API_KEY`, `OPENAI_MODEL`, and optionally `OPENAI_BASE_URL` in `.env`, then restart the API. The model must support tool/function calling and structured output. No credentials are bundled; without a key, `/health` reports degraded status and investigation requests return `blocked` with `LLM_CREDENTIALS_MISSING`. No paid service is opened automatically.

The API response does not include the key or configured base URL. Tool-call traces contain tool names, validated arguments, timings, outcomes, and evidence IDs; they do not log model credentials or raw SQL.

## Tests

```powershell
python -m pytest -q
```

The suite intentionally contains only 8 tests: five compact unit checks and three PostgreSQL integration checks, including one end-to-end investigation. Integration and end-to-end tests require both `TEST_DATABASE_URL` and `TEST_SEED_DATABASE_URL` pointing to the isolated `caseflow_test` database and its read-only/seed roles. The fixture refuses database names other than `caseflow_test`. If the URLs are absent, those tests are reported as skipped; skipped tests are not PostgreSQL or end-to-end passes.

Integration coverage includes database connectivity, deterministic import counts/idempotency, SQL interval boundaries, real BM25 retrieval, read-only permission enforcement, API response shape, and an offline fake-model investigation using actual PostgreSQL and BM25 tools. It does not substitute SQLite or a mocked database for PostgreSQL.

## Evaluation cases

`evals/v1a_cases.json` contains 20 categorized reference tasks covering normal/no-anomaly investigations, missing evidence, invalid parameters, security, tool failures/timeouts, citation validation, and uncertainty. `evals/ground_truth.json` is for evaluator use only and is not part of the agent's runtime context. V1-A includes the cases and their references; it does not claim model accuracy or measured quality improvements.

## Known limits and V1-B boundary

- PostgreSQL and live LLM acceptance depend on local Docker/PostgreSQL and user-provided model credentials.
- The BM25 index is built from the current PostgreSQL document rows per search request; there is no persistent/vector index.
- The offline fake is a deterministic test driver with limited question parsing, not an LLM substitute.
- Synthetic payments do not include card/channel response codes, latency percentiles, customer details, or production-quality alert streams.
- Citation IDs and exact quote spans are verified, but semantic entailment between a quote and a generated hypothesis is not fully automatic.
- The elapsed-time budget is checked between nodes; an in-flight LLM request can finish near its configured request timeout before the next check stops the graph. Hard call caps and PostgreSQL statement timeouts remain enforced.
- No recovery action or payment write path exists.

V1-B can add a reproducible held-out evaluation runner and richer response-code/latency evidence after V1-A's PostgreSQL-backed path is verified. pgvector, MCP, Redis, and multi-agent orchestration should remain separate decisions rather than prerequisites for that boundary.
