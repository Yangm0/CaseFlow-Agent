# CaseFlow Agent V1-B

CaseFlow is a small, reviewable payment-incident investigation agent. It connects a FastAPI request to read-only PostgreSQL queries, Chinese BM25 retrieval, and an explicit LangGraph tool loop. It is intended as an interviewable engineering project, not a production payment control plane.

## Current local verification status

Verified on Windows with Python 3.13.2 on 2026-11-10 after the V1-B quality fixes:

- The existing `.venv` has the project and development dependencies installed; FastAPI, LangGraph, SQLAlchemy, psycopg, jieba, and pytest import successfully. Package installation used the active v2rayN proxy without changing system-wide proxy settings.
- Docker Desktop is running with Docker Engine 29.9.3, Compose 5.7.0, and its WSL 2 backend. PostgreSQL 17 is healthy on `127.0.0.1:5432`. The Docker CLI binary is installed in the current user's Docker Desktop directory but is not on this PowerShell session's PATH.
- The separate `caseflow` and `caseflow_test` databases each contain 49,600 payment attempts, 18 documents, 7 changes, and 4 incidents. The test database contains no leftover evaluation fixture. The application role can select but cannot insert, update, or delete.
- Ruff passes, pytest collects exactly 8 tests, and all 8 pass against real PostgreSQL. The BM25 integration check inserts, updates, and deletes one uniquely named test document and confirms the cached index follows each change; it removes that row in `finally`.
- A fresh offline end-to-end request used PostgreSQL and BM25 in `caseflow_test`: Beijing 2025-01-21 09:00–12:00 returned 22/50 failed (44.00%) versus 10/330 (3.03%), with the related change and incident. Tool order was metrics → change events → knowledge search. It retained one deterministic offline hypothesis; both server-generated citations resolved to their source and matched exact text (2/2). This offline result is not a live-model evaluation. Local trace: `.tmp/v1b_offline_qualityfix_trace.json`.
- A separate offline metric-only case returned Shenzhen 36/2 failed (5.56%) versus 5/252 (1.98%), with both status groups. The evaluator's six deterministic checks passed (task success 1/1 for this labeled synthetic query); natural-language answer correctness remains ungraded. Local trace: `.tmp/v1b_offline_metric_validation.json`.
- Corrected document-level retrieval evaluation covers 9 synthetic queries: 6 development and 3 validation. Recall@5 is 0.708 over 8 queries with relevant-document labels (development 0.722/6; validation 0.667/2). MRR is 0.889 over all 9 queries (development 1.000/6; validation 0.667/3). The unrelated validation query has no relevant-document denominator and still returns five documents, exposing a current BM25 abstention limitation.
- The pre-fix Live smoke used OpenRouter `nvidia/nemotron-3.5-super-vl:free`, with 3 model calls, 20.5 s elapsed, and 8,988 tokens. It retained 2/7 proposed free-form citations (28.6%). In the four-case pre-fix comparison, the reported 46.7% was a macro-average of three citation-bearing cases (normal 2/5, missing-evidence 0/1, injection 2/2); the corresponding micro result was 4/8 (50%). These old results are explicitly pre-fix and are not a post-fix Live result.
- A current OpenRouter preflight confirmed the configured free variant is listed at $0 prompt and completion pricing and the key is marked free tier. The key endpoint does not expose remaining free requests; its request-rate field is deprecated. Because the earlier same-day evaluations may have used a substantial part of the 50-request daily cap, no post-fix Live model request was sent while the remaining quota was unknown. No paid model API was called.

## V1-B engineering scope

- PostgreSQL tables for synthetic payment attempts, service changes, incidents, documents, and dataset metadata.
- Parameterized SQLAlchemy aggregate tools with bounded time windows, allow-listed grouping dimensions, statement timeouts, result caps, and database read-only permissions.
- Chinese tokenization with `jieba` and a real `rank_bm25` index built from documents read from PostgreSQL.
- A single LangGraph `StateGraph` that loops through model decisions, tool execution, and structured synthesis under hard model/tool call caps, elapsed-time checks between nodes, and per-query PostgreSQL timeouts.
- `offline_test` uses a deterministic rule-based fake model for tests and demos. It is not a real LLM and is labeled as such.
- `live` uses an OpenAI-compatible Function Calling API through `langchain-openai`; credentials and endpoint settings come from environment variables.
- FastAPI endpoints: `GET /health` and `POST /api/v1/investigations`.

V1-B deliberately excludes writes, auto-remediation, pgvector, MCP, Redis, multi-agent coordination, web UI, and persistent checkpoints.

## Architecture and request path

```text
POST /api/v1/investigations
  -> LangGraph decision node
  -> one or more of the three read-only tools
  -> tool results returned to the decision node
  -> structured synthesis
  -> server-built statistical facts + validated citations
```

The API report keeps statistical facts, retrieved document passages, service changes/incidents, tentative hypotheses, uncertainties, recommendations, tool traces, and operation status in separate fields. Statistical values are copied from tool output by the server. During synthesis, the model selects stable IDs from a compact evidence catalog; it does not supply quote text. The server resolves each ID against the exact catalog passed to the model and constructs the quote from that source. Unknown IDs and numeric claims missing from selected evidence are rejected. This verifies source linkage and exact quote fidelity; it does not establish that the source semantically supports the hypothesis.

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
if (Get-Command docker -ErrorAction SilentlyContinue) {
    docker compose up -d postgres
} else {
    $docker = Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\resources\bin\docker.exe'
    & $docker compose up -d postgres
}
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

`live` uses an OpenAI-compatible API through `langchain-openai`. Function calling is enabled with automatic tool selection; the agent passes the validated tool schemas, returns tool results to the model, and can continue the investigation before synthesis. Structured output explicitly selects `json_schema` and retries once with `function_calling` only when the provider rejects the configured method. Stage-specific timeouts, token limits, model-call and tool-call caps, safe error codes, token usage, and elapsed time are recorded in the response trace.

For a local run, configure `CASEFLOW_MODE=live`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL` in the ignored `.env` file or the current PowerShell process. The smoke and evaluation scripts refuse non-`:free` models and non-OpenRouter endpoints; they also require the isolated `caseflow_test` database. The free model route used for this verification was `nvidia/nemotron-3.5-super-vl:free`. Recheck provider pricing and quota before each run because they can change. The route verified for this run was listed as free by [OpenRouter's model page](https://openrouter.ai/nvidia/nemotron-3.5-super-vl%3Afree); see [current free-route limits](https://openrouter.ai/collections/free-models).

```powershell
$env:CASEFLOW_MODE = 'live'
$env:OPENAI_BASE_URL = 'https://openrouter.ai/api/v1'
$env:OPENAI_MODEL = 'nvidia/nemotron-3.5-super-vl:free'
$env:OPENAI_API_KEY = '<your local test key>'
.\.venv\Scripts\python.exe scripts\live_smoke.py --confirm-zero-cost-free-endpoint
```

The script calls FastAPI through `TestClient`, not a production service. It connects only to `TEST_DATABASE_URL`, checks the configured database name, refuses the request unless the `:free` route is explicitly confirmed, and writes a redacted local trace to ignored `.tmp/`. Never put a real key in README, source control, or a shared trace.

The verified run used synthetic data only. OpenRouter's current free-route terms and model pricing should be rechecked before using any other data or repeating an evaluation. The configured API key and base URL are not returned in reports.

## Tests

```powershell
python -m pytest -q
```

The suite intentionally contains only 8 tests: five compact unit checks and three PostgreSQL integration checks, including one end-to-end investigation. Integration and end-to-end tests require both `TEST_DATABASE_URL` and `TEST_SEED_DATABASE_URL` pointing to the isolated `caseflow_test` database and its read-only/seed roles. The fixture refuses database names other than `caseflow_test`. If the URLs are absent, those tests are reported as skipped; skipped tests are not PostgreSQL or end-to-end passes.

Integration coverage includes database connectivity, deterministic import counts/idempotency, SQL interval boundaries, real BM25 retrieval, read-only permission enforcement, API response shape, and an offline fake-model investigation using actual PostgreSQL and BM25 tools. It does not substitute SQLite or a mocked database for PostgreSQL.

## Evaluation cases

`evals/v1a_cases.json` contains the existing reference tasks; `evals/v1b_cases.json` adds direct no-tool and adversarial-document cases. `evals/ground_truth.json` and `evals/evaluation_labels.json` are evaluator-only and are never imported into the Agent or tool prompts. The labels separate the development set from a held-out validation set. A held-out case must not be used to tune prompts or thresholds.

The evaluator reports execution completion, completed report status, exact ordered tool selection, argument checks, evidence ID source validity, exact quote fidelity, citation acceptance, latency, tokens, and bad cases separately. Each rate exposes its numerator and denominator. Answer correctness and semantic entailment remain `ungraded` unless a human supplies a label. Task success requires all task-specific automated checks to pass; a case with semantic requirements stays ungraded even if its deterministic checks pass. HTTP 200 and `status=completed` are not task-success labels.

The old Live trace stored only aggregate citation acceptance, not rejected candidate IDs or text, so it cannot support a trustworthy historical breakdown of the five rejected quotes. The new path records missing evidence IDs, unsupported numeric claims, duplicates, and exact-source quote checks. A sanitized regression rejects the fabricated `MISSING` ID; another rejects a `99%` claim when its selected source contains no matching number. Multi-source hypotheses keep separate IDs and separate server-generated quotes. Whether a valid source semantically supports its associated hypothesis remains ungraded.

Run document-level BM25 retrieval against the real test database. The evaluator ranks all chunks, collapses repeated chunks by document ID, then applies Recall@5 and MRR to unique documents. The manually assigned labels are independent of ranking output. The unrelated query has no relevant-document denominator for Recall@5 and contributes zero to MRR when no relevant result is found.

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_retrieval.py
.\.venv\Scripts\python.exe scripts\evaluate_retrieval.py --split validation
```

The process-local BM25 cache recomputes a SHA-256 signature from every ordered document ID, title, and content on each search. It retokenizes and rebuilds the index only when that signature changes, detecting inserts, in-place edits, and deletions without a database migration or distributed cache.

Run the bounded Live comparison only after verifying that the selected model remains zero-priced and the account has free-request quota remaining. The configured inference-key endpoint reports monetary use but may not expose the free request count; inspect the OpenRouter Activity page when the remaining count is unavailable by API. Failed requests also consume the free request allowance.

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_agent.py `
  --split validation `
  --condition full_agent `
  --confirm-zero-cost-free-endpoint
```

For any case involving the adversarial document, the runner inserts one uniquely named document into `caseflow_test` and removes exactly that row in `finally`. Outputs go under ignored `.tmp/`. The earlier four-case Live comparison is pre-fix only: each condition returned completed reports for 4/4 cases, but that is execution completion, not task success. Full Agent tool selection was exact in 4/4; explicit argument checks covered only one case and passed 1/1. The 46.7% citation figure was a macro-average over three citation-bearing cases; micro acceptance was 4/8 (50%). Its old heuristic numeric screen found zero unmatched numeric tokens, but it included the user's question among numeric sources and could therefore treat echoed request values as evidence. The current evaluator checks returned evidence only, with a regression that rejects a requested `99%` when the only returned fact is `5%`. These pre-fix free-form quote results are not directly comparable to post-fix evidence-ID resolution and server-generated quote fidelity. A final read-only OpenRouter preflight confirmed the remaining free-request quota was below the eight-request Live validation budget, so no post-fix Live model request was sent.

## Known limits and V1-C boundary

- PostgreSQL and live LLM acceptance depend on local Docker/PostgreSQL and user-provided model credentials.
- BM25 remains the retrieval baseline. It returns documents for the unrelated validation query and needs a calibrated abstention rule. Current scores come from only nine hand-labelled synthetic queries, so they do not establish broad retrieval performance. The in-process cache now detects changes using a content signature; there is no persistent or vector index.
- The offline fake is a deterministic test driver with limited question parsing, not an LLM substitute.
- Synthetic payments do not include card/channel response codes, latency percentiles, customer details, or production-quality alert streams.
- Stable evidence IDs and server-generated source excerpts prevent invented quote text and are checked against the exact catalog supplied to synthesis. They do not prove semantic entailment; the latter remains ungraded pending human review. Historical rejected quote causes cannot be reconstructed because the old traces did not store rejected candidates.
- The elapsed-time budget is checked between nodes; an in-flight LLM request can finish near its configured request timeout before the next check stops the graph. The implementation does not claim hard cancellation of synchronous provider calls.
- No recovery action or payment write path exists. pgvector, embedding retrieval, MCP, Redis, multi-agent coordination, web UI, and persistent checkpoints remain out of scope for this V1-B branch.

A V1-C iteration should add human-reviewed answer and entailment labels, improve BM25 behavior on irrelevant queries, and expand the held-out validation corpus without repeatedly tuning against it. Only then compare an embedding or hybrid retriever against the same relevance labels. Keep any write or remediation path outside this read-only investigation Agent.
