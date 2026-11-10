# CaseFlow Agent V1-B

CaseFlow is a small, reviewable payment-incident investigation agent. It connects a FastAPI request to read-only PostgreSQL queries, Chinese BM25 retrieval, and an explicit LangGraph tool loop. It is intended as an interviewable engineering project, not a production payment control plane.

## Current local verification status

Verified on Windows with Python 3.13.2 on 2026-11-10:

- The existing `.venv` has the project and development dependencies installed; FastAPI, LangGraph, SQLAlchemy, psycopg, jieba, and pytest import successfully. Package installation used the active v2rayN proxy without changing system-wide proxy settings.
- Docker Desktop is running with Docker Engine 29.9.3, Compose 5.7.0, and its WSL 2 backend. PostgreSQL 17 is healthy on `127.0.0.1:5432`. The Docker CLI binary is installed in the current user's Docker Desktop directory but is not on this PowerShell session's PATH.
- The separate `caseflow` and `caseflow_test` databases each contain 49,600 payment attempts, 18 documents, 7 changes, and 4 incidents. The test database contains no leftover evaluation-injection document. The application role can select but cannot insert, update, or delete.
- Ruff checks pass, pytest collects exactly 8 tests, and all 8 pass against the real PostgreSQL test database.
- A real live API smoke completed with `nvidia/nemotron-3.5-super-vl:free` through OpenRouter. It used two decision calls followed by one JSON-schema synthesis call. Tool order was `query_payment_metrics` → `query_change_events` → `search_knowledge`; all three succeeded.
- The smoke queried Beijing on 2025-01-21 09:00–12:00: 22/50 failed (44.00%) versus 10/330 (3.03%) over seven prior matching windows. The related change at 08:52 and incident at 09:07 were returned, along with five BM25 passages. The report treats timing as a clue, not proof of cause. It retained one low-confidence hypothesis with two source-checked quotations and removed five of seven proposed citations.
- A four-case live comparison completed for Direct LLM, PostgreSQL BM25-RAG, and the full Agent. Completion was 4/4 in each condition; full Agent tool selection was exact in 4/4 cases and all attempted tool calls executed successfully. Exact argument checks passed for the separately captured full investigation. The heuristic numeric screen found no unsupported numeric tokens; it is not a semantic fact checker. Citation candidate validity in the full Agent was 46.7%; normal and missing-evidence cases had rejected citations, which remain a known quality issue.
- A six-query BM25 retrieval evaluation measured mean Recall@5 0.722 and MRR 1.000. Labels were set independently of rankings; every query's first result was relevant.
- The live smoke and evaluation used only synthetic records in `caseflow_test`. The model route was checked as a zero-price `:free` route at run time; pricing and quotas can change. No paid model API was called.

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

`evals/v1a_cases.json` contains the existing reference tasks; `evals/v1b_cases.json` adds direct no-tool and adversarial-document cases. `evals/ground_truth.json` is evaluator-only and is never imported into the Agent or tool prompts. The live evaluator compares Direct LLM, PostgreSQL BM25-RAG, and the full Agent on the same selected questions, and reports completion, tool selection, schema-compatible call execution, checked argument matches, citation IDs, heuristic unsupported numeric tokens, latency, tokens, and bad cases. Completion means the condition returned a non-empty answer or completed report; it is not an answer-correctness score.

Run the independent retrieval evaluation against the real test database:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_retrieval.py
```

Run the bounded live comparison only after confirming current route pricing and quota:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_agent.py `
  --case normal-01 `
  --case missing-evidence-01 `
  --case direct-no-tool-01 `
  --case prompt-injection-01 `
  --confirm-zero-cost-free-endpoint
```

For any case involving the adversarial document, the runner inserts a uniquely named temporary document into `caseflow_test` and removes exactly that row in `finally`. Evaluation outputs are written under ignored `.tmp/`. The four-case live sample is an engineering check, not a statistically representative model benchmark. In the measured run, Direct LLM and BM25-RAG each completed 4/4 cases; the full Agent completed 4/4 and selected the expected tool sequence in 4/4. The heuristic numeric scan found zero unmatched numeric tokens. Exact argument matching was scored on one full investigation and passed 1/1. The full Agent citation candidate validity was 46.7%, so it is not yet a strong citation-quality result. The retrieval score applies only to the six manually labeled synthetic queries.

## Known limits and V1-C boundary

- PostgreSQL and live LLM acceptance depend on local Docker/PostgreSQL and user-provided model credentials.
- BM25 remains the retrieval baseline. Its tokenized document corpus and rank index are cached in process and refreshed when document count or latest update time changes; there is no persistent or vector index.
- The offline fake is a deterministic test driver with limited question parsing, not an LLM substitute.
- Synthetic payments do not include card/channel response codes, latency percentiles, customer details, or production-quality alert streams.
- Citation IDs and exact quote spans are verified, but quote matching does not prove semantic entailment. The live comparison still found rejected citations; improve quote selection and citation supervision before presenting citation quality as reliable.
- The elapsed-time budget is checked between nodes; an in-flight LLM request can finish near its configured request timeout before the next check stops the graph. The implementation does not claim hard cancellation of synchronous provider calls.
- No recovery action or payment write path exists. pgvector, embedding retrieval, MCP, Redis, multi-agent coordination, web UI, and persistent checkpoints remain out of scope for this V1-B branch.

A V1-C iteration should first improve citation validity and add a larger held-out evaluation set. Only then compare an embedding or hybrid retriever against the same BM25 relevance labels. Keep any write or remediation path outside this read-only investigation Agent.
