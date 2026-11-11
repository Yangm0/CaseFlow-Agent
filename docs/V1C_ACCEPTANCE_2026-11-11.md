# CaseFlow V1-C Final Integration Acceptance

**Acceptance date:** 2026-11-11 (Asia/Shanghai)

**Branch:** `feat/v1b-live-agent-evaluation`
**Scope:** local engineering demo on synthetic PostgreSQL data; PR #1 remains open and is not merged.

## Result summary

The V1-C MCP, PostgreSQL checkpoint, approval, pgvector/hybrid retrieval, SSE, browser UI, and Compose paths were exercised against the existing local test database. The original payment data in both databases was preserved. The eight-test pytest limit was maintained. One zero-price Live investigation completed; semantic claim support was not human-graded.

## A. Environment and containers

- Windows PowerShell, Python 3.13.2, the existing `.venv`, Docker Desktop/Engine 29.9.3, Compose 5.7.0, and WSL 2 were available.
- `pip check`, runtime import smoke, Ruff, and `docker compose config -q` passed.
- Compose services `postgres`, `mcp`, and `api` reported healthy. API and MCP are published on loopback ports 8100 and 8101 for this run.
- PostgreSQL reported 17.11 on Alpine; pgvector extension was 0.8.7.
- `caseflow_pgdata` remained mounted. No volume or database was dropped, cleared, or reinitialized.

## B. Review fixes

- `migrations/002_v1c_pgvector.sql` now provisions fixed-dimension partial expression HNSW indexes for each embedding model/dimension pair. Search queries cast to the same exact expression and model predicate. A real `EXPLAIN` selected the generated HNSW index.
- Checkpoint setup now gets database settings from `get_settings()` and guards development/test database selection.
- The MCP adapter was aligned with the installed LangChain MCP/FastMCP versions. Tool results, including structured content, are normalized into dictionaries; invalid inputs, timeouts, transport errors, and disconnects are handled.
- The approval path recovers pending database rows that lack a checkpoint interrupt, binds decisions to a server-verified payload hash, and uses locked/idempotent database functions. The long-lived checkpointer now uses a psycopg connection pool and retries a transient operational error once so thread reads recover after PostgreSQL restarts.
- The API/UI static path, report summary, and final approval button state were corrected based on runtime behavior.
- Compose builds pgvector 0.8.7 into the same `postgres:17-alpine` family as the existing volume, avoiding a data-directory libc/locale change from switching the base image.

## C. Backup, migrations, and data protection

Before applying V1-C migrations, both databases were exported as custom-format dumps and checked with `pg_restore --list`:

| Database | Backup | Bytes | SHA-256 |
|---|---|---:|---|
| `caseflow_test` | `.tmp/v1c-pre-migration-20261111/caseflow_test.dump` | 1,078,585 | `8e990835723dc6feee71678a70e503458bd674a392ec908f148d1eb33ff75751` |
| `caseflow` | `.tmp/v1c-pre-migration-20261111/caseflow.dump` | 1,078,550 | `435e4d75d85d700877d555448af0a4754b704fe1e52c4863ed39f073d2e1fb13` |

The test migration and checkpoint setup completed before the development database migration/setup. Final queries after a PostgreSQL container restart showed:

| Database | Attempts | Documents | Changes | Incidents | Vector rows | Approvals | Tickets |
|---|---:|---:|---:|---:|---:|---:|---:|
| `caseflow` | 49,600 | 18 | 7 | 4 | 0 | 0 | 0 |
| `caseflow_test` | 49,600 | 18 | 7 | 4 | 18 | 0 | 0 |

The test-only vector rows are deterministic 64-dimensional acceptance fixtures. All smoke-generated approvals, tickets, audit rows, and associated approval checkpoints were removed by their unique run tag; no approval or ticket rows remained at final verification.

Least-privilege checks showed `caseflow_app` can SELECT but cannot INSERT payment attempts or support tickets; `caseflow_action_executor` cannot directly INSERT support tickets but can execute the approval function; `caseflow_embedding_indexer` can write the embedding table; and the checkpoint role can create/use its own schema.

## D. MCP runtime

The standalone MCP smoke discovered `query_change_events`, `query_payment_metrics`, and `search_knowledge`; schema properties were present and MCP client calls returned structured dictionaries. Against `caseflow_test`, metrics returned 22 failed out of 50 valid attempts (44.00%) and the comparison baseline returned 10/330. Invalid parameters returned `INVALID_TOOL_ARGUMENTS`; a disconnected server returned `MCP_UNAVAILABLE`.

The thread investigation report recorded MCP tool execution in this order: metrics → change events → knowledge search. No database tool was bypassed on the thread path.

## E. Checkpoint, multi-turn, and restart recovery

- Two threads were created and kept separate. A follow-up question used the first thread's saved history; messages and report were present after an API restart.
- A pending approval interrupt survived reopening the checkpointer and was resumed after restart.
- The PostgreSQL container was restarted while the API process remained up. The same thread's persisted messages, completed report, and three tool-call records were returned on the first thread API request after restart. No API container restart was needed after the connection-pool fix.
- The browser reloaded the persisted thread and showed the completed 22/50 report and live SSE connection.

Offline follow-up answers use the deterministic `OfflineFakeModel`; their token count is unavailable and they are engineering-path checks, not Live model evaluations.

## F. Approval and HITL safety

The `caseflow_test` smoke exercised duplicate pending creation, a tampered payload hash, rejection after checkpoint restart, repeated rejection, rejection-to-ticket prevention, approval after restart, parallel duplicate approvals, database-only pending-row recovery, and direct table-write denial. A parallel approval batch returned the same single ticket ID. Audit rows were created within the tested workflow and removed during exact-tag cleanup.

The UI was used to submit/reject and submit/approve reviewable drafts. Final approval controls disappeared after a decision. The approved ticket was a test fixture and was removed after UI verification.

## G. BM25, pgvector, and Hybrid RAG

- BM25-only retrieval worked with no embedding provider configured.
- A deterministic local hash-vector provider indexed 18 synthetic document chunks into the test database; this is plumbing verification, not semantic embedding evaluation.
- Vector search and BM25 + pgvector Reciprocal Rank Fusion returned stable evidence IDs from both sources. The exact model/dimension expression used an HNSW index scan in `EXPLAIN`.
- Add/update/delete synchronization changed 1/1/1 fixture chunk and left zero stale rows after deletion. Re-indexing unchanged documents was idempotent.
- When vector search was unavailable or unconfigured, retrieval returned `bm25_only` with a warning rather than failing the investigation.

## H. Browser UI and SSE

In the browser, the UI loaded, created sessions, submitted an investigation question, streamed progress, showed report metrics/evidence, and loaded thread state after refresh. Both approval decisions were exercised. SSE reconnect replay using `Last-Event-ID` returned the expected event sequence. The browser displayed the persisted offline report after service restarts.

## I. Automated verification

- Pytest collect-only: **8 tests**.
- Full pytest: **8 passed**.
- Ruff: clean.
- `pip check`: no broken requirements.
- `scripts/mcp_smoke.py`: passed.
- `scripts/v1c_smoke.py`: passed against real `caseflow_test` PostgreSQL/pgvector and approval roles.
- `docker compose config -q`: passed.
- The GitHub Actions `workflow_dispatch` workflow was not dispatched.

## J. Live model request

The official read-only OpenRouter key endpoint, called before and after the run without printing or storing the key, confirmed sufficient free quota for the bounded run. The official model catalog reported `nvidia/nemotron-3.5-super-vl:free` prompt and completion price as zero. No fallback model was configured.

One standard Beijing incident investigation completed in Live mode. It used 3 successful model requests (two tool decisions, one JSON-schema synthesis), no retries or paid fallback, and reported 6,388 input + 2,020 output = 8,408 tokens. End-to-end duration was 23.343 seconds; the route price was $0.

Tool order and arguments:

1. `query_payment_metrics`: region 北京, 2025-01-21 09:00–12:00 Asia/Shanghai, grouped by service/status → 22/50 failures (44.00%) versus 10/330 (3.03%) baseline.
2. `query_change_events`: region 北京, 07:00–12:00 → `CHG-20250121-BJ-ROUTE-01` and `INCIDENT-INC-20250121-BJ-01`.
3. `search_knowledge`, top 5 → `KNOW-DOC-RETRY-001-chunk-000`, `KNOW-DOC-PAY-002-chunk-000`, `KNOW-DOC-INC-001-chunk-000`, `KNOW-DOC-BJ-001-chunk-000`, and `KNOW-DOC-PAY-001-chunk-000`.

The report resolved all 3 citations to presented evidence; all 3 quote-fidelity checks passed, with no rejected citations or unsupported numeric claims. It reported two tentative hypotheses, stated that timing does not prove causality, and did not claim a verified root cause. Semantic support for the two hypotheses remains ungraded by a human. No second independent Live case was attempted with only two free requests remaining.

## K. GitHub and PR

The V1-C changes and this report are on the existing feature branch in PR #1. The branch head was unchanged remotely before this acceptance update; after the normal push, the local and remote SHA were verified equal. PR #1 remains open and is not merged.

## L. Merge recommendation and remaining boundaries

**Recommendation:** suitable to merge for the stated local V1-C engineering-demo scope after this acceptance commit is pushed. Keep the PR description explicit that this is not production-ready.

Remaining boundaries are the lack of user authentication/multi-user authorization, no configured real Embedding provider or semantic retrieval evaluation, and no human grading of Live answer entailment. The UI remains loopback-only and human approval is required for ticket creation.
