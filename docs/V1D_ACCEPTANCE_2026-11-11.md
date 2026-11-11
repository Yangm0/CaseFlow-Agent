# CaseFlow V1-D acceptance record — 2026-11-11

This record distinguishes executed checks from model-quality claims. All databases and examples used in this acceptance were synthetic. The existing PostgreSQL data volume was preserved.

## Environment and preservation

- Windows 11, Python 3.13.2, existing `.venv`, Docker Desktop with WSL 2, Docker Compose 5.7.0.
- Docker 29.9.3; PostgreSQL 17.11; pgvector 0.8.7.
- Readable custom-format backups were made before changes: `caseflow_test` (about 1.23 MB) and `caseflow` (about 1.11 MB). `pg_restore --list` could read each archive.
- The existing PostgreSQL data volume was not removed, reset, or reinitialized. Only API/MCP containers were rebuilt. The embedding cache is a separate named volume.
- Both databases retained 49,600 payment attempts, 18 Chinese documents, 7 change records, and 4 incidents. The pre-existing deterministic 64D test vectors were retained under their separate model label.

## V1-D action workflow

The Agent proposes a structured draft with title, description, priority, region, investigation ID, supporting evidence IDs, and rationale. The API validates proposal evidence against the report, binds the proposal to the investigation, persists a pending approval, and interrupts the graph. Editing creates a new payload hash and supersedes the prior pending payload. A pending action blocks a new message on that thread.

`scripts/v1d_smoke.py` ran against `caseflow_test` using the real MCP server, PostgreSQL, checkpoint store, action functions, and API. It verified a high-priority Beijing proposal supported by the 22/50 metric; pending-state interruption; rejection idempotence; stale hash rejection; API/checkpoint restart recovery; approval creating one ticket; duplicate approval returning that ticket; and exact tagged-artifact cleanup. After cleanup, the test database contained no approval, ticket, or audit rows from the smoke.

The Live browser case separately created one demo approval, which was approved by a human and created ticket `0ae77363-ebe6-487d-a01d-755abd0b0f29`. Its approval record is `3be94ae2-bd47-4d1c-aac0-b8cb3f142df4`; the associated report trace is `117d9612-fb76-457c-b798-596554274a5b`. The dev database has that one demonstration ticket and its approval/audit record. No real payment data was written.

## Embedding and retrieval

The configured provider was FastEmbed CPU using `BAAI/bge-small-zh-v1.5` with 512 dimensions. No GPU framework or remote embedding API was used. The model produced actual semantic embeddings (not deterministic hash vectors). Indexing created 18 chunks in `caseflow_test` and 18 in `caseflow`; a repeat development indexing run reported 18 documents, 18 chunks, 0 newly indexed chunks, 0 deleted chunks.

The query path casts the stored `vector` to the configured `vector(512)` and uses the same model and dimensions in its filter and cosine-distance ordering. `EXPLAIN` of that query selected `ix_doc_emb_hnsw_74336d1a997de02e229e915e`; sequential scans were disabled only for this plan inspection because the corpus is intentionally small. The index is a real HNSW expression/partial index, not a claim inferred from source code alone. A post-index MCP query returned `retrieval_mode=hybrid`, sources `bm25` and `pgvector`, and no fallback warning.

Nine existing labeled synthetic queries were scored at `k=5` with the same labels. Recall and nDCG exclude the single query without relevant labels; MRR includes all queries. Values are means.

| Retriever | Development Recall@5 (6 scored) | Development MRR (6) | Development nDCG@5 (6) | Validation Recall@5 (2 scored) | Validation MRR (3) | Validation nDCG@5 (2) |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.7222 | 1.0000 | 0.7336 | 0.6667 | 0.6667 | 0.7654 |
| Vector | 0.8889 | 1.0000 | 0.8870 | 0.5000 | 0.3333 | 0.5000 |
| Hybrid RRF | 0.8333 | 1.0000 | 0.8533 | 0.8333 | 0.4444 | 0.6918 |

Hybrid improved validation Recall@5 but reduced validation MRR and nDCG against BM25. The small synthetic evaluation does not support a broad quality claim. No labels or RRF weights were changed to improve these scores. No reranker was added.

## Live investigation and action

Before inference, the official OpenRouter key endpoint reported 50 free model requests available; the model catalog showed `nvidia/nemotron-3.5-super-vl:free` with prompt and completion price `0.0`. A final read-only quota query reported 3 used of 50 and 47 remaining. The request set `provider.allow_fallbacks=false` and no paid model fallback. The API key came from the existing host process environment, was never printed, written to `.env`, or passed into the Docker API container. Three successful model requests were made in total; there were no failed calls, retries, or structured-output fallbacks. The model was not called again after the investigation.

- Model route: `nvidia/nemotron-3.5-super-vl:free`.
- Requests: 3; input tokens 6,506; output tokens 2,433; total 8,939; elapsed 17.818 seconds; reported price $0.
- Tool order: `query_payment_metrics` → `query_change_events` → `search_knowledge`.
- Metrics arguments: Beijing, `2025-01-21T09:00:00+08:00` through `12:00:00+08:00`, grouped by service and status. MCP/SQL result: 22 failed out of 50 valid attempts (44%); baseline 10/330 (3.03%). The `payment-router` group accounts for the 22 failures.
- Change lookup extended from 07:00 to 12:00 Beijing time. It found change `CHANGE-CHG-20250121-BJ-ROUTE-01` at 08:52, deploying payment-router 2.8.4 and changing Beijing UnionPay priority/retry settings.
- Knowledge/incident evidence included `INCIDENT-INC-20250121-BJ-01`, recorded at 09:07 with an upstream-timeout alert and investigation still in progress.
- Citation validation: 4 citations retained from 4 presented; 0 unresolved/rejected; 4 quote-fidelity checks with 0 failures. Evidence IDs included metric `MET-83083366b6c34c32`, the change and incident IDs above, and five `KNOW-*` document chunks.
- The report marked the two causal explanations as low-confidence hypotheses; semantic support was not graded. It did not present change/incident time overlap as proven causation. It called out missing failure-code and per-channel-rate evidence.
- The Agent proposed a high-priority Beijing support ticket with report-bound evidence. Human approval created one synthetic ticket; the UI then displayed the executed status and ticket ID. The `approval_created`, `approved`, and `ticket_created` audit path was exercised.
- The thread and report were fetched after a host API restart. A report-history inconsistency discovered during refresh was fixed so the saved assistant summary now agrees with the action state; browser reload showed the approval result and stored tool/model trace. Browser console had no errors. The Compose API was separately kept in offline mode.

This was a bounded engineering smoke, not a statistical LLM evaluation. Citation/quote matching does not establish entailment or causal correctness. No additional inference call was made for a second Live case.

## Offline integration and checks

- PostgreSQL 17.11 and pgvector 0.8.7 remained healthy; Docker Compose config validated. API and MCP were rebuilt without replacing PostgreSQL or its volume. Compose host ports were set locally to 8100 and 8101 because another local service occupied port 8000.
- MCP smoke found exactly three allowlisted tools and checked their schemas. It exercised invalid arguments, structured content, 22/50 data through the MCP client, and `MCP_UNAVAILABLE` after disconnect.
- V1-C smoke passed HNSW selection, BM25 fallback, add/update/delete synchronization, approval hash/idempotence/concurrency, restart, orphan, and read-only database permission checks.
- V1-D smoke passed the automatic action draft and approval workflow described above.
- `ruff check src scripts tests`: passed. `node --check web/app.js`: passed. Pytest collection: exactly 8; full run: 8 passed. `pip check`: passed. Compose config: passed.
- The browser used the actual UI and API, not only static asset inspection: it submitted an investigation, rendered stored report/tool evidence and the pending action, executed a human decision, reloaded after process restart, and restored durable state. SSE re-connect/replay was exercised in the prior V1-C browser acceptance; the journal remains process-local.
- The V1-D GitHub Actions workflow was configured for pull requests and pushes to `main`. Its hosted run result is recorded separately after the PR workflow completes; local success is not represented as a hosted CI result.

## Known limits and handoff

- Nine retrieval labels are too few for broad semantic claims; validation set has three queries and only two with relevant documents.
- Live model hypotheses require human review. No causal model evaluation or second-case generalization test is claimed.
- The app has no authentication or multi-tenant isolation. Keep all services loopback-only.
- SSE intermediate events are process-local, although thread history, report, and graph checkpoint survive API restart.
- Synthetic-only support-ticket creation is implemented; there are no production payment mutations, refunds, routing changes, or rollback actions.
- The BGE local model cache is a local Docker named volume and is not committed. Fresh deployments need the optional model download on first indexing.

## Reproduction commands

```powershell
ruff check src scripts tests
.\.venv\Scripts\python.exe -m pytest --collect-only -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\mcp_smoke.py
.\.venv\Scripts\python.exe scripts\v1d_smoke.py
.\.venv\Scripts\python.exe scripts\evaluate_retrieval.py --split all --retriever bm25
.\.venv\Scripts\python.exe scripts\evaluate_retrieval.py --split all --retriever vector
.\.venv\Scripts\python.exe scripts\evaluate_retrieval.py --split all --retriever hybrid
```
