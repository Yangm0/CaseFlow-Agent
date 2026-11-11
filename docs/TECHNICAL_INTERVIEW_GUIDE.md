# CaseFlow technical interview guide

## Why LangGraph rather than a simple Chain?

The workflow needs a bounded tool-use loop and a durable pause/resume point. A chain would sequence a fixed set of operations, but here the model can choose among three allowlisted MCP tools, inspect returned evidence, and decide whether to investigate further or synthesize. LangGraph carries the investigation state and checkpoints the thread. `interrupt()` pauses before a business action and the API resumes it with the human's decision. This is a small graph, not a multi-agent system.

## What does MCP add here?

The Agent discovers tool names and JSON schemas from an independent Streamable HTTP server, then calls the same tools through the MCP client. The service boundary makes capabilities inspectable and testable and avoids giving the model arbitrary SQL or write access. It is a meaningful interface boundary in this project, not a claim that MCP is inherently safer than every direct function call; database roles and server-side validation enforce the actual security boundary.

## How does checkpoint recovery work?

Each thread uses a `thread_id`; LangGraph's PostgreSQL checkpointer stores graph state, while the application stores user/assistant history and the latest report. The API can load the thread and resume from its saved checkpoint after process restart. A pending approval is registered separately and the graph is interrupted before the side effect. Intermediate SSE events are held in a bounded process-local journal, so a restarted process restores durable report/history but not the old in-memory event stream.

## Why can the model not approve its own ticket?

The model is an untrusted proposer. A proposal must cite IDs present in the server's evidence catalog, and the server binds it to the source investigation. The approval row stores a payload hash. Editing supersedes the earlier pending proposal. A human decision resumes the graph; only the action-executor role can call the narrow database function. The function locks and checks status/hash and creates at most one ticket while writing audit events transactionally. Repeated approvals are idempotent; rejected and superseded drafts cannot execute. The app has no user authentication, so this control is suitable only for a local demo, not a multi-user deployment.

## How does hybrid retrieval work?

BM25 ranks the Chinese document chunks. FastEmbed's local `BAAI/bge-small-zh-v1.5` creates 512D query/document vectors stored in pgvector. A model-and-dimension-specific HNSW expression/partial index supports cosine distance. Vector and BM25 ranked lists are combined with Reciprocal Rank Fusion, stable evidence IDs are deduplicated, and stale content hashes are excluded. If the embedding provider or matching index is unavailable, the tool returns BM25 results and indicates fallback.

## What did evaluation show?

Nine labeled synthetic queries were divided into development (six) and validation (three). At top five, BM25 / vector / hybrid validation Recall@5 was 0.667 / 0.500 / 0.833; MRR was 0.667 / 0.333 / 0.444; nDCG@5 was 0.765 / 0.500 / 0.692. One validation query has no relevant label, so Recall and nDCG use two scored queries, while MRR uses all three. Hybrid gained recall but lost rank-sensitive metrics versus BM25. The set is too small to make a general quality claim; no reranking was added.

## How are database permissions and idempotence enforced?

Business-query credentials are read-only. The MCP tool allowlist accepts bounded parameters rather than arbitrary SQL. A separate checkpoint role operates in its dedicated schema. The embedding indexer writes only to the vector table and can invoke the restricted index-creation function. The action executor has no direct table DML and receives execute rights on narrowly scoped `SECURITY DEFINER` functions. Those functions validate pending status and payload hash under row lock, create at most one ticket per approval, and insert audit entries in the same transaction.

## What are the model limitations?

The one Live case used three requests and produced 4/4 identifier/quote-fidelity checks, but semantic entailment of two hypotheses was not graded. The model described a possible connection between a nearby change and an upstream timeout; the report labels that as an unverified hypothesis and lists missing failure-code/channel evidence. Temporal proximity is not causality. There was no second Live case, no broad LLM benchmark, and no claim that a correct citation alone proves the conclusion. All data and actions were synthetic.
