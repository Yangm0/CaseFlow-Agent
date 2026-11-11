from __future__ import annotations

import hashlib
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.engine import make_url

from caseflow.approvals.service import (
    ActionStore,
    ApprovalDecision,
    ApprovalWorkflow,
    TicketDraft,
    make_action_engine,
)
from caseflow.checkpoints.runtime import open_checkpointer
from caseflow.config import get_settings
from caseflow.db import make_engine, readonly_transaction
from caseflow.retrieval.vector import EmbeddingIndexer, VectorSearchService
from caseflow.schemas import KnowledgeSearchInput
from caseflow.tools.knowledge import KnowledgeService


class DeterministicTestEmbeddings:
    """A local hash-vector fixture for database plumbing, not semantic evaluation."""

    model = "caseflow-local-deterministic-acceptance-v1"
    dimensions = 64

    @classmethod
    def embed_documents(cls, texts: list[str]) -> list[list[float]]:
        return [cls.embed_query(item) for item in texts]

    @classmethod
    def embed_query(cls, value: str) -> list[float]:
        vector = [0.0] * cls.dimensions
        for token in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+", value.casefold()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % cls.dimensions
            vector[bucket] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


class UnavailableVectorSearch:
    def search(self, query, chunks, top_k):
        raise RuntimeError("Deterministic acceptance check for unavailable vector service.")


def _test_url(value: str | None, label: str) -> str:
    if not value:
        raise RuntimeError(f"{label} is not configured.")
    url = make_url(value)
    if url.database != "caseflow_test":
        raise RuntimeError(f"{label} must target caseflow_test; refusing to continue.")
    return value


def _test_role_url(value: str | None, label: str) -> str:
    if not value:
        raise RuntimeError(f"{label} is not configured.")
    url = make_url(value)
    if url.database not in {"caseflow", "caseflow_test"}:
        raise RuntimeError(f"{label} must target a CaseFlow database; refusing to continue.")
    return url.set(database="caseflow_test").render_as_string(hide_password=False)


def _check_hnsw_plan(engine, provider: DeterministicTestEmbeddings) -> list[str]:
    model_literal = "'" + provider.model.replace("'", "''") + "'"
    sql = text(
        "EXPLAIN (COSTS OFF) SELECT document_id, chunk_id "
        "FROM caseflow_vectors.document_embeddings "
        f"WHERE embedding_model = {model_literal} AND dimensions = {provider.dimensions} "
        "ORDER BY embedding::vector(64) <=> CAST(:embedding AS vector(64)) LIMIT 5"
    )
    with engine.connect() as connection, connection.begin():
        connection.exec_driver_sql("SET LOCAL enable_seqscan = off")
        plan = (
            connection.execute(
                sql,
                {
                    "embedding": "["
                    + ",".join(map(str, provider.embed_query("北京支付路由超时")))
                    + "]",
                },
            )
            .scalars()
            .all()
        )
    output = [str(line) for line in plan]
    if not any("ix_doc_emb_hnsw_" in line and "Index Scan" in line for line in output):
        raise AssertionError(f"Vector query did not use its fixed-dimension HNSW index: {output}")
    return output


def _vector_smoke(read_engine, seed_engine, writer_engine, settings) -> dict[str, object]:
    provider = DeterministicTestEmbeddings()
    query = "北京 支付路由 超时 重试"
    baseline = KnowledgeService(read_engine, settings.sql_statement_timeout_ms).search(
        KnowledgeSearchInput(query=query, top_k=5)
    )
    if baseline["retrieval_mode"] != "bm25_only" or not baseline["results"]:
        raise AssertionError("BM25 baseline failed before vector indexing.")
    fallback = KnowledgeService(
        read_engine,
        settings.sql_statement_timeout_ms,
        vector_search=UnavailableVectorSearch(),
    ).search(KnowledgeSearchInput(query=query, top_k=5))
    if fallback["retrieval_mode"] != "bm25_only" or not fallback["results"]:
        raise AssertionError("BM25 did not recover when Vector Search was unavailable.")

    indexer = EmbeddingIndexer(read_engine, writer_engine, provider)
    first_index = indexer.index_documents()
    second_index = indexer.index_documents()
    if first_index["chunks"] < 18 or second_index["indexed_chunks"] != 0:
        raise AssertionError(
            f"Vector indexing or idempotence failed: {first_index}, {second_index}"
        )

    search = VectorSearchService(read_engine, provider)
    hybrid = KnowledgeService(
        read_engine,
        settings.sql_statement_timeout_ms,
        vector_search=search,
    ).search(KnowledgeSearchInput(query=query, top_k=5))
    if hybrid["retrieval_mode"] != "hybrid" or not hybrid["results"]:
        raise AssertionError(f"Hybrid retrieval did not return results: {hybrid}")
    sources = {source for row in hybrid["results"] for source in row["retrieval_sources"]}
    if not {"bm25", "pgvector"}.issubset(sources):
        raise AssertionError(f"RRF did not merge both retrieval sources: {sources}")
    with read_engine.connect() as connection:
        valid_document_ids = {
            str(document_id)
            for document_id in connection.execute(
                text("SELECT document_id FROM documents")
            ).scalars()
        }
    if any(row["document_id"] not in valid_document_ids for row in hybrid["results"]):
        raise AssertionError("Hybrid retrieval returned an unresolvable document ID.")
    plan = _check_hnsw_plan(read_engine, provider)

    smoke_document_id = f"V1C-SMOKE-{uuid4().hex}"
    cleanup_counts: dict[str, int] = {}
    try:
        with seed_engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO documents (document_id, title, source_type, content, updated_at) "
                    "VALUES (:id, :title, 'acceptance_smoke', :content, :updated_at)"
                ),
                {
                    "id": smoke_document_id,
                    "title": "Smoke 文档初始版本",
                    "content": "初始内容用于检查新增文档的向量写入。",
                    "updated_at": datetime.now(UTC),
                },
            )
        added = indexer.index_documents()
        with seed_engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE documents SET title = :title, content = :content, "
                    "updated_at = :updated_at "
                    "WHERE document_id = :id"
                ),
                {
                    "id": smoke_document_id,
                    "title": "Smoke 文档更新版本",
                    "content": "更新后的内容用于检查向量重建与内容哈希同步。",
                    "updated_at": datetime.now(UTC),
                },
            )
        updated = indexer.index_documents()
        with writer_engine.connect() as connection:
            indexed_content = connection.execute(
                text(
                    "SELECT content FROM caseflow_vectors.document_embeddings "
                    "WHERE document_id = :id AND embedding_model = :model"
                ),
                {"id": smoke_document_id, "model": provider.model},
            ).scalar_one()
        if indexed_content != "更新后的内容用于检查向量重建与内容哈希同步。":
            raise AssertionError("Updated document text was not refreshed in the vector table.")
        with seed_engine.begin() as connection:
            connection.execute(
                text("DELETE FROM documents WHERE document_id = :id"),
                {"id": smoke_document_id},
            )
        deleted = indexer.index_documents()
        with writer_engine.connect() as connection:
            remaining = connection.execute(
                text(
                    "SELECT count(*) FROM caseflow_vectors.document_embeddings "
                    "WHERE document_id = :id AND embedding_model = :model"
                ),
                {"id": smoke_document_id, "model": provider.model},
            ).scalar_one()
        if remaining != 0:
            raise AssertionError("Deleted source document left stale vector rows.")
        cleanup_counts = {
            "added_chunks": added["indexed_chunks"],
            "updated_chunks": updated["indexed_chunks"],
            "deleted_chunks": deleted["deleted_chunks"],
            "stale_rows_after_delete": int(remaining),
        }
    finally:
        with seed_engine.begin() as connection:
            connection.execute(
                text("DELETE FROM documents WHERE document_id = :id"),
                {"id": smoke_document_id},
            )
        indexer.index_documents()

    return {
        "provider": provider.model,
        "dimensions": provider.dimensions,
        "semantic_embedding_evaluation": False,
        "embedding_api_configured": bool(
            settings.embedding_api_key
            and settings.embedding_api_key.get_secret_value().strip()
            and settings.embedding_model
            and settings.embedding_dimensions
        ),
        "bm25_mode": baseline["retrieval_mode"],
        "unavailable_vector_fallback_mode": fallback["retrieval_mode"],
        "unavailable_vector_fallback_warning": fallback["retrieval_warning"],
        "bm25_evidence_ids": [item["evidence_id"] for item in baseline["results"]],
        "index_first_run": first_index,
        "index_second_run": second_index,
        "hybrid_mode": hybrid["retrieval_mode"],
        "hybrid_retriever": hybrid["retriever"],
        "hybrid_evidence_ids": [item["evidence_id"] for item in hybrid["results"]],
        "hybrid_sources": sorted(sources),
        "hnsw_plan": plan,
        "mutation_consistency": cleanup_counts,
    }


def _approval_smoke(settings, app_engine) -> dict[str, object]:
    action_url = _test_role_url(
        settings.action_executor_database_url, "ACTION_EXECUTOR_DATABASE_URL"
    )
    action_engine = make_action_engine(action_url)
    smoke_tag = uuid4().hex
    store = ActionStore(action_engine)
    draft = TicketDraft(
        title="V1-C 审批幂等验收",
        description="只用于 caseflow_test 审批工作流验收，结束后会精确清理。",
        priority="low",
        region="北京",
        source_investigation_id=f"smoke-{smoke_tag}",
        supporting_evidence_ids=[f"METRIC-{smoke_tag}"],
        rationale="合成指标高于对照基线，需要值班人员复核。",
    )
    results: dict[str, object] = {"smoke_tag": smoke_tag}
    try:
        reject_thread = f"v1c-smoke-{smoke_tag}-reject"
        rejected = store.create_pending(reject_thread, draft)
        duplicate = store.create_pending(reject_thread, draft)
        if rejected["approval_id"] != duplicate["approval_id"]:
            raise AssertionError("Duplicate pending draft created a second approval.")
        results["duplicate_pending_same_id"] = True
        results["reject_approval_id"] = str(rejected["approval_id"])

        try:
            store.approve_and_execute(rejected["approval_id"], "0" * 64)
        except Exception:
            results["tampered_payload_hash_rejected"] = True
        else:
            raise AssertionError("Tampered approval payload hash was accepted.")
        if store.get(rejected["approval_id"])["status"] != "pending":
            raise AssertionError("Tampered hash changed the pending approval state.")

        edited_draft = draft.model_copy(
            update={"title": "V1-C 修改后的审批草稿", "priority": "medium"}
        )
        previous_id = str(rejected["approval_id"])
        rejected = store.revise_pending(
            previous_id,
            str(rejected["payload_hash"]),
            edited_draft,
        )
        if (
            str(rejected["approval_id"]) == previous_id
            or rejected["payload_hash"] == duplicate["payload_hash"]
            or store.get(previous_id)["status"] != "superseded"
        ):
            raise AssertionError(
                "Editing a pending draft did not invalidate the previous approval."
            )
        try:
            store.approve_and_execute(previous_id, str(duplicate["payload_hash"]))
        except Exception:
            results["superseded_payload_cannot_execute"] = True
        else:
            raise AssertionError("A superseded payload unexpectedly remained executable.")
        results["edited_payload_has_new_approval_and_hash"] = True

        with open_checkpointer(
            _test_url(settings.test_checkpoint_database_url, "TEST_CHECKPOINT_DATABASE_URL")
        ) as checkpointer:
            workflow = ApprovalWorkflow(checkpointer, store)
            workflow.ensure_pending_interrupt(rejected)
            if not workflow.graph.get_state(workflow.config(str(rejected["approval_id"]))).next:
                raise AssertionError("Checkpoint did not interrupt for human rejection.")
            reject_checkpoint = str(rejected["approval_id"])
        # Reopen the saver to simulate a restart while approval is pending.
        with open_checkpointer(
            _test_url(settings.test_checkpoint_database_url, "TEST_CHECKPOINT_DATABASE_URL")
        ) as checkpointer:
            workflow = ApprovalWorkflow(checkpointer, store)
            pending_snapshot = workflow.graph.get_state(workflow.config(reject_checkpoint))
            if not pending_snapshot.next:
                raise AssertionError(
                    "Pending approval interrupt did not survive checkpointer restart."
                )
            reject_result = workflow.resume(
                reject_checkpoint,
                ApprovalDecision(decision="reject"),
            )["result"]
        if reject_result["status"] != "rejected":
            raise AssertionError(f"Reject workflow returned {reject_result}.")
        if store.reject(reject_checkpoint, rejected["payload_hash"])["status"] != "rejected":
            raise AssertionError("Repeated rejection was not idempotent.")
        try:
            store.approve_and_execute(reject_checkpoint, rejected["payload_hash"])
        except Exception:
            results["rejected_cannot_create_ticket"] = True
        else:
            raise AssertionError("Rejected approval unexpectedly created a ticket.")
        results["rejection_after_restart"] = reject_result

        approve_thread = f"v1c-smoke-{smoke_tag}-approve"
        approved = store.create_pending(
            approve_thread,
            draft.model_copy(update={"source_investigation_id": f"smoke-approve-{smoke_tag}"}),
        )
        results["approve_approval_id"] = str(approved["approval_id"])
        with open_checkpointer(
            _test_url(settings.test_checkpoint_database_url, "TEST_CHECKPOINT_DATABASE_URL")
        ) as checkpointer:
            workflow = ApprovalWorkflow(checkpointer, store)
            workflow.ensure_pending_interrupt(approved)
        with open_checkpointer(
            _test_url(settings.test_checkpoint_database_url, "TEST_CHECKPOINT_DATABASE_URL")
        ) as checkpointer:
            workflow = ApprovalWorkflow(checkpointer, store)
            if not workflow.graph.get_state(workflow.config(str(approved["approval_id"]))).next:
                raise AssertionError(
                    "Approval did not persist as a pending interrupt across restart."
                )
            approve_result = workflow.resume(
                str(approved["approval_id"]), ApprovalDecision(decision="approve")
            )["result"]
        if approve_result["status"] != "executed" or not approve_result.get("ticket_id"):
            raise AssertionError(f"Approval did not create exactly one ticket: {approve_result}")
        duplicate_results: list[dict[str, object]]
        with ThreadPoolExecutor(max_workers=4) as executor:
            duplicate_results = list(
                executor.map(
                    lambda _: store.approve_and_execute(
                        str(approved["approval_id"]), approved["payload_hash"]
                    ),
                    range(4),
                )
            )
        ticket_ids = {str(item.get("ticket_id")) for item in duplicate_results}
        if ticket_ids != {str(approve_result["ticket_id"])}:
            raise AssertionError(
                f"Repeated concurrent approval changed ticket identity: {ticket_ids}"
            )
        results["approval_after_restart"] = approve_result
        results["parallel_duplicate_approvals_same_ticket"] = True

        orphan_thread = f"v1c-smoke-{smoke_tag}-recovery"
        orphan = store.create_pending(
            orphan_thread,
            draft.model_copy(update={"source_investigation_id": f"smoke-recovery-{smoke_tag}"}),
        )
        with open_checkpointer(
            _test_url(settings.test_checkpoint_database_url, "TEST_CHECKPOINT_DATABASE_URL")
        ) as checkpointer:
            workflow = ApprovalWorkflow(checkpointer, store)
            repaired = workflow.ensure_pending_interrupt(orphan)
            if not workflow.graph.get_state(workflow.config(str(orphan["approval_id"]))).next:
                raise AssertionError("A DB-only pending approval did not recover its checkpoint.")
            orphan_result = workflow.resume(
                str(orphan["approval_id"]), ApprovalDecision(decision="reject")
            )["result"]
        results["orphan_checkpoint_recovered"] = bool(repaired)
        results["orphan_rejected_without_ticket"] = orphan_result["status"] == "rejected"
        results["recovery_approval_id"] = str(orphan["approval_id"])

        try:
            with app_engine.begin() as connection:
                connection.execute(
                    text(
                        "INSERT INTO caseflow_actions.support_tickets "
                        "(approval_id, title, description, priority, region, "
                        "source_investigation_id) "
                        "VALUES (gen_random_uuid(), 'unauthorized', 'no direct insert allowed', "
                        "'low', '北京', 'smoke')"
                    )
                )
        except Exception:
            results["business_read_role_direct_write_denied"] = True
        else:
            raise AssertionError("Business read role could directly insert a support ticket.")
    finally:
        action_engine.dispose()
    return results


def main() -> int:
    settings = get_settings()
    read_url = _test_url(settings.test_database_url, "TEST_DATABASE_URL")
    seed_url = _test_url(settings.test_seed_database_url, "TEST_SEED_DATABASE_URL")
    writer_url = _test_role_url(
        settings.embedding_index_database_url, "EMBEDDING_INDEX_DATABASE_URL"
    )
    read_engine = make_engine(read_url)
    seed_engine = make_engine(seed_url)
    writer_engine = make_engine(writer_url)
    result: dict[str, object] = {"database": "caseflow_test"}
    try:
        with readonly_transaction(read_engine, settings.sql_statement_timeout_ms) as connection:
            version, extension = connection.execute(
                text(
                    "SELECT current_setting('server_version'), "
                    "(SELECT extversion FROM pg_extension WHERE extname = 'vector')"
                )
            ).one()
        if not extension:
            raise AssertionError("pgvector extension is not installed in caseflow_test.")
        result["postgres_version"] = str(version)
        result["pgvector_version"] = str(extension)
        result["vector_rag"] = _vector_smoke(read_engine, seed_engine, writer_engine, settings)
        result["approvals"] = _approval_smoke(settings, read_engine)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    finally:
        read_engine.dispose()
        seed_engine.dispose()
        writer_engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
