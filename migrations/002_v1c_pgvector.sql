\set ON_ERROR_STOP on

CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS caseflow_vectors AUTHORIZATION caseflow_seed;
REVOKE ALL ON SCHEMA caseflow_vectors FROM PUBLIC;
GRANT USAGE ON SCHEMA caseflow_vectors TO caseflow_app, caseflow_embedding_indexer;

CREATE TABLE IF NOT EXISTS caseflow_vectors.document_embeddings (
    document_id varchar(80) NOT NULL,
    chunk_id varchar(40) NOT NULL,
    title varchar(240) NOT NULL,
    content text NOT NULL,
    content_hash varchar(64) NOT NULL,
    embedding_model varchar(240) NOT NULL,
    dimensions integer NOT NULL CHECK (dimensions BETWEEN 1 AND 2000),
    embedding vector NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (document_id, chunk_id, embedding_model),
    CHECK (dimensions = vector_dims(embedding))
);

CREATE INDEX IF NOT EXISTS ix_document_embeddings_model_dims
    ON caseflow_vectors.document_embeddings (embedding_model, dimensions);
CREATE INDEX IF NOT EXISTS ix_document_embeddings_cosine
    ON caseflow_vectors.document_embeddings USING hnsw (embedding vector_cosine_ops);

REVOKE ALL ON ALL TABLES IN SCHEMA caseflow_vectors FROM PUBLIC, caseflow_seed, caseflow_action_executor;
GRANT SELECT ON caseflow_vectors.document_embeddings TO caseflow_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON caseflow_vectors.document_embeddings TO caseflow_embedding_indexer;
