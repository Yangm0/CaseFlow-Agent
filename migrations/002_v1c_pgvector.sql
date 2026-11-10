\set ON_ERROR_STOP on

CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS caseflow_vectors AUTHORIZATION caseflow_vector_owner;
ALTER SCHEMA caseflow_vectors OWNER TO caseflow_vector_owner;
REVOKE ALL ON SCHEMA caseflow_vectors FROM PUBLIC, caseflow_seed, caseflow_action_executor;
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

ALTER TABLE caseflow_vectors.document_embeddings OWNER TO caseflow_vector_owner;

-- HNSW requires a fixed vector dimension. Provision one expression/partial index for each
-- configured model and dimension via the restricted embedding-indexer function below.
CREATE OR REPLACE FUNCTION caseflow_vectors.ensure_hnsw_index(
    p_model text,
    p_dimensions integer
) RETURNS text
LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, caseflow_vectors
AS $$
DECLARE
    index_name text;
BEGIN
    IF p_model IS NULL OR length(p_model) NOT BETWEEN 1 AND 240
       OR p_dimensions IS NULL OR p_dimensions NOT BETWEEN 1 AND 2000 THEN
        RAISE EXCEPTION 'invalid embedding model or dimensions';
    END IF;

    index_name := 'ix_doc_emb_hnsw_' || substr(md5(p_model || ':' || p_dimensions::text), 1, 24);
    EXECUTE format(
        'CREATE INDEX IF NOT EXISTS %I ON caseflow_vectors.document_embeddings '
        'USING hnsw ((embedding::public.vector(%s)) public.vector_cosine_ops) '
        'WHERE embedding_model = %L AND dimensions = %s',
        index_name,
        p_dimensions,
        p_model,
        p_dimensions
    );
    RETURN index_name;
END;
$$;

ALTER FUNCTION caseflow_vectors.ensure_hnsw_index(text, integer)
    OWNER TO caseflow_vector_owner;
REVOKE ALL ON FUNCTION caseflow_vectors.ensure_hnsw_index(text, integer) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION caseflow_vectors.ensure_hnsw_index(text, integer)
    TO caseflow_embedding_indexer;

REVOKE ALL ON ALL TABLES IN SCHEMA caseflow_vectors FROM PUBLIC, caseflow_seed, caseflow_action_executor;
GRANT SELECT ON caseflow_vectors.document_embeddings TO caseflow_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON caseflow_vectors.document_embeddings TO caseflow_embedding_indexer;
