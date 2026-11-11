#!/bin/sh
set -eu

psql -v ON_ERROR_STOP=1 \
  --username "$POSTGRES_USER" \
  --dbname "$POSTGRES_DB" \
  -v app_password="$CASEFLOW_APP_PASSWORD" \
  -v seed_password="$CASEFLOW_SEED_PASSWORD" \
  -v checkpoint_password="$CASEFLOW_CHECKPOINT_PASSWORD" \
  -v action_password="$CASEFLOW_ACTION_PASSWORD" \
  -v embedding_password="$CASEFLOW_EMBEDDING_INDEX_PASSWORD" <<'SQL'
CREATE ROLE caseflow_app LOGIN PASSWORD :'app_password'
  NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;
CREATE ROLE caseflow_seed LOGIN PASSWORD :'seed_password'
  NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;
CREATE ROLE caseflow_checkpoint LOGIN PASSWORD :'checkpoint_password'
  NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;
CREATE ROLE caseflow_action_executor LOGIN PASSWORD :'action_password'
  NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;
CREATE ROLE caseflow_action_owner NOLOGIN
  NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;
CREATE ROLE caseflow_embedding_indexer LOGIN PASSWORD :'embedding_password'
  NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;
GRANT CONNECT ON DATABASE caseflow TO caseflow_app, caseflow_seed, caseflow_checkpoint,
  caseflow_action_executor, caseflow_embedding_indexer;
CREATE DATABASE caseflow_test OWNER caseflow_admin;
GRANT CONNECT ON DATABASE caseflow_test TO caseflow_app, caseflow_seed, caseflow_checkpoint,
  caseflow_embedding_indexer;
GRANT USAGE ON SCHEMA public TO caseflow_app;
GRANT USAGE, CREATE ON SCHEMA public TO caseflow_seed;
CREATE SCHEMA caseflow_checkpoint AUTHORIZATION caseflow_checkpoint;
GRANT USAGE, CREATE ON SCHEMA caseflow_checkpoint TO caseflow_checkpoint;
ALTER ROLE caseflow_checkpoint IN DATABASE caseflow SET search_path = caseflow_checkpoint, public;
CREATE SCHEMA caseflow_actions AUTHORIZATION caseflow_action_owner;
CREATE SCHEMA caseflow_vectors AUTHORIZATION caseflow_seed;
CREATE EXTENSION IF NOT EXISTS vector;
GRANT USAGE ON SCHEMA caseflow_vectors TO caseflow_app, caseflow_embedding_indexer;
GRANT USAGE ON SCHEMA caseflow_actions TO caseflow_seed;
SQL

psql -v ON_ERROR_STOP=1 \
  --username "$POSTGRES_USER" \
  --dbname caseflow_test \
  -v checkpoint_password="$CASEFLOW_CHECKPOINT_PASSWORD" <<'SQL'
GRANT USAGE ON SCHEMA public TO caseflow_app;
GRANT USAGE, CREATE ON SCHEMA public TO caseflow_seed;
CREATE SCHEMA caseflow_checkpoint AUTHORIZATION caseflow_checkpoint;
GRANT USAGE, CREATE ON SCHEMA caseflow_checkpoint TO caseflow_checkpoint;
ALTER ROLE caseflow_checkpoint IN DATABASE caseflow_test SET search_path = caseflow_checkpoint, public;
CREATE SCHEMA caseflow_vectors AUTHORIZATION caseflow_seed;
CREATE EXTENSION IF NOT EXISTS vector;
GRANT USAGE ON SCHEMA caseflow_vectors TO caseflow_app, caseflow_embedding_indexer;
SQL
