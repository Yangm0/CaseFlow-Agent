#!/bin/sh
set -eu

psql -v ON_ERROR_STOP=1 \
  --username "$POSTGRES_USER" \
  --dbname "$POSTGRES_DB" \
  -v app_password="$CASEFLOW_APP_PASSWORD" \
  -v seed_password="$CASEFLOW_SEED_PASSWORD" <<'SQL'
CREATE ROLE caseflow_app LOGIN PASSWORD :'app_password'
  NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;
CREATE ROLE caseflow_seed LOGIN PASSWORD :'seed_password'
  NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;
GRANT CONNECT ON DATABASE caseflow TO caseflow_app, caseflow_seed;
CREATE DATABASE caseflow_test OWNER caseflow_admin;
GRANT CONNECT ON DATABASE caseflow_test TO caseflow_app, caseflow_seed;
GRANT USAGE ON SCHEMA public TO caseflow_app;
GRANT USAGE, CREATE ON SCHEMA public TO caseflow_seed;
SQL

psql -v ON_ERROR_STOP=1 \
  --username "$POSTGRES_USER" \
  --dbname caseflow_test <<'SQL'
GRANT USAGE ON SCHEMA public TO caseflow_app;
GRANT USAGE, CREATE ON SCHEMA public TO caseflow_seed;
SQL
