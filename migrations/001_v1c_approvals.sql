\set ON_ERROR_STOP on

-- Run as the database administrator with these psql variables:
--   checkpoint_password, action_password, embedding_password
-- This migration is explicit and is not run by the application at startup.
SELECT format('CREATE ROLE caseflow_checkpoint LOGIN PASSWORD %L NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT', :'checkpoint_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'caseflow_checkpoint') \gexec
SELECT format('ALTER ROLE caseflow_checkpoint PASSWORD %L', :'checkpoint_password') \gexec
SELECT format('CREATE ROLE caseflow_action_executor LOGIN PASSWORD %L NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT', :'action_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'caseflow_action_executor') \gexec
SELECT format('ALTER ROLE caseflow_action_executor PASSWORD %L', :'action_password') \gexec
SELECT 'CREATE ROLE caseflow_action_owner NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT'
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'caseflow_action_owner') \gexec
SELECT 'CREATE ROLE caseflow_vector_owner NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT'
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'caseflow_vector_owner') \gexec
SELECT format('CREATE ROLE caseflow_embedding_indexer LOGIN PASSWORD %L NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT', :'embedding_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'caseflow_embedding_indexer') \gexec
SELECT format('ALTER ROLE caseflow_embedding_indexer PASSWORD %L', :'embedding_password') \gexec

GRANT CONNECT ON DATABASE caseflow TO caseflow_checkpoint, caseflow_action_executor, caseflow_embedding_indexer;
GRANT CONNECT ON DATABASE caseflow_test TO caseflow_checkpoint, caseflow_action_executor, caseflow_embedding_indexer;
CREATE SCHEMA IF NOT EXISTS caseflow_checkpoint AUTHORIZATION caseflow_checkpoint;
ALTER SCHEMA caseflow_checkpoint OWNER TO caseflow_checkpoint;
GRANT USAGE, CREATE ON SCHEMA caseflow_checkpoint TO caseflow_checkpoint;
ALTER ROLE caseflow_checkpoint IN DATABASE caseflow SET search_path = caseflow_checkpoint, public;
ALTER ROLE caseflow_checkpoint IN DATABASE caseflow_test SET search_path = caseflow_checkpoint, public;

CREATE SCHEMA IF NOT EXISTS caseflow_actions AUTHORIZATION caseflow_action_owner;
ALTER SCHEMA caseflow_actions OWNER TO caseflow_action_owner;
REVOKE ALL ON SCHEMA caseflow_actions FROM PUBLIC;
REVOKE ALL ON SCHEMA caseflow_actions FROM caseflow_app;
GRANT USAGE ON SCHEMA caseflow_actions TO caseflow_action_executor;

CREATE TABLE IF NOT EXISTS caseflow_actions.action_approvals (
    approval_id uuid PRIMARY KEY,
    thread_id text NOT NULL,
    action_type text NOT NULL CHECK (action_type = 'create_support_ticket'),
    action_payload jsonb NOT NULL CHECK (jsonb_typeof(action_payload) = 'object'),
    payload_hash varchar(64) NOT NULL,
    status text NOT NULL CHECK (status IN ('pending', 'executed', 'rejected')),
    decision_at timestamptz,
    execution_result jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_action_approvals_active_thread_payload
    ON caseflow_actions.action_approvals (thread_id, payload_hash)
    WHERE status <> 'rejected';

CREATE TABLE IF NOT EXISTS caseflow_actions.support_tickets (
    ticket_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    approval_id uuid NOT NULL UNIQUE REFERENCES caseflow_actions.action_approvals(approval_id),
    title varchar(180) NOT NULL,
    description text NOT NULL,
    priority text NOT NULL CHECK (priority IN ('low', 'medium', 'high')),
    status text NOT NULL DEFAULT 'open' CHECK (status = 'open'),
    region varchar(40) NOT NULL,
    source_investigation_id varchar(80) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS caseflow_actions.action_audit_log (
    audit_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    approval_id uuid NOT NULL REFERENCES caseflow_actions.action_approvals(approval_id),
    event_type text NOT NULL CHECK (event_type IN ('approval_created', 'approved', 'rejected', 'ticket_created')),
    payload_hash varchar(64) NOT NULL,
    event_detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (approval_id, event_type)
);

ALTER TABLE caseflow_actions.action_approvals OWNER TO caseflow_action_owner;
ALTER TABLE caseflow_actions.support_tickets OWNER TO caseflow_action_owner;
ALTER TABLE caseflow_actions.action_audit_log OWNER TO caseflow_action_owner;

CREATE OR REPLACE FUNCTION caseflow_actions.create_pending_approval(
    p_approval_id uuid,
    p_thread_id text,
    p_payload jsonb
) RETURNS jsonb
LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, caseflow_actions
AS $$
DECLARE
    existing caseflow_actions.action_approvals%ROWTYPE;
    computed_hash text;
BEGIN
    computed_hash := encode(sha256(convert_to(p_payload::text, 'UTF8')), 'hex');
    IF length(trim(p_thread_id)) < 1 OR length(computed_hash) <> 64
       OR jsonb_typeof(p_payload) <> 'object'
       OR length(trim(coalesce(p_payload->>'title', ''))) NOT BETWEEN 4 AND 180
       OR length(trim(coalesce(p_payload->>'description', ''))) NOT BETWEEN 10 AND 4000
       OR coalesce(p_payload->>'priority', '') NOT IN ('low', 'medium', 'high')
       OR length(trim(coalesce(p_payload->>'region', ''))) NOT BETWEEN 1 AND 40
       OR length(trim(coalesce(p_payload->>'source_investigation_id', ''))) NOT BETWEEN 1 AND 80 THEN
        RAISE EXCEPTION 'invalid ticket draft';
    END IF;
    INSERT INTO caseflow_actions.action_approvals
        (approval_id, thread_id, action_type, action_payload, payload_hash, status)
    VALUES (p_approval_id, p_thread_id, 'create_support_ticket', p_payload, computed_hash, 'pending')
    ON CONFLICT DO NOTHING;
    SELECT * INTO existing
      FROM caseflow_actions.action_approvals
      WHERE thread_id = p_thread_id AND payload_hash = computed_hash AND status <> 'rejected'
      ORDER BY created_at
      LIMIT 1
      FOR UPDATE;
    IF NOT FOUND THEN
        SELECT * INTO existing
          FROM caseflow_actions.action_approvals
          WHERE approval_id = p_approval_id
          FOR UPDATE;
    END IF;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'approval could not be created';
    END IF;
    IF existing.payload_hash <> computed_hash OR existing.thread_id <> p_thread_id THEN
        RAISE EXCEPTION 'approval ID does not match the original draft';
    END IF;
    INSERT INTO caseflow_actions.action_audit_log (approval_id, event_type, payload_hash)
    VALUES (existing.approval_id, 'approval_created', computed_hash)
    ON CONFLICT (approval_id, event_type) DO NOTHING;
    RETURN jsonb_build_object(
        'approval_id', existing.approval_id,
        'thread_id', existing.thread_id,
        'action_type', existing.action_type,
        'action_payload', existing.action_payload,
        'payload_hash', existing.payload_hash,
        'status', existing.status,
        'decision_at', existing.decision_at,
        'execution_result', existing.execution_result,
        'created_at', existing.created_at
    );
END;
$$;

CREATE OR REPLACE FUNCTION caseflow_actions.get_approval(p_approval_id uuid)
RETURNS jsonb
LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = pg_catalog, caseflow_actions
AS $$
    SELECT jsonb_build_object(
        'approval_id', a.approval_id,
        'thread_id', a.thread_id,
        'action_type', a.action_type,
        'action_payload', a.action_payload,
        'payload_hash', a.payload_hash,
        'status', a.status,
        'decision_at', a.decision_at,
        'execution_result', a.execution_result,
        'created_at', a.created_at
    ) FROM caseflow_actions.action_approvals AS a WHERE a.approval_id = p_approval_id
$$;

CREATE OR REPLACE FUNCTION caseflow_actions.list_thread_approvals(p_thread_id text)
RETURNS jsonb
LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = pg_catalog, caseflow_actions
AS $$
    SELECT coalesce(jsonb_agg(caseflow_actions.get_approval(a.approval_id) ORDER BY a.created_at DESC), '[]'::jsonb)
    FROM caseflow_actions.action_approvals AS a WHERE a.thread_id = p_thread_id
$$;

CREATE OR REPLACE FUNCTION caseflow_actions.approve_and_create_ticket(
    p_approval_id uuid,
    p_payload_hash text
) RETURNS jsonb
LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, caseflow_actions
AS $$
DECLARE
    approval caseflow_actions.action_approvals%ROWTYPE;
    ticket caseflow_actions.support_tickets%ROWTYPE;
BEGIN
    SELECT * INTO approval FROM caseflow_actions.action_approvals
    WHERE approval_id = p_approval_id FOR UPDATE;
    IF NOT FOUND OR approval.payload_hash <> p_payload_hash THEN
        RAISE EXCEPTION 'approval was not found or its payload hash changed';
    END IF;
    IF approval.status = 'rejected' THEN RAISE EXCEPTION 'rejected approval cannot execute'; END IF;
    IF approval.status = 'executed' THEN
        SELECT * INTO ticket FROM caseflow_actions.support_tickets WHERE approval_id = p_approval_id;
        RETURN jsonb_build_object('status', approval.status, 'ticket_id', ticket.ticket_id, 'approval_id', p_approval_id);
    END IF;
    UPDATE caseflow_actions.action_approvals
      SET status = 'executed', decision_at = coalesce(decision_at, now())
      WHERE approval_id = p_approval_id;
    INSERT INTO caseflow_actions.action_audit_log (approval_id, event_type, payload_hash)
      VALUES (p_approval_id, 'approved', p_payload_hash)
      ON CONFLICT (approval_id, event_type) DO NOTHING;
    INSERT INTO caseflow_actions.support_tickets
      (approval_id, title, description, priority, region, source_investigation_id)
    VALUES (
      p_approval_id,
      approval.action_payload->>'title',
      approval.action_payload->>'description',
      approval.action_payload->>'priority',
      approval.action_payload->>'region',
      approval.action_payload->>'source_investigation_id'
    ) ON CONFLICT (approval_id) DO NOTHING;
    SELECT * INTO ticket FROM caseflow_actions.support_tickets WHERE approval_id = p_approval_id;
    UPDATE caseflow_actions.action_approvals
      SET execution_result = jsonb_build_object('status', 'executed', 'ticket_id', ticket.ticket_id)
      WHERE approval_id = p_approval_id;
    INSERT INTO caseflow_actions.action_audit_log
      (approval_id, event_type, payload_hash, event_detail)
      VALUES (p_approval_id, 'ticket_created', p_payload_hash,
        jsonb_build_object('ticket_id', ticket.ticket_id))
      ON CONFLICT (approval_id, event_type) DO NOTHING;
    RETURN jsonb_build_object('status', 'executed', 'ticket_id', ticket.ticket_id, 'approval_id', p_approval_id);
END;
$$;

CREATE OR REPLACE FUNCTION caseflow_actions.reject_approval(
    p_approval_id uuid,
    p_payload_hash text
) RETURNS jsonb
LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, caseflow_actions
AS $$
DECLARE approval caseflow_actions.action_approvals%ROWTYPE;
BEGIN
    SELECT * INTO approval FROM caseflow_actions.action_approvals
      WHERE approval_id = p_approval_id FOR UPDATE;
    IF NOT FOUND OR approval.payload_hash <> p_payload_hash THEN
        RAISE EXCEPTION 'approval was not found or its payload hash changed';
    END IF;
    IF approval.status = 'executed' THEN RAISE EXCEPTION 'executed approval cannot be rejected'; END IF;
    UPDATE caseflow_actions.action_approvals
      SET status = 'rejected', decision_at = coalesce(decision_at, now())
      WHERE approval_id = p_approval_id;
    INSERT INTO caseflow_actions.action_audit_log (approval_id, event_type, payload_hash)
      VALUES (p_approval_id, 'rejected', p_payload_hash)
      ON CONFLICT (approval_id, event_type) DO NOTHING;
    RETURN jsonb_build_object('status', 'rejected', 'approval_id', p_approval_id);
END;
$$;

ALTER FUNCTION caseflow_actions.create_pending_approval(uuid, text, jsonb) OWNER TO caseflow_action_owner;
ALTER FUNCTION caseflow_actions.get_approval(uuid) OWNER TO caseflow_action_owner;
ALTER FUNCTION caseflow_actions.list_thread_approvals(text) OWNER TO caseflow_action_owner;
ALTER FUNCTION caseflow_actions.approve_and_create_ticket(uuid, text) OWNER TO caseflow_action_owner;
ALTER FUNCTION caseflow_actions.reject_approval(uuid, text) OWNER TO caseflow_action_owner;

REVOKE ALL ON ALL TABLES IN SCHEMA caseflow_actions FROM PUBLIC, caseflow_app, caseflow_action_executor;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA caseflow_actions FROM PUBLIC, caseflow_app, caseflow_action_executor;
REVOKE ALL ON ALL FUNCTIONS IN SCHEMA caseflow_actions FROM PUBLIC;
GRANT EXECUTE ON FUNCTION caseflow_actions.create_pending_approval(uuid, text, jsonb) TO caseflow_action_executor;
GRANT EXECUTE ON FUNCTION caseflow_actions.get_approval(uuid) TO caseflow_action_executor;
GRANT EXECUTE ON FUNCTION caseflow_actions.list_thread_approvals(text) TO caseflow_action_executor;
GRANT EXECUTE ON FUNCTION caseflow_actions.approve_and_create_ticket(uuid, text) TO caseflow_action_executor;
GRANT EXECUTE ON FUNCTION caseflow_actions.reject_approval(uuid, text) TO caseflow_action_executor;
