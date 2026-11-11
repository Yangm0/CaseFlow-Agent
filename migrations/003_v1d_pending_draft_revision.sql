\set ON_ERROR_STOP on

BEGIN;

ALTER TABLE caseflow_actions.action_approvals
    DROP CONSTRAINT IF EXISTS action_approvals_status_check;
ALTER TABLE caseflow_actions.action_approvals
    ADD CONSTRAINT action_approvals_status_check
    CHECK (status IN ('pending', 'executed', 'rejected', 'superseded'));

ALTER TABLE caseflow_actions.action_audit_log
    DROP CONSTRAINT IF EXISTS action_audit_log_event_type_check;
ALTER TABLE caseflow_actions.action_audit_log
    ADD CONSTRAINT action_audit_log_event_type_check
    CHECK (event_type IN ('approval_created', 'approved', 'rejected', 'ticket_created', 'draft_revised'));

DROP INDEX IF EXISTS caseflow_actions.uq_action_approvals_active_thread_payload;
CREATE UNIQUE INDEX uq_action_approvals_active_thread_payload
    ON caseflow_actions.action_approvals (thread_id, payload_hash)
    WHERE status IN ('pending', 'executed');
CREATE UNIQUE INDEX IF NOT EXISTS uq_action_approvals_pending_thread
    ON caseflow_actions.action_approvals (thread_id)
    WHERE status = 'pending';

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
    IF length(trim(p_thread_id)) < 1
       OR jsonb_typeof(p_payload) IS DISTINCT FROM 'object'
       OR length(trim(coalesce(p_payload->>'title', ''))) NOT BETWEEN 4 AND 180
       OR length(trim(coalesce(p_payload->>'description', ''))) NOT BETWEEN 10 AND 4000
       OR coalesce(p_payload->>'priority', '') NOT IN ('low', 'medium', 'high')
       OR length(trim(coalesce(p_payload->>'region', ''))) NOT BETWEEN 1 AND 40
       OR length(trim(coalesce(p_payload->>'source_investigation_id', ''))) NOT BETWEEN 1 AND 80
       OR length(trim(coalesce(p_payload->>'rationale', ''))) NOT BETWEEN 8 AND 1200
       OR jsonb_typeof(p_payload->'supporting_evidence_ids') IS DISTINCT FROM 'array'
       OR coalesce(jsonb_array_length(p_payload->'supporting_evidence_ids'), 0) NOT BETWEEN 1 AND 8
       OR EXISTS (
           SELECT 1 FROM jsonb_array_elements_text(p_payload->'supporting_evidence_ids') AS ids(id)
           WHERE length(trim(ids.id)) NOT BETWEEN 1 AND 128
       ) THEN
        RAISE EXCEPTION 'invalid ticket draft';
    END IF;
    INSERT INTO caseflow_actions.action_approvals
        (approval_id, thread_id, action_type, action_payload, payload_hash, status)
    VALUES (p_approval_id, p_thread_id, 'create_support_ticket', p_payload, computed_hash, 'pending')
    ON CONFLICT DO NOTHING;
    SELECT * INTO existing
      FROM caseflow_actions.action_approvals
      WHERE thread_id = p_thread_id AND payload_hash = computed_hash
        AND status IN ('pending', 'executed')
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
        RAISE EXCEPTION 'approval could not be created; this thread may already have a pending action';
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

CREATE OR REPLACE FUNCTION caseflow_actions.revise_pending_approval(
    p_approval_id uuid,
    p_expected_payload_hash text,
    p_new_approval_id uuid,
    p_payload jsonb
) RETURNS jsonb
LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, caseflow_actions
AS $$
DECLARE
    previous caseflow_actions.action_approvals%ROWTYPE;
    replacement caseflow_actions.action_approvals%ROWTYPE;
    computed_hash text;
BEGIN
    SELECT * INTO previous FROM caseflow_actions.action_approvals
      WHERE approval_id = p_approval_id FOR UPDATE;
    IF NOT FOUND OR previous.status <> 'pending'
       OR previous.payload_hash <> p_expected_payload_hash THEN
        RAISE EXCEPTION 'pending approval was not found or its payload hash changed';
    END IF;
    computed_hash := encode(sha256(convert_to(p_payload::text, 'UTF8')), 'hex');
    IF jsonb_typeof(p_payload) IS DISTINCT FROM 'object'
       OR length(trim(coalesce(p_payload->>'title', ''))) NOT BETWEEN 4 AND 180
       OR length(trim(coalesce(p_payload->>'description', ''))) NOT BETWEEN 10 AND 4000
       OR coalesce(p_payload->>'priority', '') NOT IN ('low', 'medium', 'high')
       OR length(trim(coalesce(p_payload->>'region', ''))) NOT BETWEEN 1 AND 40
       OR length(trim(coalesce(p_payload->>'source_investigation_id', ''))) NOT BETWEEN 1 AND 80
       OR length(trim(coalesce(p_payload->>'rationale', ''))) NOT BETWEEN 8 AND 1200
       OR jsonb_typeof(p_payload->'supporting_evidence_ids') IS DISTINCT FROM 'array'
       OR coalesce(jsonb_array_length(p_payload->'supporting_evidence_ids'), 0) NOT BETWEEN 1 AND 8
       OR EXISTS (
           SELECT 1 FROM jsonb_array_elements_text(p_payload->'supporting_evidence_ids') AS ids(id)
           WHERE length(trim(ids.id)) NOT BETWEEN 1 AND 128
       ) THEN
        RAISE EXCEPTION 'invalid ticket draft';
    END IF;

    UPDATE caseflow_actions.action_approvals
      SET status = 'superseded', decision_at = now()
      WHERE approval_id = p_approval_id;
    INSERT INTO caseflow_actions.action_audit_log
        (approval_id, event_type, payload_hash, event_detail)
    VALUES (
        p_approval_id,
        'draft_revised',
        p_expected_payload_hash,
        jsonb_build_object(
            'replacement_approval_id', p_new_approval_id,
            'replacement_payload_hash', computed_hash
        )
    ) ON CONFLICT (approval_id, event_type) DO NOTHING;

    INSERT INTO caseflow_actions.action_approvals
        (approval_id, thread_id, action_type, action_payload, payload_hash, status)
    VALUES (
        p_new_approval_id,
        previous.thread_id,
        previous.action_type,
        p_payload,
        computed_hash,
        'pending'
    ) ON CONFLICT DO NOTHING;
    SELECT * INTO replacement FROM caseflow_actions.action_approvals
      WHERE thread_id = previous.thread_id AND payload_hash = computed_hash
        AND status IN ('pending', 'executed')
      ORDER BY created_at
      LIMIT 1
      FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'replacement approval could not be created';
    END IF;
    INSERT INTO caseflow_actions.action_audit_log
        (approval_id, event_type, payload_hash)
    VALUES (replacement.approval_id, 'approval_created', replacement.payload_hash)
    ON CONFLICT (approval_id, event_type) DO NOTHING;
    RETURN jsonb_build_object(
        'approval_id', replacement.approval_id,
        'thread_id', replacement.thread_id,
        'action_type', replacement.action_type,
        'action_payload', replacement.action_payload,
        'payload_hash', replacement.payload_hash,
        'status', replacement.status,
        'decision_at', replacement.decision_at,
        'execution_result', replacement.execution_result,
        'created_at', replacement.created_at
    );
END;
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
    IF approval.status = 'executed' THEN
        SELECT * INTO ticket FROM caseflow_actions.support_tickets
          WHERE approval_id = p_approval_id;
        IF NOT FOUND THEN
            RAISE EXCEPTION 'executed approval has no support ticket';
        END IF;
        RETURN jsonb_build_object(
            'status', 'executed', 'ticket_id', ticket.ticket_id, 'approval_id', p_approval_id
        );
    END IF;
    IF approval.status <> 'pending' THEN
        RAISE EXCEPTION 'only a pending approval can execute';
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
    RETURN jsonb_build_object(
        'status', 'executed', 'ticket_id', ticket.ticket_id, 'approval_id', p_approval_id
    );
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
    IF approval.status = 'rejected' THEN
        RETURN jsonb_build_object('status', 'rejected', 'approval_id', p_approval_id);
    END IF;
    IF approval.status <> 'pending' THEN
        RAISE EXCEPTION 'only a pending approval can be rejected';
    END IF;
    UPDATE caseflow_actions.action_approvals
      SET status = 'rejected', decision_at = coalesce(decision_at, now())
      WHERE approval_id = p_approval_id;
    INSERT INTO caseflow_actions.action_audit_log (approval_id, event_type, payload_hash)
      VALUES (p_approval_id, 'rejected', p_payload_hash)
      ON CONFLICT (approval_id, event_type) DO NOTHING;
    RETURN jsonb_build_object('status', 'rejected', 'approval_id', p_approval_id);
END;
$$;

ALTER FUNCTION caseflow_actions.create_pending_approval(uuid, text, jsonb)
    OWNER TO caseflow_action_owner;
ALTER FUNCTION caseflow_actions.revise_pending_approval(uuid, text, uuid, jsonb)
    OWNER TO caseflow_action_owner;
ALTER FUNCTION caseflow_actions.approve_and_create_ticket(uuid, text)
    OWNER TO caseflow_action_owner;
ALTER FUNCTION caseflow_actions.reject_approval(uuid, text)
    OWNER TO caseflow_action_owner;
REVOKE ALL ON FUNCTION caseflow_actions.revise_pending_approval(uuid, text, uuid, jsonb) FROM PUBLIC;
REVOKE ALL ON FUNCTION caseflow_actions.approve_and_create_ticket(uuid, text) FROM PUBLIC;
REVOKE ALL ON FUNCTION caseflow_actions.reject_approval(uuid, text) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION caseflow_actions.revise_pending_approval(uuid, text, uuid, jsonb)
    TO caseflow_action_executor;
GRANT EXECUTE ON FUNCTION caseflow_actions.approve_and_create_ticket(uuid, text)
    TO caseflow_action_executor;
GRANT EXECUTE ON FUNCTION caseflow_actions.reject_approval(uuid, text)
    TO caseflow_action_executor;

COMMIT;
