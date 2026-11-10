from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from caseflow.db import readonly_transaction
from caseflow.models import ChangeEvent, Incident
from caseflow.schemas import ChangeEventsInput
from caseflow.tools.common import translate_database_error

DISPLAY_TZ = ZoneInfo("Asia/Shanghai")
RESULT_LIMIT = 50


class ChangeEventsService:
    def __init__(self, engine: Engine, statement_timeout_ms: int = 3000):
        self.engine = engine
        self.statement_timeout_ms = statement_timeout_ms

    def query(self, params: ChangeEventsInput) -> dict[str, Any]:
        change_stmt = select(
            ChangeEvent.event_id,
            ChangeEvent.occurred_at,
            ChangeEvent.region,
            ChangeEvent.service_name,
            ChangeEvent.summary,
        ).where(
            ChangeEvent.occurred_at >= params.start_at,
            ChangeEvent.occurred_at < params.end_at,
        )
        incident_stmt = select(
            Incident.incident_id,
            Incident.occurred_at,
            Incident.region,
            Incident.service_name,
            Incident.summary,
            Incident.status,
        ).where(
            Incident.occurred_at >= params.start_at,
            Incident.occurred_at < params.end_at,
        )
        if params.region:
            change_stmt = change_stmt.where(
                (ChangeEvent.region == params.region) | ChangeEvent.region.is_(None)
            )
            incident_stmt = incident_stmt.where(Incident.region == params.region)
        change_stmt = change_stmt.order_by(ChangeEvent.occurred_at).limit(RESULT_LIMIT)
        incident_stmt = incident_stmt.order_by(Incident.occurred_at).limit(RESULT_LIMIT)
        try:
            with readonly_transaction(self.engine, self.statement_timeout_ms) as connection:
                changes = connection.execute(change_stmt).mappings().all()
                incidents = connection.execute(incident_stmt).mappings().all()
        except SQLAlchemyError as error:
            raise translate_database_error(error) from error

        events: list[dict[str, object]] = []
        for item in changes:
            events.append(
                {
                    "evidence_id": f"CHANGE-{item['event_id']}",
                    "event_type": "change",
                    "event_id": item["event_id"],
                    "occurred_at": item["occurred_at"].astimezone(DISPLAY_TZ).isoformat(),
                    "region": item["region"],
                    "service_name": item["service_name"],
                    "summary": item["summary"],
                    "status": None,
                }
            )
        for item in incidents:
            events.append(
                {
                    "evidence_id": f"INCIDENT-{item['incident_id']}",
                    "event_type": "incident",
                    "event_id": item["incident_id"],
                    "occurred_at": item["occurred_at"].astimezone(DISPLAY_TZ).isoformat(),
                    "region": item["region"],
                    "service_name": item["service_name"],
                    "summary": item["summary"],
                    "status": item["status"],
                }
            )
        events.sort(key=lambda event: str(event["occurred_at"]))
        events = events[:RESULT_LIMIT]
        return {
            "query_id": f"EVT-{uuid4().hex[:16]}",
            "query_parameters": {
                "region": params.region,
                "start_at": params.start_at.astimezone(DISPLAY_TZ).isoformat(),
                "end_at": params.end_at.astimezone(DISPLAY_TZ).isoformat(),
                "time_zone": "Asia/Shanghai",
                "interval_semantics": "[start_at, end_at)",
            },
            "result_limit": RESULT_LIMIT,
            "events": events,
            "causality_note": (
                "Temporal overlap is an investigation lead; it does not establish that a change caused an incident."
            ),
        }
