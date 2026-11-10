from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

from sqlalchemy import Integer, and_, cast, extract, func, or_, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from caseflow.db import readonly_transaction
from caseflow.models import PaymentAttempt
from caseflow.schemas import GroupDimension, PaymentMetricsInput
from caseflow.tools.common import ToolFailure, translate_database_error

DISPLAY_TZ = ZoneInfo("Asia/Shanghai")
STATEMENT_RESULT_LIMIT = 50


def calculate_failure_rate(valid_attempts: int, failed_attempts: int) -> float | None:
    if valid_attempts < 0 or failed_attempts < 0 or failed_attempts > valid_attempts:
        raise ValueError("counts must satisfy 0 <= failed_attempts <= valid_attempts")
    if valid_attempts == 0:
        return None
    return failed_attempts / valid_attempts


def build_target_filter(region: str, start_at: datetime, end_at: datetime):
    return and_(
        PaymentAttempt.region == region,
        PaymentAttempt.attempted_at >= start_at,
        PaymentAttempt.attempted_at < end_at,
        PaymentAttempt.status.in_(("succeeded", "failed")),
    )


def _windows_for_baseline(start_at: datetime, end_at: datetime) -> tuple[str, list[tuple[datetime, datetime]]]:
    duration = end_at - start_at
    if duration <= timedelta(days=1):
        windows = [
            (start_at - timedelta(days=offset), end_at - timedelta(days=offset))
            for offset in range(1, 8)
        ]
        return "previous_seven_matching_daily_windows", windows
    return "immediately_preceding_equal_duration_window", [(start_at - duration, start_at)]


class MetricsService:
    def __init__(self, engine: Engine, statement_timeout_ms: int = 3000):
        self.engine = engine
        self.statement_timeout_ms = statement_timeout_ms

    def query(self, params: PaymentMetricsInput) -> dict[str, Any]:
        valid_status = PaymentAttempt.status.in_(("succeeded", "failed"))
        failed_status = PaymentAttempt.status == "failed"
        target_filter = build_target_filter(params.region, params.start_at, params.end_at)
        target_stmt = select(
            func.count().label("valid_attempts"),
            func.count().filter(failed_status).label("failed_attempts"),
        ).where(target_filter)

        baseline_method, baseline_windows = _windows_for_baseline(params.start_at, params.end_at)
        baseline_filter = and_(
            PaymentAttempt.region == params.region,
            valid_status,
            or_(
                *[
                    and_(PaymentAttempt.attempted_at >= left, PaymentAttempt.attempted_at < right)
                    for left, right in baseline_windows
                ]
            ),
        )
        baseline_stmt = select(
            func.count().label("valid_attempts"),
            func.count().filter(failed_status).label("failed_attempts"),
        ).where(baseline_filter)

        group_exprs: list[tuple[str, Any]] = []
        for dimension in params.group_by:
            if dimension is GroupDimension.SERVICE_NAME:
                group_exprs.append((dimension.value, PaymentAttempt.service_name))
            elif dimension is GroupDimension.STATUS:
                group_exprs.append((dimension.value, PaymentAttempt.status))
            elif dimension is GroupDimension.HOUR_OF_DAY:
                local_timestamp = func.timezone("Asia/Shanghai", PaymentAttempt.attempted_at)
                group_exprs.append(
                    (dimension.value, cast(extract("hour", local_timestamp), Integer))
                )
        grouped_rows: list[dict[str, object]] = []
        grouped_stmt = None
        if group_exprs:
            group_columns = [expr.label(label) for label, expr in group_exprs]
            grouped_stmt = (
                select(
                    *group_columns,
                    func.count().label("valid_attempts"),
                    func.count().filter(failed_status).label("failed_attempts"),
                )
                .where(target_filter)
                .group_by(*[expr for _, expr in group_exprs])
                .order_by(func.count().desc())
                .limit(STATEMENT_RESULT_LIMIT)
            )

        try:
            with readonly_transaction(self.engine, self.statement_timeout_ms) as connection:
                target = connection.execute(target_stmt).mappings().one()
                baseline = connection.execute(baseline_stmt).mappings().one()
                if grouped_stmt is not None:
                    for row in connection.execute(grouped_stmt).mappings():
                        item = {label: row[label] for label, _ in group_exprs}
                        valid = int(row["valid_attempts"])
                        failed = int(row["failed_attempts"])
                        item.update(
                            {
                                "valid_attempts": valid,
                                "failed_attempts": failed,
                                "failure_rate": calculate_failure_rate(valid, failed),
                            }
                        )
                        grouped_rows.append(item)
        except SQLAlchemyError as error:
            raise translate_database_error(error) from error

        target_valid = int(target["valid_attempts"])
        target_failed = int(target["failed_attempts"])
        baseline_valid = int(baseline["valid_attempts"])
        baseline_failed = int(baseline["failed_attempts"])
        evidence_id = f"MET-{uuid4().hex[:16]}"
        return {
            "evidence_id": evidence_id,
            "query_parameters": {
                "region": params.region,
                "start_at": params.start_at.astimezone(DISPLAY_TZ).isoformat(),
                "end_at": params.end_at.astimezone(DISPLAY_TZ).isoformat(),
                "time_zone": "Asia/Shanghai",
                "interval_semantics": "[start_at, end_at)",
                "group_by": [item.value for item in params.group_by],
            },
            "metrics": {
                "valid_attempts": target_valid,
                "failed_attempts": target_failed,
                "failure_rate": calculate_failure_rate(target_valid, target_failed),
            },
            "baseline": {
                "method": baseline_method,
                "windows": [
                    {
                        "start_at": left.astimezone(DISPLAY_TZ).isoformat(),
                        "end_at": right.astimezone(DISPLAY_TZ).isoformat(),
                    }
                    for left, right in baseline_windows
                ],
                "valid_attempts": baseline_valid,
                "failed_attempts": baseline_failed,
                "failure_rate": calculate_failure_rate(baseline_valid, baseline_failed),
            },
            "groups": grouped_rows,
            "calculation": (
                "failure_rate = failed_attempts / valid_attempts; valid attempts have status "
                "succeeded or failed; pending and other non-terminal statuses are excluded."
            ),
            "execution": {
                "read_only": True,
                "statement_timeout_ms": self.statement_timeout_ms,
                "result_limit": STATEMENT_RESULT_LIMIT,
            },
        }
