from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse

from caseflow.config import get_settings
from caseflow.db import check_database, make_engine
from caseflow.schemas import GroupDimension
from caseflow.tools.common import ToolFailure
from caseflow.tools.registry import build_tool_registry

settings = get_settings()
engine = make_engine(settings.database_url)
tools = build_tool_registry(engine, settings.sql_statement_timeout_ms, settings)
mcp = FastMCP(
    "CaseFlow read-only investigation tools",
    host=settings.mcp_host,
    port=settings.mcp_port,
)


def _invoke(name: str, args: dict[str, Any]) -> Any:
    try:
        return tools[name].invoke(args)
    except ToolFailure as error:
        return {"error": {"code": error.code, "message": error.public_message}}
    except ValidationError:
        return {
            "error": {
                "code": "INVALID_TOOL_ARGUMENTS",
                "message": "Tool arguments failed validation.",
            }
        }
    except Exception:
        return {"error": {"code": "TOOL_EXECUTION_FAILED", "message": "The read-only tool failed."}}


@mcp.tool()
def query_payment_metrics(
    region: str,
    start_at: datetime,
    end_at: datetime,
    group_by: list[GroupDimension] | None = None,
) -> dict[str, Any]:
    """Read bounded payment counts and rates for a region and timezone-aware [start, end) window."""
    return _invoke(
        "query_payment_metrics",
        {"region": region, "start_at": start_at, "end_at": end_at, "group_by": group_by or []},
    )


@mcp.tool()
def query_change_events(
    start_at: datetime,
    end_at: datetime,
    region: str | None = None,
) -> dict[str, Any]:
    """Read service changes and incidents in a timezone-aware [start, end) window."""
    return _invoke(
        "query_change_events",
        {"start_at": start_at, "end_at": end_at, "region": region},
    )


@mcp.tool()
def search_knowledge(query: str, top_k: int = 5) -> dict[str, Any]:
    """Search approved operational documents; returns exact source text and stable evidence IDs."""
    return _invoke("search_knowledge", {"query": query, "top_k": top_k})


@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    available = await asyncio.to_thread(check_database, engine)
    return JSONResponse(
        {
            "status": "ok" if available else "degraded",
            "database": "available" if available else "unavailable",
        },
        status_code=200 if available else 503,
    )


def main() -> None:
    try:
        mcp.run(transport="streamable-http")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
