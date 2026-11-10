from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy.engine import Engine

from caseflow.agent import InvestigationAgent, build_agent
from caseflow.config import Settings, get_settings
from caseflow.db import check_database, make_engine
from caseflow.schemas import InvestigationReport, InvestigationRequest
from caseflow.tools.registry import build_tool_registry


def create_app(
    settings: Settings | None = None,
    engine: Engine | None = None,
    agent: InvestigationAgent | None = None,
) -> FastAPI:
    app_settings = settings or get_settings()
    app_engine = engine or make_engine(app_settings.database_url)
    tools = build_tool_registry(app_engine, app_settings.sql_statement_timeout_ms)
    app_agent = agent if agent is not None else build_agent(app_settings, tools)

    application = FastAPI(
        title="CaseFlow Payment Investigation Agent",
        version="0.1.0",
        description="Read-only payment incident investigation API (V1-A).",
    )
    application.state.settings = app_settings
    application.state.engine = app_engine
    application.state.agent = app_agent

    @application.get("/health")
    def health():
        database_ok = check_database(app_engine)
        live_blocked = app_settings.mode == "live" and app_agent is None
        response = {
            "status": "ok" if database_ok and not live_blocked else "degraded",
            "database": "available" if database_ok else "unavailable",
            "mode": app_settings.mode,
            "model": (
                "offline_test_stub"
                if app_settings.mode == "offline_test"
                else "configured"
                if app_agent is not None
                else "credentials_missing"
            ),
        }
        return JSONResponse(
            status_code=200 if database_ok and not live_blocked else 503, content=response
        )

    @application.post("/api/v1/investigations", response_model=InvestigationReport)
    def investigate(request: InvestigationRequest):
        if app_agent is None:
            return InvestigationReport(
                investigation_id=str(uuid4()),
                trace_id=str(uuid4()),
                mode=app_settings.mode,
                status="blocked",
                summary="live 模式尚未配置模型凭据，本次调查未启动。",
                statistical_facts=[],
                document_evidence=[],
                system_events=[],
                hypotheses=[],
                uncertainties=[
                    "请通过 OPENAI_API_KEY、OPENAI_MODEL 和可选 OPENAI_BASE_URL 配置 live 模式。"
                ],
                recommendations=[],
                tool_calls=[],
                operations_executed=["未执行数据库或业务操作。"],
                error={
                    "code": "LLM_CREDENTIALS_MISSING",
                    "message": "Live model credentials are not configured.",
                },
            )
        return app_agent.investigate(request.question)

    return application


app = create_app()
