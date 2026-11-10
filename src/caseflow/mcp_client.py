from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from caseflow.config import Settings
from caseflow.schemas import ChangeEventsInput, KnowledgeSearchInput, PaymentMetricsInput
from caseflow.tools.common import ToolFailure

MCP_TOOL_SCHEMAS = {
    "query_payment_metrics": PaymentMetricsInput,
    "query_change_events": ChangeEventsInput,
    "search_knowledge": KnowledgeSearchInput,
}


def build_mcp_tool_registry(settings: Settings) -> dict[str, BaseTool]:
    """Discover the independent Streamable HTTP server and expose only reviewed tools.

    This synchronous bridge is called from FastAPI's worker thread. The first-party LangChain
    adapter reopens a Streamable HTTP client for each tool call, so calls are not pinned to a
    stale session.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError("MCP discovery must run outside an active event loop.")
    discovered = asyncio.run(_discover(settings))

    by_name = {tool.name: tool for tool in discovered}
    if set(by_name) != set(MCP_TOOL_SCHEMAS):
        raise RuntimeError("The MCP server tool allowlist does not match the CaseFlow contract.")

    result: dict[str, BaseTool] = {}
    for name, schema in MCP_TOOL_SCHEMAS.items():
        remote = by_name[name]
        remote_schema = remote.args_schema
        remote_properties = (
            remote_schema.model_json_schema().get("properties", {})
            if hasattr(remote_schema, "model_json_schema")
            else remote_schema.get("properties", {})
            if isinstance(remote_schema, dict)
            else {}
        )
        expected_properties = set(schema.model_json_schema().get("properties", {}))
        if set(remote_properties) != expected_properties:
            raise RuntimeError(f"MCP schema mismatch for the approved tool {name}.")

        async def invoke_remote(
            _remote: BaseTool = remote, _schema: type = schema, **arguments: Any
        ) -> Any:
            try:
                wire_arguments = _schema.model_validate(arguments).model_dump(mode="json")
            except Exception as error:
                raise ToolFailure(
                    "INVALID_TOOL_ARGUMENTS",
                    "Tool arguments failed local contract validation.",
                ) from error
            last_error: Exception | None = None
            for _attempt in range(2):
                try:
                    result = await asyncio.wait_for(
                        _remote.ainvoke(wire_arguments), timeout=settings.mcp_tool_timeout_seconds
                    )
                    output = _structured_tool_output(result)
                    if not isinstance(output, dict):
                        raise ToolFailure(
                            "MCP_INVALID_RESULT",
                            "The MCP tool did not return a structured object.",
                        )
                    if isinstance(output, dict) and output.get("error"):
                        detail = output["error"]
                        raise ToolFailure(
                            str(detail.get("code", "MCP_TOOL_FAILED")),
                            str(detail.get("message", "The MCP tool returned an error.")),
                        )
                    return output
                except ToolFailure:
                    raise
                except TimeoutError as error:
                    last_error = error
                    break
                except Exception as error:
                    last_error = error
            if isinstance(last_error, (TimeoutError, asyncio.TimeoutError)):
                raise ToolFailure("MCP_TOOL_TIMEOUT", "The MCP tool exceeded its time limit.")
            raise ToolFailure(
                "MCP_UNAVAILABLE",
                "The read-only MCP service could not complete the request.",
            )

        def invoke_remote_sync(_call=invoke_remote, **arguments: Any) -> Any:
            return asyncio.run(_call(**arguments))

        result[name] = StructuredTool.from_function(
            func=invoke_remote_sync,
            coroutine=invoke_remote,
            name=name,
            description=remote.description or "Read-only CaseFlow MCP tool.",
            args_schema=schema,
        )
    return result


async def _discover(settings: Settings) -> list[BaseTool]:
    from langchain.mcp import MCPAdapter

    adapter = MCPAdapter(settings.mcp_server_url)
    async with adapter:
        return await asyncio.wait_for(adapter.list_tools(), timeout=10)


def _structured_tool_output(value: Any) -> Any:
    """Normalize MCP artifacts and JSON text blocks into the tool's structured dictionary."""
    if isinstance(value, tuple) and len(value) == 2:
        content, artifact = value
        return _structured_tool_output(artifact if artifact is not None else content)
    structured = getattr(value, "structuredContent", None) or getattr(
        value, "structured_content", None
    )
    if structured is not None:
        return _structured_tool_output(structured)
    if isinstance(value, dict):
        structured = value.get("structuredContent", value.get("structured_content"))
        if structured is not None:
            return _structured_tool_output(structured)
        if value.get("isError"):
            return {"error": {"code": "MCP_TOOL_FAILED", "message": "MCP reported a tool error."}}
        if isinstance(value.get("content"), list):
            return _structured_tool_output(value["content"])
        return value
    if isinstance(value, list):
        text_blocks = [
            block["text"]
            for block in value
            if isinstance(block, dict)
            and block.get("type") in ("text", "output_text")
            and isinstance(block.get("text"), str)
        ]
        if text_blocks:
            try:
                parsed = json.loads("\n".join(text_blocks))
            except json.JSONDecodeError:
                return None
            return _structured_tool_output(parsed)
        return None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return _structured_tool_output(parsed)
    if hasattr(value, "model_dump"):
        return _structured_tool_output(value.model_dump(mode="json"))
    return None
