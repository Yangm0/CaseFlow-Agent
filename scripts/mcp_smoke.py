from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import time

import httpx

from caseflow.config import get_settings
from caseflow.mcp_client import _structured_tool_output, build_mcp_tool_registry
from caseflow.tools.common import ToolFailure


async def _exercise(url: str) -> dict[str, object]:
    from langchain.mcp import MCPAdapter

    async with MCPAdapter(url) as adapter:
        tools = await adapter.list_tools()
        by_name = {tool.name: tool for tool in tools}
        expected = {"query_payment_metrics", "query_change_events", "search_knowledge"}
        if set(by_name) != expected:
            raise RuntimeError("MCP tool discovery returned an unexpected allowlist.")
        schema = by_name["query_payment_metrics"].args_schema
        schema_fields = (
            schema.model_json_schema().get("properties", {})
            if hasattr(schema, "model_json_schema")
            else schema.get("properties", {})
        )
        required = {"region", "start_at", "end_at", "group_by"}
        if set(schema_fields) != required:
            raise RuntimeError("The discovered metrics schema did not match the expected contract.")
        result = await by_name["query_payment_metrics"].ainvoke(
            {
                "region": "北京",
                "start_at": "2025-01-21T09:00:00+08:00",
                "end_at": "2025-01-21T12:00:00+08:00",
                "group_by": [],
            }
        )
    if isinstance(result, dict) and result.get("error"):
        raise RuntimeError("The MCP metrics call returned a tool error.")
    invalid_result = await by_name["query_payment_metrics"].ainvoke(
        {
            "region": "",
            "start_at": "2025-01-21T09:00:00+08:00",
            "end_at": "2025-01-21T12:00:00+08:00",
            "group_by": [],
        }
    )
    invalid_output = _structured_tool_output(invalid_result)
    invalid_error = invalid_output.get("error", {}) if isinstance(invalid_output, dict) else {}
    if invalid_error.get("code") != "INVALID_TOOL_ARGUMENTS":
        raise RuntimeError("The MCP server did not return its structured invalid-parameter error.")
    return {"discovered_tools": sorted(by_name), "schema_properties": sorted(schema_fields)}


async def main() -> int:
    parser = argparse.ArgumentParser(description="Start MCP locally and check its read-only path.")
    parser.add_argument("--port", type=int, default=8761)
    args = parser.parse_args()
    settings = get_settings()
    test_url = settings.test_database_url
    if not test_url:
        raise SystemExit("TEST_DATABASE_URL is required; MCP smoke is test-database only.")
    from caseflow.db import make_engine

    test_engine = make_engine(test_url)
    if test_engine.url.database != "caseflow_test":
        test_engine.dispose()
        raise SystemExit("Refusing MCP smoke outside the isolated caseflow_test database.")
    test_engine.dispose()

    endpoint = f"http://127.0.0.1:{args.port}/mcp"
    tools = None
    environment = os.environ.copy()
    environment["DATABASE_URL"] = test_url
    environment["CASEFLOW_MCP_HOST"] = "127.0.0.1"
    environment["CASEFLOW_MCP_PORT"] = str(args.port)
    process = subprocess.Popen(
        [sys.executable, "-m", "caseflow_mcp.server"],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        health_url = f"http://127.0.0.1:{args.port}/health"
        deadline = time.monotonic() + 15
        async with httpx.AsyncClient(timeout=1) as http:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("The local MCP server exited before becoming healthy.")
                try:
                    response = await http.get(health_url)
                    if response.status_code == 200:
                        break
                except httpx.HTTPError:
                    await asyncio.sleep(0.25)
            else:
                raise RuntimeError("The local MCP server did not become healthy in time.")
        discovery = await _exercise(endpoint)
        wrapped_settings = settings.model_copy(
            update={"database_url": test_url, "mcp_server_url": endpoint}
        )
        tools = await asyncio.to_thread(build_mcp_tool_registry, wrapped_settings)
        result = await asyncio.to_thread(
            tools["query_payment_metrics"].invoke,
            {
                "region": "北京",
                "start_at": "2025-01-21T09:00:00+08:00",
                "end_at": "2025-01-21T12:00:00+08:00",
                "group_by": [],
            },
        )
        metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
        baseline = result.get("baseline", {}) if isinstance(result, dict) else {}
        if metrics.get("valid_attempts") != 50 or metrics.get("failed_attempts") != 22:
            raise RuntimeError("The MCP client did not return the structured test metrics.")
        if baseline.get("valid_attempts") != 330 or baseline.get("failed_attempts") != 10:
            raise RuntimeError("The MCP client did not return the seven-window baseline.")
        print(
            {
                **discovery,
                "client_result_type": type(result).__name__,
                "invalid_parameter_code": "INVALID_TOOL_ARGUMENTS",
                "valid_attempts": metrics["valid_attempts"],
                "failed_attempts": metrics["failed_attempts"],
                "failure_rate": metrics["failure_rate"],
                "baseline_attempts": baseline["valid_attempts"],
                "baseline_failures": baseline["failed_attempts"],
            }
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    if tools is None:
        raise RuntimeError("MCP registry was not created during the connected smoke.")
    try:
        await asyncio.to_thread(
            tools["search_knowledge"].invoke,
            {"query": "断线检查", "top_k": 1},
        )
    except ToolFailure as error:
        if error.code != "MCP_UNAVAILABLE":
            raise RuntimeError("MCP disconnection returned an unexpected error code.") from error
        print({"disconnect_error_code": error.code})
    else:
        raise RuntimeError("MCP tool call unexpectedly succeeded after the server stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
