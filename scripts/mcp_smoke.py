from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import time

import httpx

from caseflow.config import get_settings


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
    return {"discovered_tools": sorted(by_name), "metrics_result": result}


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
        print(await _exercise(endpoint))
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
