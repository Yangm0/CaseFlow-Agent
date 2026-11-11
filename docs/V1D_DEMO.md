# V1-D browser demo guide

This walkthrough uses the local synthetic databases. Keep the API and MCP bound to `127.0.0.1`; do not place secrets in a screen recording or browser-visible report.

## Start and open

1. Start PostgreSQL and the API/MCP services with `docker compose up -d postgres mcp api`.
2. Confirm `http://127.0.0.1:8000/health` and `http://127.0.0.1:8001/health` are healthy (or use the loopback ports configured in the local `.env`).
3. Open the API root in a browser. Choose offline mode for a repeatable demo. Live mode is optional and consumes model requests.

## Investigation and human action

Use this question in a new thread:

> 请调查北京地区 2025-01-21 09:00 到 12:00 支付失败率是否异常，查询相关变更和知识文档。如果需要处理，请帮我创建一个高优先级工单。

Show these moments:

1. The streamed activity list as the Agent calls metrics, change events, then knowledge search through the separate MCP service.
2. The report's 22/50 Beijing failures (44%) and 10/330 baseline (3.03%), its supporting citations, and its uncertainty statement.
3. The automatically proposed ticket's title, description, high priority, Beijing region, investigation ID, and evidence list. Edit the draft once to demonstrate that the pending payload is revised and the old hash is superseded.
4. Reject the revised draft in one run; in a separate clean thread, approve it. Approval is a human UI action. The result panel shows the executed status and one synthetic ticket ID.
5. Refresh the page or restart only the API process, reopen the same thread, and show restored conversation/report/action state. The current trace shows actual tool calls, evidence IDs, token counts, model request count, and trace ID for the Live example.

The already completed Live demonstration used the real free route and created one synthetic ticket. Its report, trace, and exact evidence are in `docs/V1D_ACCEPTANCE_2026-11-11.md`. Do not repeat Live calls merely to recreate a screenshot.

## Expected safety behavior

- While an approval is pending, a new message on that same thread is rejected until the action is decided.
- Rejecting creates no ticket. Repeating the same decision is idempotent.
- Editing changes the payload hash; an approval for the superseded hash is rejected.
- Only an explicit human approval reaches the function-authorized executor.
- The case study is entirely synthetic; it does not modify payment attempts.

`scripts/v1d_smoke.py` reproduces the isolated API/MCP/database approval flow without a browser and cleans only its uniquely tagged test records.
