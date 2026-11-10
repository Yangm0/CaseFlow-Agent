(() => {
  const $ = (id) => document.getElementById(id);
  const state = { threadId: null, eventSource: null, report: null, approvalId: null };
  const activity = (label, detail = "") => {
    const item = document.createElement("li");
    item.textContent = detail ? `${label} · ${detail}` : label;
    $("activity-list").prepend(item);
  };
  const message = (role, content) => {
    const item = document.createElement("div");
    item.className = `message ${role}`;
    item.textContent = content;
    $("messages").append(item);
    item.scrollIntoView({ block: "nearest" });
  };
  const request = async (url, options = {}) => {
    const response = await fetch(url, {
      ...options,
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.detail || `HTTP ${response.status}`);
    return body;
  };
  const showReport = (report) => {
    state.report = report;
    $("report").textContent = JSON.stringify(report, null, 2);
    $("thread-status").textContent = `${report.status} · ${report.elapsed_ms} ms`;
    const answer = report.summary || "调查完成。";
    message("assistant", answer);
    if (report.token_usage) {
      activity("报告完成", `Token ${report.token_usage.total_tokens ?? "不可用"} · ${report.elapsed_ms} ms`);
    }
    if (report.statistical_facts?.length) {
      const fact = report.statistical_facts[0];
      $("ticket-title").value = `支付异常调查：${fact.query_parameters.region}`;
      $("ticket-description").value = `${report.summary}\n\n证据 ID：${fact.evidence_id}`;
      $("approval-panel").classList.remove("hidden");
    }
  };
  const connectEvents = (threadId) => {
    if (state.eventSource) state.eventSource.close();
    const source = new EventSource(`/api/v1/threads/${threadId}/events`);
    state.eventSource = source;
    source.onopen = () => { $("connection-state").textContent = "实时连接"; };
    source.onerror = () => { $("connection-state").textContent = "重连中"; };
    source.addEventListener("status", (event) => {
      const data = JSON.parse(event.data);
      $("thread-status").textContent = data.status || "处理中";
      activity("状态", data.status || "处理中");
    });
    source.addEventListener("agent_step", (event) => activity("Agent 阶段", JSON.parse(event.data).stage));
    source.addEventListener("tool_start", (event) => {
      const data = JSON.parse(event.data);
      activity("调用工具", `${data.tool_name} ${JSON.stringify(data.arguments)}`);
    });
    source.addEventListener("tool_result", (event) => {
      const data = JSON.parse(event.data);
      activity("工具返回", `${data.tool_name} · ${data.status} · ${(data.evidence_ids || []).join(", ")}`);
    });
    source.addEventListener("evidence", (event) => activity("证据", JSON.parse(event.data).evidence_ids.join(", ") || "无"));
    source.addEventListener("approval_required", (event) => renderApproval(JSON.parse(event.data)));
    source.addEventListener("completed", () => refreshThread());
    source.addEventListener("error", (event) => {
      if (event.data) activity("请求错误", JSON.parse(event.data).code || "错误");
    });
  };
  const refreshThread = async () => {
    if (!state.threadId) return;
    try {
      const thread = await request(`/api/v1/threads/${state.threadId}`);
      if (thread.report) {
        state.report = thread.report;
        $("report").textContent = JSON.stringify(thread.report, null, 2);
      }
      $("thread-status").textContent = thread.status;
      $("resume-thread").classList.toggle("hidden", thread.status !== "interrupted");
      const approvals = thread.pending_approvals || [];
      if (approvals.length) {
        renderApproval({
          ...approvals[0],
          draft: approvals[0].action_payload,
          evidence_ids: (thread.report?.statistical_facts || []).map((item) => item.evidence_id),
        });
      }
    } catch (error) { activity("恢复状态失败", error.message); }
  };
  const renderApproval = (approval) => {
    state.approvalId = approval.approval_id;
    const draft = approval.draft || approval.action_payload || {};
    $("approval-details").textContent = `审批 ${approval.approval_id}\n标题：${draft.title}\n优先级：${draft.priority}\n地区：${draft.region}\n调查：${draft.source_investigation_id}\n描述：${draft.description}\n证据：${(approval.evidence_ids || []).join(", ")}`;
    $("approval-form").classList.add("hidden");
    $("pending-approval").classList.remove("hidden");
    const isPending = approval.status === "pending";
    $("approve-ticket").classList.toggle("hidden", !isPending);
    $("reject-ticket").classList.toggle("hidden", !isPending);
    $("approval-result").textContent = isPending
      ? "当前状态：等待人工审批"
      : `当前状态：${approval.status} · ${JSON.stringify(approval.execution_result || {})}`;
  };
  const newThread = async () => {
    const thread = await request("/api/v1/threads", { method: "POST", body: "{}" });
    state.threadId = thread.thread_id;
    localStorage.setItem("caseflow-thread-id", thread.thread_id);
    $("thread-id").textContent = thread.thread_id;
    $("messages").replaceChildren();
    $("activity-list").replaceChildren();
    $("report").textContent = "等待调查结果";
    $("thread-status").textContent = "会话已创建";
    $("approval-panel").classList.add("hidden");
    $("pending-approval").classList.add("hidden");
    $("approval-form").classList.remove("hidden");
    $("resume-thread").classList.add("hidden");
    connectEvents(thread.thread_id);
  };
  $("new-thread").addEventListener("click", () => newThread().catch((error) => activity("创建会话失败", error.message)));
  $("resume-thread").addEventListener("click", async () => {
    if (!state.threadId) return;
    $("resume-thread").disabled = true;
    try {
      const report = await request(`/api/v1/threads/${state.threadId}/resume`, { method: "POST" });
      showReport(report);
      await refreshThread();
    } catch (error) {
      activity("恢复调查失败", error.message);
      await refreshThread();
    } finally { $("resume-thread").disabled = false; }
  });
  $("message-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!state.threadId) await newThread();
    const question = $("question").value.trim();
    if (!question) return;
    message("user", question);
    $("send-message").disabled = true;
    $("resume-thread").classList.add("hidden");
    $("thread-status").textContent = "调查中";
    try {
      const report = await request(`/api/v1/threads/${state.threadId}/messages`, {
        method: "POST", body: JSON.stringify({ question }),
      });
      showReport(report);
      $("question").value = "";
    } catch (error) {
      activity("调查失败", error.message);
      $("thread-status").textContent = "失败";
      await refreshThread();
    } finally { $("send-message").disabled = false; }
  });
  $("approval-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!state.report || !state.threadId) return;
    try {
      const approval = await request(`/api/v1/threads/${state.threadId}/approvals`, {
        method: "POST",
        body: JSON.stringify({ draft: {
          title: $("ticket-title").value,
          description: $("ticket-description").value,
          priority: $("ticket-priority").value,
          region: $("ticket-region").value,
          source_investigation_id: state.report.investigation_id,
        } }),
      });
      renderApproval({ ...approval, draft: approval.action_payload, evidence_ids: state.report.statistical_facts.map((item) => item.evidence_id) });
      activity("待人工审批", approval.approval_id);
    } catch (error) { $("approval-result").textContent = error.message; }
  });
  const decide = async (decision) => {
    if (!state.approvalId || !state.threadId) return;
    try {
      const result = await request(`/api/v1/threads/${state.threadId}/approvals/${state.approvalId}/decision`, {
        method: "POST", body: JSON.stringify({ decision }),
      });
      $("approval-result").textContent = JSON.stringify(result, null, 2);
      activity("审批结果", JSON.stringify(result));
    } catch (error) { $("approval-result").textContent = error.message; }
  };
  $("approve-ticket").addEventListener("click", () => decide("approve"));
  $("reject-ticket").addEventListener("click", () => decide("reject"));

  const savedThreadId = localStorage.getItem("caseflow-thread-id");
  if (savedThreadId) {
    state.threadId = savedThreadId;
    $("thread-id").textContent = savedThreadId;
    connectEvents(savedThreadId);
    request(`/api/v1/threads/${savedThreadId}`).then((thread) => {
      $("messages").replaceChildren();
      for (const item of thread.messages || []) message(item.role, item.content);
      if (thread.report) {
        state.report = thread.report;
        $("report").textContent = JSON.stringify(thread.report, null, 2);
        if (thread.report.statistical_facts?.length) $("approval-panel").classList.remove("hidden");
      }
      $("thread-status").textContent = thread.status;
      $("resume-thread").classList.toggle("hidden", thread.status !== "interrupted");
      const pending = thread.pending_approvals || [];
      if (pending.length) {
        renderApproval({
          ...pending[0],
          draft: pending[0].action_payload,
          evidence_ids: (thread.report?.statistical_facts || []).map((item) => item.evidence_id),
        });
      }
    }).catch((error) => {
      activity("无法恢复上次会话", error.message);
      localStorage.removeItem("caseflow-thread-id");
    });
  }
})();
