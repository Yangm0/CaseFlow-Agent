from collections.abc import Iterator
from datetime import datetime, time, timedelta
from random import Random
from uuid import NAMESPACE_URL, uuid5
from zoneinfo import ZoneInfo

TZ_NAME = "Asia/Shanghai"
TZ = ZoneInfo(TZ_NAME)
DATASET_SEED = 20250121
DATASET_START = datetime(2025, 1, 1, tzinfo=TZ)
DATASET_AS_OF = datetime(2025, 2, 1, tzinfo=TZ)
REGIONS = ("北京", "上海", "深圳", "杭州")
ATTEMPTS_PER_REGION_PER_DAY = 400
SCENARIO_START = datetime(2025, 1, 21, 9, 0, tzinfo=TZ)
SCENARIO_END = datetime(2025, 1, 21, 12, 0, tzinfo=TZ)


DOCUMENTS: tuple[dict[str, str], ...] = (
    {
        "document_id": "DOC-PAY-001",
        "title": "支付指标口径与失败率计算",
        "source_type": "operations_manual",
        "content": (
            "支付尝试以每次进入支付网关的请求为单位。只有终态为 succeeded 或 failed 的记录"
            "纳入有效支付尝试；pending、cancelled 等非终态记录不进入失败率分母。失败率定义为"
            "失败数除以有效支付尝试数。查询区间统一使用左闭右开 [start, end)，按 Asia/Shanghai"
            "展示业务时间。比较不同时间窗口时，应同时报告样本数和窗口长度，避免小样本误读。"
        ),
    },
    {
        "document_id": "DOC-PAY-002",
        "title": "支付失败率告警与波动判读",
        "source_type": "operations_manual",
        "content": (
            "支付失败率升高是排查信号，不等同于故障定论。优先核对同地区、相近时段的历史基线、"
            "有效样本量、失败码构成和告警覆盖范围。短窗口样本较小时，应给出不确定性，不宜仅凭"
            "单个百分比触发大范围操作。异常门槛需结合服务等级目标和历史分布设定。"
        ),
    },
    {
        "document_id": "DOC-ROUTE-001",
        "title": "支付路由优先级配置说明",
        "source_type": "service_runbook",
        "content": (
            "payment-router 根据地区、渠道可用性和路由权重选择上游通道。优先级或权重变更可能"
            "改变流量分布。灰度发布应按地区观察成功率、上游超时率和重试次数；如果多个指标同时"
            "偏离基线，值班人员应检查变更记录并评估回滚。时间接近只能建立调查关联，不能单独证明"
            "变更导致了故障。"
        ),
    },
    {
        "document_id": "DOC-GW-001",
        "title": "网关超时与上游失败排查手册",
        "source_type": "service_runbook",
        "content": (
            "网关超时通常表现为请求等待上游响应超过配置阈值，支付侧可能记录 timeout、upstream"
            "unavailable 或 unknown。排查时按地区和上游渠道拆分请求，检查 p95/p99 延迟、连接池、"
            "上游健康状态以及超时重试。不要把用户主动取消与网关超时混为同一种失败。"
        ),
    },
    {
        "document_id": "DOC-RETRY-001",
        "title": "支付重试与幂等控制",
        "source_type": "engineering_standard",
        "content": (
            "支付重试必须使用稳定幂等键，且重试次数应有上限和退避间隔。增加重试可能提高最终成功"
            "率，也可能扩大上游拥塞和请求放大。分析失败率时应同时检查重试量、重复请求比例与上游"
            "超时，不能仅根据重试配置变化推断净效果。"
        ),
    },
    {
        "document_id": "DOC-INC-001",
        "title": "支付异常调查流程",
        "source_type": "incident_playbook",
        "content": (
            "调查流程：确认指标口径与查询窗口；与匹配时段基线比较；按地区、服务和状态拆分；检索"
            "告警、事故和发布记录；形成多个可验证假设；由值班负责人决定是否回滚或切流。调查报告"
            "应区分观测事实、可能原因和已验证根因，并记录尚缺少的证据。"
        ),
    },
    {
        "document_id": "DOC-CHANGE-001",
        "title": "变更审查、灰度与回滚规范",
        "source_type": "engineering_standard",
        "content": (
            "生产变更应记录服务名、发布时间、配置差异、影响范围和回滚步骤。灰度应使用可观测的"
            "地区或流量标签，并设置停止条件。事故时间与变更时间重叠只是相关线索；需要对照发布前后"
            "指标、受影响与未受影响地区、日志和配置差异，才能加强因果判断。"
        ),
    },
    {
        "document_id": "DOC-BJ-001",
        "title": "北京地区支付渠道运行说明",
        "source_type": "service_runbook",
        "content": (
            "北京地区 payment-router 可选择银联北京、网联和备用通道。工作日早间存在银行侧流量"
            "变化，需查看地区标签、通道优先级和各上游健康探针。此说明不代表任何具体时段必然发生"
            "渠道故障。切换通道前应确认容量和幂等策略。"
        ),
    },
    {
        "document_id": "DOC-ALERT-001",
        "title": "支付服务监控与告警字段",
        "source_type": "operations_manual",
        "content": (
            "支付监控按 region、service、channel 和 response_class 聚合。告警包括成功率下降、上游"
            "超时升高、重试量突增和连接池耗尽。先确认告警窗口、去重规则及延迟数据，再与交易聚合"
            "交叉核对。监控聚合与数据库终态记录的口径可能不同。"
        ),
    },
    {
        "document_id": "DOC-STATUS-001",
        "title": "支付状态与响应分类",
        "source_type": "data_dictionary",
        "content": (
            "succeeded 表示支付成功终态，failed 表示明确失败终态；pending 表示仍在处理，不能"
            "直接记作失败；cancelled 表示取消，需要按具体分析口径单独处理。响应分类 timeout、declined"
            "和 unavailable 用于定位原因，但不能仅凭状态标签确认上游责任。"
        ),
    },
    {
        "document_id": "DOC-ROLLBACK-001",
        "title": "payment-router 回滚检查清单",
        "source_type": "service_runbook",
        "content": (
            "回滚前确认变更版本、受影响地区和当前上游健康状况；比较发布前后的成功率及延迟；确认"
            "回滚不会将流量导入已降级通道。执行后持续观察至少一个完整业务窗口，并记录回滚时间和"
            "验证结果。回滚是有权限的值班操作，调查 Agent 不会自动执行。"
        ),
    },
    {
        "document_id": "DOC-TIMEOUT-001",
        "title": "数据库查询与时间范围约定",
        "source_type": "engineering_standard",
        "content": (
            "服务端时间戳以带时区时间写入数据库。API 查询必须传入带 UTC offset 的时间；数据库保存"
            "timestamptz，展示时转换到 Asia/Shanghai。查询区间左闭右开，避免相邻窗口边界重复计数。"
            "生产调查查询设置语句超时，并只授予只读数据库账号。"
        ),
    },
    {
        "document_id": "DOC-CHANNEL-001",
        "title": "上游渠道健康检查",
        "source_type": "service_runbook",
        "content": (
            "渠道健康检查包含连接成功率、响应延迟、错误码分布和探针状态。单个渠道错误率变化时，"
            "应核对该渠道样本数和相邻地区对照。通道切换会改变后续样本构成，报告中应标注切换时间。"
        ),
    },
    {
        "document_id": "DOC-CAPACITY-001",
        "title": "支付链路容量与连接池",
        "source_type": "service_runbook",
        "content": (
            "高并发时检查连接池等待、线程池队列、上游并发限制和重试放大。容量问题可能同时影响多地，"
            "也可能因路由流量集中只影响单一地区。排查需要将请求量、成功率和延迟放在同一时间轴上。"
        ),
    },
    {
        "document_id": "DOC-SETTLEMENT-001",
        "title": "上海结算任务运行说明",
        "source_type": "service_runbook",
        "content": (
            "上海日间结算任务由 settlement-worker 执行，与支付网关请求链路分开部署。调查支付失败时，"
            "结算任务变更可作为时间线背景，但只有发现共享依赖或资源争用证据后才应纳入可能原因。"
        ),
    },
    {
        "document_id": "DOC-CACHE-001",
        "title": "路由配置缓存刷新机制",
        "source_type": "engineering_standard",
        "content": (
            "payment-router 将路由配置缓存到实例内存，配置发布后通过版本号和刷新事件同步。实例版本"
            "不一致可能导致同一地区请求走不同通道。排查时对比实例配置版本、刷新延迟与错误请求样本。"
        ),
    },
    {
        "document_id": "DOC-REGION-001",
        "title": "多地区支付指标对照方法",
        "source_type": "operations_manual",
        "content": (
            "多地区对照有助于区分局部路由问题与全局依赖问题。比较时应匹配业务时段、通道和有效样本"
            "定义。地区间流量结构不同，原始失败率不能直接解释为服务质量差异。"
        ),
    },
    {
        "document_id": "DOC-REPORT-001",
        "title": "事故调查报告字段模板",
        "source_type": "incident_playbook",
        "content": (
            "调查报告记录：问题范围、观察窗口、指标公式、匹配基线、引用证据、变更与事故时间线、"
            "候选假设、已排除项、缺失数据、处理建议及操作状态。未经验证的解释必须标为假设；不得把"
            "调查建议写成已经执行的操作。"
        ),
    },
)


def _stable_id(prefix: str, key: str) -> str:
    return f"{prefix}-{uuid5(NAMESPACE_URL, f'caseflow-v1a:{key}').hex[:24]}"


def generate_attempts() -> Iterator[dict[str, object]]:
    """Yield 31 * 4 * 400 reproducible attempts without reading hidden labels."""
    day_count = (DATASET_AS_OF.date() - DATASET_START.date()).days
    base_failure_rate = {"北京": 0.025, "上海": 0.019, "深圳": 0.022, "杭州": 0.018}
    for day_offset in range(day_count):
        day = DATASET_START + timedelta(days=day_offset)
        day_key = day.date().isoformat()
        for region_index, region in enumerate(REGIONS):
            rng = Random(DATASET_SEED + day_offset * 101 + region_index * 1009)
            for ordinal in range(ATTEMPTS_PER_REGION_PER_DAY):
                attempted_at = day + timedelta(seconds=rng.randrange(24 * 60 * 60))
                if rng.random() < 0.02:
                    status = "pending"
                else:
                    probability = base_failure_rate[region]
                    if (
                        region == "北京"
                        and attempted_at.date() == SCENARIO_START.date()
                        and time(9, 0) <= attempted_at.timetz().replace(tzinfo=None) < time(12, 0)
                    ):
                        probability = 0.40
                    status = "failed" if rng.random() < probability else "succeeded"
                yield {
                    "attempt_id": _stable_id("PAY", f"{day_key}:{region}:{ordinal}"),
                    "region": region,
                    "attempted_at": attempted_at,
                    "service_name": "payment-router",
                    "status": status,
                }


def generate_changes() -> tuple[dict[str, object], ...]:
    return (
        {
            "event_id": "CHG-20250121-BJ-ROUTE-01",
            "occurred_at": datetime(2025, 1, 21, 8, 52, tzinfo=TZ),
            "region": "北京",
            "service_name": "payment-router",
            "summary": "发布 payment-router 2.8.4；调整北京银联通道优先级并提高超时重试上限。",
        },
        {
            "event_id": "CHG-20250121-SH-SETTLE-01",
            "occurred_at": datetime(2025, 1, 21, 9, 30, tzinfo=TZ),
            "region": "上海",
            "service_name": "settlement-worker",
            "summary": "上海结算看板字段升级；不涉及支付网关路由配置。",
        },
        {
            "event_id": "CHG-20250120-GLOBAL-MON-01",
            "occurred_at": datetime(2025, 1, 20, 16, 0, tzinfo=TZ),
            "region": None,
            "service_name": "monitoring-agent",
            "summary": "调整监控采样间隔；不修改支付请求处理逻辑。",
        },
        {
            "event_id": "CHG-20250118-SZ-GW-01",
            "occurred_at": datetime(2025, 1, 18, 14, 15, tzinfo=TZ),
            "region": "深圳",
            "service_name": "gateway-proxy",
            "summary": "深圳网关代理证书轮换，变更窗口内健康探针正常。",
        },
        {
            "event_id": "CHG-20250112-BJ-DASH-01",
            "occurred_at": datetime(2025, 1, 12, 11, 5, tzinfo=TZ),
            "region": "北京",
            "service_name": "payment-dashboard",
            "summary": "北京支付看板增加分渠道筛选项，不修改支付服务。",
        },
        {
            "event_id": "CHG-20250108-HZ-CACHE-01",
            "occurred_at": datetime(2025, 1, 8, 10, 20, tzinfo=TZ),
            "region": "杭州",
            "service_name": "payment-router",
            "summary": "杭州路由配置缓存刷新逻辑小版本升级。",
        },
        {
            "event_id": "CHG-20250105-GLOBAL-DOC-01",
            "occurred_at": datetime(2025, 1, 5, 12, 0, tzinfo=TZ),
            "region": None,
            "service_name": "docs-service",
            "summary": "内部操作手册索引重建，不影响支付请求处理。",
        },
    )


def generate_incidents() -> tuple[dict[str, object], ...]:
    return (
        {
            "incident_id": "INC-20250121-BJ-01",
            "occurred_at": datetime(2025, 1, 21, 9, 7, tzinfo=TZ),
            "region": "北京",
            "service_name": "payment-router",
            "summary": "北京支付成功率告警触发；部分请求报告上游响应超时，根因调查进行中。",
            "status": "investigating",
        },
        {
            "incident_id": "INC-20250110-BJ-01",
            "occurred_at": datetime(2025, 1, 10, 10, 12, tzinfo=TZ),
            "region": "北京",
            "service_name": "payment-router",
            "summary": "北京渠道探针短时抖动；恢复后成功率回到当日基线。",
            "status": "resolved",
        },
        {
            "incident_id": "INC-20250107-SZ-01",
            "occurred_at": datetime(2025, 1, 7, 15, 40, tzinfo=TZ),
            "region": "深圳",
            "service_name": "gateway-proxy",
            "summary": "深圳代理节点连接池耗尽，扩容后延迟下降。",
            "status": "resolved",
        },
        {
            "incident_id": "INC-20250103-HZ-01",
            "occurred_at": datetime(2025, 1, 3, 8, 25, tzinfo=TZ),
            "region": "杭州",
            "service_name": "payment-router",
            "summary": "杭州少量上游拒绝响应；未发现跨地区影响。",
            "status": "resolved",
        },
    )


def dataset_metadata() -> dict[str, str]:
    return {
        "dataset_id": "caseflow-v1a-synthetic-2025-01",
        "generator_version": "1",
        "random_seed": str(DATASET_SEED),
        "time_zone": TZ_NAME,
        "dataset_start": DATASET_START.isoformat(),
        "dataset_end_exclusive": DATASET_AS_OF.isoformat(),
        "as_of": DATASET_AS_OF.isoformat(),
        "attempts_per_region_per_day": str(ATTEMPTS_PER_REGION_PER_DAY),
        "anomaly_region": "北京",
        "anomaly_start": SCENARIO_START.isoformat(),
        "anomaly_end_exclusive": SCENARIO_END.isoformat(),
        "ground_truth_file": "evals/ground_truth.json",
    }
