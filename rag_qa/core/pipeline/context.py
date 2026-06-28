# -*- coding: utf-8 -*-
"""
管道上下文 — Stage 间共享的数据容器

PipelineContext 是智能管道中所有 Stage 的"中转站"。
每个 Stage 从 ctx 读取所需数据，处理后将结果写回 ctx。
这消除了 Stage 间的链式传参，使 Stage 接口统一为 execute(ctx) -> StageResult。

设计原则:
  - 所有中间状态集中于一处，便于调试和日志
  - 字段初始化为安全的默认值（空字符串/空列表/False）
  - 使用 dataclass + field(default_factory) 避免可变默认值陷阱
"""

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StageResult:
    """
    单个 Stage 的执行结果

    Attributes:
        status: 执行状态
            "success"  — 正常完成，管道继续
            "skip"     — Stage 判定应短路（如领域守卫拦截），管道应终止并返回已有答案
            "degraded" — 降级完成（如 KG 不可用），管道继续但质量可能降低
            "failed"   — 意外错误，管道终止并返回错误应答
        data: Stage 产生的数据（用于调试和监控）
        timing_ms: Stage 执行耗时（毫秒）
        error: 错误信息（仅 status="failed" 时有值）
    """

    status: str = "success"       # "success" | "skip" | "degraded" | "failed"
    data: dict = field(default_factory=dict)
    timing_ms: float = 0.0
    error: str = ""


@dataclass
class PipelineContext:
    """
    管道上下文 — 一次完整请求的所有状态

    Stage 通过 ctx 读取上游结果、写入自身产出。
    管道结束后，SmartPipeline 从 ctx 构建 RAGResponse。

    字段分组:
      Input — 请求入口参数（Stage 只读）
      Classification — QueryAnalyzer + ClassifyStage 产出
      Retrieval — RetrievalStage 产出
      Generation — GenerateStage 产出
      Compliance — ComplianceStage 产出
      Agent — Agent 通道专用字段
      Timing — 各阶段性能记录
    """

    # ── Input: 请求入口 ──
    query: str = ""
    session_id: str = ""
    user_role: str = "agent"

    # ── Classification: 分类阶段产出 ──
    complexity: int = 1           # 0=FAQ/闲聊, 1=简单咨询, 2=复杂业务
    intent: str = ""
    entities: dict = field(default_factory=dict)
    classifier_source: str = ""   # "rule" / "bert" / "llm" / "fallback"

    # ── Short-circuit: 快速路径 ──
    faq_answer: Optional[str] = None       # FAQ 命中时直接返回
    domain_guard_passed: bool = True
    cache_hit: bool = False

    # ── KG: 知识图谱推理 ──
    kg_context: str = ""          # KG 推理出的上下文文本（注入 LLM prompt）

    # ── Retrieval: 检索阶段产出 ──
    strategy_plan: Optional[object] = None  # StrategyPlan
    retrieved_chunks: list = field(default_factory=list)
    quality_passed: bool = True
    quality_fallback: str = ""

    # ── Generation: 生成阶段产出 ──
    generated_answer: str = ""
    sources: list = field(default_factory=list)

    # ── Compliance: 合规阶段产出 ──
    compliance_passed: bool = True
    compliance_violations: list = field(default_factory=list)
    compliance_modified: str = ""

    # ── Agent: Agent 通道专用 ──
    route_mode: str = "rag"       # "rag" | "multi_agent"
    agent_route: Optional[dict] = None
    agent_results: dict = field(default_factory=dict)

    # ── Timing: 性能监控 ──
    pipeline: dict = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
