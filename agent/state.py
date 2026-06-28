# -*- coding: utf-8 -*-
"""
Agent 状态定义 — LangGraph 状态图中流转的数据结构

LangGraph 的核心概念:
  整个 Agent 的运行是一个状态图 (StateGraph)，
  每个节点 (node) 接收当前状态、处理后返回更新后的状态。
  状态 (state) 就是在节点之间流转的数据包。

  类比: 状态 = 流水线上的工件，每个工位 (节点) 加工后传递给下一个工位。

状态字段说明:
  messages: 对话历史 (LangGraph 标准字段)
  plan: 当前执行计划 (工具列表 + 参数)
  tool_results: 工具执行结果收集
  iteration: 当前循环次数 (防止无限循环)
  final_answer: 最终答案
  ...
"""

from dataclasses import dataclass, field
from typing import Annotated, Optional

from langgraph.graph.message import add_messages

# ============================================================
# 工具调用计划
# ============================================================


@dataclass
class ToolCall:
    """
    单个工具调用计划

    由 Planner 生成，由 Executor 执行。

    Attributes:
        tool_name: 工具名称 (如 "policy_query")
        tool_args: 工具参数 (如 {"insurer": "平安", "product": "e生保"})
        depends_on: 依赖的前置工具调用索引 (用于串行/并行决策)
        status: 执行状态 "pending" / "running" / "done" / "failed"
        result: 工具返回结果
    """

    tool_name: str
    tool_args: dict = field(default_factory=dict)
    depends_on: list[int] = field(default_factory=list)
    status: str = "pending"  # pending / running / done / failed / skipped
    result: Optional[str] = None
    error_type: str = ""  # "" / "transient" / "permanent" — 用于区分可恢复/不可恢复错误


@dataclass
class Plan:
    """
    完整执行计划 — Planner 的输出

    Attributes:
        tool_calls: 工具调用列表 (按执行顺序排列)
        reasoning: Planner 的推理过程 (用于日志和调试)
        is_complete: 是否已完成规划
    """

    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning: str = ""
    is_complete: bool = False


# ============================================================
# Agent 状态 — 在 LangGraph 节点间流转
# ============================================================


@dataclass
class AgentState:
    """
    Agent 状态 — LangGraph 的核心数据结构

    每个 LangGraph 节点接收这个 state，修改后返回。
    LangGraph 负责在节点间传递和合并 state。

    Attributes:
        messages: 对话消息列表 (LangGraph 标准字段)
                  使用 Annotated + add_messages 实现自动追加
        session_id: 会话 ID (用于 Checkpoint 恢复)
        user_query: 用户原始 query (脱敏后)
        intent: 意图分类结果
        entities: 提取的实体 (保司/产品名等)

        plan: 当前执行计划 (Planner 生成)
        current_tool_index: 当前正在执行的工具索引
        tool_results: 所有工具的执行结果

        iteration: 当前 Reflect 循环次数 (防无限循环)
        max_iterations: 最大循环次数

        final_answer: 最终答案 (Synthesizer 生成)
        sources: 引用的条款来源
        error: 错误信息

        pipeline: 各阶段耗时记录 (用于监控)
    """

    # ── 对话消息 (LangGraph 标准字段) ──
    # Annotated + add_messages: 每次节点返回 messages 时自动追加，不覆盖
    messages: Annotated[list, add_messages] = field(default_factory=list)

    # ── 会话信息 ──
    session_id: str = ""
    user_query: str = ""
    intent: str = ""
    entities: dict = field(default_factory=dict)

    # ── 规划与执行 ──
    plan: Optional[Plan] = None
    current_tool_index: int = 0
    tool_results: list[dict] = field(default_factory=list)

    # ── 循环控制 ──
    iteration: int = 0
    max_iterations: int = 5  # 最多 Reflect 5 次

    # ── 输出 ──
    final_answer: str = ""
    sources: list[dict] = field(default_factory=list)
    error: str = ""

    # ── 角色 ──
    user_role: str = "agent"

    # ── 监控 ──
    pipeline: dict = field(default_factory=dict)


# ============================================================
# 多 Agent 架构状态 — Orchestrator + 子 Agent
# ============================================================
# 以下类型用于多 Agent 协作模式。


@dataclass
class SubAgentTask:
    """
    Orchestrator 分发给子 Agent 的单个任务

    当一个用户请求需要多个 Agent 协作时，
    Orchestrator 将复杂意图拆解为多个 SubAgentTask，
    每个 Task 交给对应的领域子 Agent 执行。

    属性:
        task_id: 任务唯一标识 (如 "task_0", "task_1")
        agent_name: 目标子 Agent 名称 ("insurance"/"underwriting"/"claim"/"service")
        intent: 子意图 (如 "产品推荐"、"核保审核")
        user_query: 用户原始 query (上游 Agent 的结果会注入到 context 中)
        entities: 提取的实体 (保司/产品名/疾病等)
        context: Orchestrator 传递的上下文 (用户画像、上游 Agent 结果等)
        dependencies: 依赖的 task_id 列表 (上游任务完成后才能执行)
        priority: 优先级 (越大越先执行，目前未使用)
    """

    task_id: str = ""
    agent_name: str = ""
    intent: str = ""
    user_query: str = ""
    entities: dict = field(default_factory=dict)
    context: dict = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    priority: int = 0
    user_role: str = "agent"  # 角色: customer / agent / underwriter / admin


@dataclass
class OrchestratorState:
    """
    Orchestrator 的状态 — 在意图路由和任务分发间流转

    与 AgentState 的本质区别:
      AgentState 关注的是"单个工具调用的细节"
      OrchestratorState 关注的是"哪些子 Agent 需要被调用、什么顺序、结果怎么聚合"

    流程:
      user_query → 路由 → 构建任务列表 → 按序执行 → 聚合结果 → 合规终审 → final_answer

    属性:
        session_id: 会话 ID
        user_query: 用户原始问题
        user_profile: 长期记忆中的用户画像 (已购产品/理赔记录/偏好)

        primary_intent: 主意图 (投保/核保/理赔/客服)
        complexity: 复杂度 "simple" | "moderate" | "complex"
         route_mode: 路由模式
            "multi_agent"   → 多子 Agent 协作
            "rag_fallback"   → 降级为纯 RAG

        tasks: 拆解后的任务列表
        results: 执行结果 {task_id: SubAgentResult}
        errors: 错误收集

        final_answer: 最终答案
        sources: 所有子 Agent 的条款引用
        pipeline: 各阶段耗时监控
    """

    # ── 用户输入 ──
    session_id: str = ""
    user_query: str = ""
    user_profile: dict = field(default_factory=dict)

    # ── 意图与路由 ──
    primary_intent: str = ""
    complexity: str = "moderate"
    route_mode: str = "multi_agent"

    # ── 任务规划 ──
    tasks: list[SubAgentTask] = field(default_factory=list)
    execution_order: list[str] = field(default_factory=list)

    # ── 执行结果 ──
    results: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    # ── 最终输出 ──
    final_answer: str = ""
    sources: list[dict] = field(default_factory=list)

    # ── 监控 ──
    pipeline: dict = field(default_factory=dict)
