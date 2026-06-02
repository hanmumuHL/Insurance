# -*- coding: utf-8 -*-
"""
子 Agent 基类 — 所有领域 Agent 的统一接口

设计原则:
  1. 每个子 Agent 是一个独立的 LangGraph 状态图
  2. 统一接口: sub_agent.invoke(task) → SubAgentResult
  3. 子 Agent 之间不直接通信，通过 Orchestrator 协调
  4. 每个子 Agent 有独立的工具集、Prompt 模板、合规约束

子 Agent 与单 Agent 的区别:
  单 Agent: 一个 Planner 管所有 7 个工具，Prompt 里塞了所有规则
  子 Agent: 只暴露本领域 3-5 个工具，Prompt 是领域专属的，规则不冲突
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from langgraph.graph import StateGraph, START, END


@dataclass
class SubAgentResult:
    """
    子 Agent 的执行结果 — 统一返回格式

    无论哪个子 Agent，都返回这个结构。
    Orchestrator 不需要知道子 Agent 内部实现细节，
    只需要读 status、result、confidence 三个字段做决策。
    """
    agent_name: str                              # Agent 名称
    task_id: str                                 # 任务 ID (Orchestrator 分配)
    status: str                                  # "success" | "degraded" | "failed"
    result: str                                  # 执行结果 (自然语言)
    structured_data: dict = field(default_factory=dict)  # 结构化数据 (供前端渲染)
    sources: list = field(default_factory=list)          # 引用的条款来源
    confidence: float = 0.0                      # 置信度 0-1
    disclaimer: str = ""                         # 合规声明
    suggested_next: str = ""                     # 建议的下一步 (供 Orchestrator 决策)
    latency_ms: float = 0.0                      # 执行耗时
    error: str = ""                              # 错误信息


class SubAgent(ABC):
    """
    子 Agent 基类 — 所有领域 Agent 的抽象父类

    每个子 Agent 必须实现:
      - _get_system_prompt(): 领域专属 System Prompt（定义"人设"）
      - _get_tools(): 领域专属工具集（3-5 个工具）
      - _get_compliance_rules(): 领域合规约束

    标准流程 (可覆盖):
      plan → exec → check → synthesize
      与单 Agent 相同，但 Prompt 和工具集是领域专属的。

    使用方法:
      agent = ClaimAgent(vector_store=vs, mysql_session=db)
      result = agent.invoke({
          "task_id": "task_1",
          "user_query": "肺炎住院能赔吗",
          "entities": {"product": "尊享e生"},
          "context": {"user_profile": {...}},
      })
    """

    def __init__(self, name: str, model: str = "deepseek-v3"):
        """
        Args:
            name: Agent 名称 (用于日志和路由，如 "claim", "insurance")
            model: 使用的模型
                   "deepseek-v3"  → 轻量任务 (投保推荐/客服FAQ)
                   "deepseek-r1"  → 需要推理的任务 (理赔判断/核保评估)
        """
        self.name = name
        self.model = model
        self.graph = self._build_graph()

    # ================================================================
    # 抽象方法 — 子类必须实现
    # ================================================================

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """
        领域专属 System Prompt

        定义该 Agent 的"人设":
          - 职责边界 (能做什么、不能做什么)
          - 专业知识 (该领域特有的术语和规则)
          - 回答风格 (正式/亲切/简洁)

        Orchestrator 在构建 Planner Prompt 时会把这个 prompt 作为前缀。
        """
        ...

    @abstractmethod
    def _get_tools(self) -> list:
        """
        领域专属工具集 (LangChain BaseTool 列表)

        每个 Agent 只暴露本领域需要的工具:
          投保 Agent: product_search, premium_calc, product_compare, health_questionnaire
          核保 Agent: underwriting_rules, risk_assessment
          理赔 Agent: policy_query, clause_search, claim_eligibility, premium_calc
          客服 Agent: policy_query, faq_search, manual_handoff
        """
        ...

    @abstractmethod
    def _get_compliance_rules(self) -> dict:
        """
        领域合规约束

        Returns:
            dict: {
                "forbidden_phrases": [str, ...],
                "required_disclaimers": [str, ...],
                "force_handoff_triggers": [str, ...],
            }
        """
        ...

    # ================================================================
    # 标准 Plan-Execute 状态图 (子类可覆盖)
    # ================================================================

    def _build_graph(self) -> StateGraph:
        """
        构建子 Agent 的 LangGraph 状态图

        标准架构: plan → exec → check → synthesize
        与单 Agent 图的区别:
          1. Planner 的 Prompt 加了 _get_system_prompt()
          2. Synthesizer 加了 _get_compliance_rules()
          3. 状态更聚焦——不需要意图路由 (Orchestrator 已处理)
        """
        graph = StateGraph(dict)

        # ── 节点 ──
        graph.add_node("plan", self._plan_node)            # 规划: 选工具 + 定顺序
        graph.add_node("exec", self._exec_node)            # 执行: 按计划调工具
        graph.add_node("check", self._check_node)          # 检查: 结果是否充分
        graph.add_node("synthesize", self._synthesize_node) # 合成: 整合结果生成答案

        # ── 边 ──
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "exec")
        graph.add_edge("exec", "check")
        graph.add_conditional_edges(
            "check",
            self._should_continue,
            {"plan": "plan", "synthesize": "synthesize"},
        )
        graph.add_edge("synthesize", END)

        return graph.compile()

    # ================================================================
    # 主入口: invoke(task) → SubAgentResult
    # ================================================================

    def invoke(self, task: dict) -> SubAgentResult:
        """
        执行子 Agent 任务

        Args:
            task: {
                "task_id": str,        # Orchestrator 分配的任务 ID
                "intent": str,         # 子意图
                "user_query": str,     # 用户原始问题
                "context": dict,       # 上游 Agent 的上下文 (Orchestrator 注入)
                "entities": dict,      # 提取的实体
                "session_id": str,     # 会话 ID
            }

        Returns:
            SubAgentResult: 统一格式的执行结果
        """
        import time
        t0 = time.time()

        # ── 构造初始状态 ──
        initial_state = {
            "user_query": task.get("user_query", ""),
            "intent": task.get("intent", ""),
            "entities": task.get("entities", {}),
            "context": task.get("context", {}),
            "session_id": task.get("session_id", ""),
            "iteration": 0,
            "max_iterations": 3,       # 子 Agent 最多迭代 3 次 (比单 Agent 少)
            "messages": [],
            "plan": None,
            "current_tool_index": 0,
            "tool_results": [],
            "final_answer": "",
        }

        # ── 执行状态图 ──
        try:
            result = self.graph.invoke(initial_state)
            latency = (time.time() - t0) * 1000

            return SubAgentResult(
                agent_name=self.name,
                task_id=task.get("task_id", ""),
                status="success",
                result=result.get("final_answer", ""),
                sources=result.get("sources", []),
                confidence=self._estimate_confidence(result),
                disclaimer=self._get_compliance_rules().get(
                    "required_disclaimers", [""])[0] if self._get_compliance_rules().get(
                    "required_disclaimers") else "",
                latency_ms=latency,
            )

        except Exception as e:
            latency = (time.time() - t0) * 1000
            from base.logger import logger
            logger.error(
                f"[{self.name}] 执行失败: {e} | query={task.get('user_query', '')[:50]}",
                exc_info=True,
            )

            return SubAgentResult(
                agent_name=self.name,
                task_id=task.get("task_id", ""),
                status="failed",
                result=self._get_fallback_response(),
                latency_ms=latency,
                error=str(e),
            )

    # ================================================================
    # 图节点实现
    # ================================================================

    def _plan_node(self, state: dict) -> dict:
        """
        Planner 节点 — 分析 query + context，生成工具调用计划

        与单 Agent Planner 的区别:
          1. Prompt 前缀加 _get_system_prompt()
          2. Context 中包含 Orchestrator 传递的上游结果
          3. 工具列表是领域专属的 (3-5 个而非 7 个)
        """
        from base.llm_client import get_llm_client
        from base.logger import logger

        query = state.get("user_query", "")
        intent = state.get("intent", "")
        entities = state.get("entities", {})
        context = state.get("context", {})

        # ── 领域专属工具 ──
        tools = self._get_tools()
        if not tools:
            logger.info(f"[{self.name}] 无可用工具，跳过规划")
            from agent.state import Plan
            return {"plan": Plan(tool_calls=[], reasoning="无工具可用", is_complete=True)}

        tool_descriptions = "\n".join(
            f"  - {t.name}: {t.description}" for t in tools
        )

        # ── 领域专属 Prompt ──
        planner_prompt = f"""{self._get_system_prompt()}

用户问题: {query}
用户意图: {intent}
提取实体: {entities}

附加上下文:
{context}

可用工具:
{tool_descriptions}

请分析用户问题，决定需要调用哪些工具来完成回答。
输出 JSON 格式:
{{
    "tool_calls": [
        {{"tool_name": "工具名", "tool_args": {{参数}}, "depends_on": []}}
    ],
    "reasoning": "你的推理过程 (1-2句话)",
    "confidence": 0.0-1.0
}}

注意:
- depends_on 是前置依赖的工具索引 (从0开始)，空列表=无依赖
- 只选必要的工具，不要过度调用
- 如果问题简单不需要工具，tool_calls 可以为空列表
"""

        client = get_llm_client()
        plan_data = client.chat_json(
            messages=[{"role": "user", "content": planner_prompt}],
            temperature=0.1,
        )

        # 复用现有解析逻辑
        from agent.state import Plan, ToolCall

        # 检查 LLM 返回
        if not plan_data or "error" in plan_data:
            logger.warning(f"[{self.name}] Planner LLM 返回异常: {plan_data}")
            return {"plan": Plan(
                tool_calls=[],
                reasoning=f"Planner 异常: {plan_data}",
                is_complete=True,
            )}

        raw_calls = plan_data.get("tool_calls", [])
        reasoning = plan_data.get("reasoning", "LLM 动态规划")
        confidence = plan_data.get("confidence", 0.8)

        tool_calls = []
        for tc in raw_calls:
            if isinstance(tc, dict) and "tool_name" in tc:
                tool_calls.append(ToolCall(
                    tool_name=tc["tool_name"],
                    tool_args=tc.get("tool_args", {}),
                    depends_on=tc.get("depends_on", []),
                ))

        plan = Plan(tool_calls=tool_calls, reasoning=reasoning, is_complete=True)
        # 置信度暂存到 reasoning 尾部 (Plan dataclass 没有 confidence 字段)
        plan.reasoning = f"{reasoning} [confidence={confidence}]"

        logger.info(
            f"[{self.name}] 规划完成: {len(tool_calls)} 个工具 | {reasoning[:50]}"
        )

        return {"plan": plan, "current_tool_index": 0, "tool_results": []}

    def _exec_node(self, state: dict) -> dict:
        """
        Executor 节点 — 按计划执行工具调用

        直接复用 agent/graph.py 的 exec_node，因为执行逻辑完全相同。
        """
        from agent.graph import exec_node as _exec
        return _exec(state)

    def _check_node(self, state: dict) -> dict:
        """
        Checker 节点 — 检查结果完整性

        直接复用 agent/graph.py 的 reflect_node。
        """
        from agent.graph import reflect_node as _reflect
        return _reflect(state)

    def _synthesize_node(self, state: dict) -> dict:
        """
        Synthesizer 节点 — 整合工具结果，生成最终答案

        与单 Agent Synthesizer 的区别:
          1. System Prompt 使用领域专属的 (_get_system_prompt)
          2. 附加合规约束 (_get_compliance_rules)
        """
        from base.llm_client import get_llm_client
        from base.logger import logger

        query = state.get("user_query", "")
        tool_results = state.get("tool_results", [])

        # ── 收集成功结果 ──
        success_results = [r for r in tool_results if r.get("status") == "done"]

        if not success_results:
            return {"final_answer": self._get_fallback_response()}

        # ── 拼接工具结果 ──
        context_parts = []
        for r in success_results:
            context_parts.append(f"### [{r['tool']}]\n{r['result']}")
        context = "\n\n".join(context_parts)

        # ── 合规约束 ──
        rules = self._get_compliance_rules()
        forbidden = "\n".join(f"  - ❌ {p}" for p in rules.get("forbidden_phrases", []))
        disclaimers = "\n".join(f"  - ⚠️ {d}" for d in rules.get("required_disclaimers", []))

        # ── 领域专属合成 Prompt ──
        synthesize_prompt = f"""{self._get_system_prompt()}

用户问题: {query}

工具返回结果:
{context}

合规要求:
禁止表述:
{forbidden}

必须附加声明:
{disclaimers}

请基于工具结果，用清晰、专业的语言回答用户问题。
要求:
1. 直接回答问题，不要复述工具结果
2. 如有金额或条款信息，引用具体来源
3. 遵守合规要求
4. 如果不确定，明确告知用户并建议核实
"""

        client = get_llm_client()
        response = client.chat(
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": synthesize_prompt},
            ],
            temperature=0.3,
            max_tokens=2048,
        )

        if response.error:
            logger.error(f"[{self.name}] Synthesizer LLM 失败: {response.error}")
            answer = f"工具结果汇总:\n\n{context}\n\n（答案生成失败，以上为原始工具结果）"
        else:
            answer = response.content
            logger.info(
                f"[{self.name}] 答案生成完成: {len(answer)} 字符 | "
                f"latency={response.latency_ms:.0f}ms"
            )

        return {"final_answer": answer}

    # ================================================================
    # 辅助方法
    # ================================================================

    def _should_continue(self, state: dict) -> str:
        """
        条件边: 决定 check 之后走 plan 还是 synthesize

        与单 Agent 的区别:
          子 Agent 的最大迭代次数是 3 (而非 5)，因为:
          1. 子 Agent 的职责更聚焦，工具更少，不太需要反复重试
          2. 如果 3 次都搞不定，Orchestrator 会降级处理
        """
        iteration = state.get("iteration", 0)
        max_iter = state.get("max_iterations", 3)
        plan = state.get("plan")

        if iteration >= max_iter:
            return "synthesize"

        all_done = all(
            tc.status in ("done", "failed")
            for tc in (plan.tool_calls if plan else [])
        )
        if all_done:
            return "synthesize"

        return "plan"

    def _estimate_confidence(self, result: dict) -> float:
        """从结果中估算置信度"""
        plan = result.get("plan")
        if plan and hasattr(plan, "reasoning"):
            import re
            match = re.search(r'confidence=([\d.]+)', plan.reasoning)
            if match:
                return float(match.group(1))

        # 默认: 有工具结果 → 0.8，无工具结果 → 0.5
        tool_results = result.get("tool_results", [])
        success_count = sum(1 for r in tool_results if r.get("status") == "done")
        if success_count > 0:
            return 0.8
        return 0.5

    def _get_fallback_response(self) -> str:
        """降级回复 — 子 Agent 失败时的兜底话术"""
        return (
            "抱歉，当前服务繁忙，暂时无法处理您的请求。\n"
            "建议您联系人工客服获取帮助，"
            "我们的客服人员将为您详细解答。"
        )
