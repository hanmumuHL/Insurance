# -*- coding: utf-8 -*-
"""
LangGraph 状态图定义 — Agent 的核心运行逻辑 (DEPRECATED)

⚠️  此模块已废弃 — Phase 3 统一为多 Agent 架构。
────────
所有请求现统一走 agent/orchestrator.py (Orchestrator + 4 领域 Agent)。
此文件仅保留供以下用途:
  1. 面试展示: 展示从单 Agent 到多 Agent 的演进过程
  2. 代码参考: exec_node / reflect_node 等仍被子 Agent 基类复用
  3. 独立测试: 可脱离多 Agent 架构单独测试 Plan-Execute 循环

不建议在生产代码中直接 import agent.graph.run_agent()。
使用 agent/orchestrator.py:Orchestrator.process() 替代。"""

from langgraph.graph import StateGraph, START, END

from agent.state import Plan, ToolCall
from agent.tools.all_tools import get_all_tools, get_tool_by_name
from base.logger import logger
from base.llm_client import get_llm_client


# ============================================================
# 工具依赖注入 — 从全局上下文获取服务实例
# ============================================================


def _get_tools_with_deps(tool_name: str = None):
    """
    获取注入真实依赖的工具实例

    Args:
        tool_name: 若指定，返回单个工具；否则返回全部工具列表

    优先级:
      1. 检查 agent/tools/all_tools.py 中的 _global_deps 字典
      2. LLM 客户端始终从全局单例获取
      3. VectorStore / MySQL / Redis 由启动时 gateway/app.py 注入
      4. 都没有 → 返回无依赖的工具（返回"服务未连接"错误）
    """
    vector_store = None
    mysql_session = None
    redis_session = None
    llm_client = None

    try:
        from base.llm_client import get_llm_client

        llm_client = get_llm_client()
    except Exception:
        pass

    # 来自 gateway/app.py 启动时注入的全局依赖
    try:
        from agent.tools.all_tools import _global_deps

        vector_store = _global_deps.get("vector_store")
        mysql_session = _global_deps.get("mysql_session")
        redis_session = _global_deps.get("redis_session")
    except Exception:
        pass

    all_tools = get_all_tools(
        vector_store=vector_store,
        mysql_session=mysql_session,
        redis_session=redis_session,
        llm_client=llm_client,
    )

    if tool_name:
        for t in all_tools:
            if t.name == tool_name:
                return t
        return None

    return all_tools


# ============================================================
# 节点 1: Planner — 分析用户意图，生成工具调用计划
# ============================================================


def plan_node(state: dict) -> dict:
    """
    Planner 节点 — LLM 分析 query，生成执行计划

    输入 state 中的关键信息:
      user_query: 用户问题
      intent: 意图分类结果
      entities: 提取的实体

    输出 (写入 state):
      plan: 执行计划 (工具列表 + 参数)

    LLM 的工作:
      1. 分析用户意图，理解需要做什么
      2. 选择合适的工具（从 7 个工具中选）
      3. 确定工具的调用参数
      4. 确定调用顺序（哪些可以并行，哪些有依赖）
    """
    query = state.get("user_query", "")
    intent = state.get("intent", "")
    entities = state.get("entities", {})

    logger.info(f"[Planner] 开始规划: query='{query[:50]}' intent={intent}")

    # ── 获取可用工具列表 ──
    # 工具需要依赖注入: VectorStore / MySQL / LLM
    # 从全局上下文获取这些服务实例
    tools = _get_tools_with_deps()
    tool_descriptions = "\n".join(f"- {t.name}: {t.description}" for t in tools)

    # ── 构造 Planner Prompt ──
    # 让 LLM 决定调用哪些工具、什么参数、什么顺序
    planner_prompt = f"""你是一个保险智能客服的任务规划器。

用户问题: {query}
识别的意图: {intent}
提取的实体: {entities}

可用工具:
{tool_descriptions}

请分析用户问题，决定需要调用哪些工具来完成回答。
输出 JSON 格式:
{{
    "tool_calls": [
        {{"tool_name": "工具名", "tool_args": {{参数}}, "depends_on": []}}
    ],
    "reasoning": "你的推理过程"
}}

注意:
- depends_on 是依赖的工具调用索引（从0开始），空列表表示无依赖
- 只选择必要的工具，不要过度调用
- 如果问题很简单（如闲聊），tool_calls 可以为空列表
"""

    # ── 调用 LLM (DeepSeek 做规划推理) ──
    # 让 LLM 动态决定调用哪些工具、什么参数、什么顺序
    # 使用 chat_json 确保返回结构化 JSON
    client = get_llm_client()
    plan_data = client.chat_json(
        messages=[{"role": "user", "content": planner_prompt}],
        temperature=0.1,  # 规划任务需要确定性
    )

    # 解析 LLM 返回的工具调用计划
    plan = _parse_plan_from_llm(plan_data, intent, entities)

    logger.info(
        f"[Planner] 计划完成: {len(plan.tool_calls)} 个工具调用 | {plan.reasoning}"
    )

    return {
        "plan": plan,
        "current_tool_index": 0,
        "tool_results": [],
    }


def _parse_plan_from_llm(plan_data: dict, intent: str, entities: dict) -> Plan:
    """
    解析 LLM 返回的 JSON 为 Plan 对象

    LLM 返回的 JSON 格式:
    {
        "tool_calls": [
            {"tool_name": "policy_query", "tool_args": {"insurer": "平安"}, "depends_on": []}
        ],
        "reasoning": "理赔咨询需要先查保单..."
    }

    如果 LLM 返回格式异常，降级为基于意图的默认计划。

    Args:
        plan_data: LLM 返回的 JSON dict
        intent: 意图分类结果 (用于降级)
        entities: 提取的实体 (用于降级)

    Returns:
        Plan: 解析后的执行计划
    """
    # ── 检查 LLM 返回是否有效 ──
    if not plan_data or "error" in plan_data:
        logger.warning(f"LLM 规划返回异常: {plan_data}, 降级为默认计划")
        return _generate_default_plan(intent, entities)

    # ── 解析 tool_calls ──
    raw_calls = plan_data.get("tool_calls", [])
    reasoning = plan_data.get("reasoning", "LLM 动态规划")

    if not raw_calls:
        logger.info("LLM 规划: 无需工具调用")
        return Plan(tool_calls=[], reasoning=reasoning, is_complete=True)

    tool_calls = []
    for tc in raw_calls:
        if not isinstance(tc, dict) or "tool_name" not in tc:
            continue

        tool_calls.append(
            ToolCall(
                tool_name=tc["tool_name"],
                tool_args=tc.get("tool_args", {}),
                depends_on=tc.get("depends_on", []),
            )
        )

    logger.info(f"LLM 规划解析: {len(tool_calls)} 个工具调用")

    return Plan(
        tool_calls=tool_calls,
        reasoning=reasoning,
        is_complete=True,
    )


def _generate_default_plan(intent: str, entities: dict) -> Plan:
    """
    根据意图生成执行计划（占位逻辑，实际由 LLM 动态生成）

    不同意图对应不同的默认工具组合:
      理赔咨询 → 保单查询 + 条款检索 + 理赔资格预检
      产品对比 → 多产品对比
      保费试算 → 保费试算
      保单查询 → 保单查询
    """
    tool_calls = []

    if intent == "理赔咨询":
        # 理赔咨询需要先查保单，再查条款，最后预检资格
        tool_calls = [
            ToolCall(
                tool_name="policy_query",
                tool_args={
                    "insurer": entities.get("insurer", ""),
                    "product_name": entities.get("product_name", ""),
                },
                depends_on=[],  # 无依赖，第一个执行
            ),
            ToolCall(
                tool_name="clause_search",
                tool_args={
                    "product_name": entities.get("product_name", ""),
                    "keywords": "理赔 赔付",
                },
                depends_on=[],  # 可以和保单查询并行
            ),
            ToolCall(
                tool_name="claim_eligibility",
                tool_args={
                    "disease_or_event": entities.get("event", ""),
                    "product_name": entities.get("product_name", ""),
                    "insurer": entities.get("insurer", ""),
                },
                depends_on=[0, 1],  # 依赖前两个工具的结果
            ),
        ]
        reasoning = "理赔咨询: 查保单→查条款→预检资格 (串行+并行)"

    elif intent == "产品对比":
        tool_calls = [
            ToolCall(
                tool_name="product_compare",
                tool_args={
                    "products": f"{entities.get('product_a', '')},{entities.get('product_b', '')}"
                },
                depends_on=[],
            ),
        ]
        reasoning = "产品对比: 直接调用对比工具"

    elif intent == "保费试算":
        tool_calls = [
            ToolCall(
                tool_name="premium_calc",
                tool_args={
                    "product_name": entities.get("product_name", ""),
                    "age": entities.get("age", 30),
                },
                depends_on=[],
            ),
        ]
        reasoning = "保费试算: 直接调用试算工具"

    elif intent == "保单查询":
        tool_calls = [
            ToolCall(
                tool_name="policy_query",
                tool_args={
                    "insurer": entities.get("insurer", ""),
                    "product_name": entities.get("product_name", ""),
                },
                depends_on=[],
            ),
        ]
        reasoning = "保单查询: 直接调用查询工具"

    elif intent == "条款解读":
        tool_calls = [
            ToolCall(
                tool_name="clause_search",
                tool_args={
                    "product_name": entities.get("product_name", ""),
                    "keywords": "",
                },
                depends_on=[],
            ),
        ]
        reasoning = "条款解读: 检索相关条款"

    else:
        # 其他意图不需要工具调用
        reasoning = f"{intent}: 无需工具调用"

    return Plan(
        tool_calls=tool_calls,
        reasoning=reasoning,
        is_complete=True,
    )


# ============================================================
# 节点 2: Executor — 按计划执行工具调用
# ============================================================


def exec_node(state: dict) -> dict:
    """
    Executor 节点 — 按 Planner 的计划执行工具调用

    执行策略:
      1. 遍历 plan.tool_calls
      2. 检查依赖: depends_on 中的工具是否都已完成
      3. 无依赖的工具可以并行（当前简化为串行）
      4. 调用工具的 _run 方法，收集结果
      5. 记录每个工具的执行状态

    输入 state:
      plan: 执行计划
      current_tool_index: 当前执行到第几个工具

    输出 (写入 state):
      tool_results: 工具执行结果列表
      current_tool_index: 更新到下一个待执行的工具
    """
    plan = state.get("plan")
    if not plan or not plan.tool_calls:
        logger.info("[Executor] 无工具调用计划")
        return {"tool_results": []}

    tool_calls = plan.tool_calls
    results = list(state.get("tool_results", []))

    logger.info(f"[Executor] 开始执行: {len(tool_calls)} 个工具调用")

    for i, tc in enumerate(tool_calls):
        # 跳过已完成和正在运行的
        if tc.status in ("done", "running"):
            continue

        # 检查依赖是否都已完成
        deps_ok = all(
            tool_calls[dep_idx].status == "done"
            for dep_idx in tc.depends_on
            if dep_idx < len(tool_calls)
        )
        if not deps_ok:
            logger.info(f"[Executor] 工具 {tc.tool_name} 等待依赖完成")
            continue

        # ── 执行工具 ──
        tc.status = "running"
        tool = _get_tools_with_deps(tc.tool_name)

        if tool is None:
            logger.error(f"[Executor] 工具不存在: {tc.tool_name}")
            tc.status = "failed"
            tc.result = f"工具 {tc.tool_name} 不存在"
            results.append(
                {"tool": tc.tool_name, "status": "failed", "result": tc.result}
            )
            continue

        try:
            # 调用工具的 _run 方法
            result = tool._run(**tc.tool_args)
            tc.status = "done"
            tc.result = result
            results.append({"tool": tc.tool_name, "status": "done", "result": result})
            logger.info(f"[Executor] 工具 {tc.tool_name} 执行成功")
        except Exception as e:
            tc.status = "failed"
            tc.result = f"工具执行失败: {e}"
            results.append(
                {"tool": tc.tool_name, "status": "failed", "result": tc.result}
            )
            logger.error(f"[Executor] 工具 {tc.tool_name} 执行失败: {e}")

    return {
        "plan": plan,
        "tool_results": results,
    }


# ============================================================
# 节点 3: Reflector — 检查结果是否完整
# ============================================================


def reflect_node(state: dict) -> dict:
    """
    Reflector 节点 — 检查工具执行结果是否足以回答用户问题

    检查维度:
      1. 所有工具是否都执行成功
      2. 结果是否包含足够的信息
      3. 是否需要补充调用其他工具

    输出:
      如果结果充分 → 流转到 synthesize
      如果结果不足 → 增加 iteration，流转到 plan 补充
    """
    plan = state.get("plan")
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 5)

    # 检查是否所有工具都已完成
    all_done = all(
        tc.status in ("done", "failed") for tc in (plan.tool_calls if plan else [])
    )

    if not all_done and iteration < max_iter:
        logger.info(
            f"[Reflector] 部分工具未完成，返回 Planner (iteration={iteration + 1})"
        )
        return {"iteration": iteration + 1}

    # 检查是否有工具失败
    failed = [tc for tc in (plan.tool_calls if plan else []) if tc.status == "failed"]
    if failed and iteration < max_iter:
        logger.warning(f"[Reflector] {len(failed)} 个工具失败，尝试重试")
        # 重置失败工具的状态为 pending，让 Planner 重新规划
        for tc in failed:
            tc.status = "pending"
        return {"iteration": iteration + 1}

    logger.info("[Reflector] 所有工具执行完毕，进入 Synthesize")
    return {"iteration": iteration}


def should_continue(state: dict) -> str:
    """
    条件边: 决定 Reflect 之后的流转方向

    返回:
      "plan" → 回到 Planner 补充工具调用
      "synthesize" → 进入 Synthesizer 生成最终答案
    """
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 5)
    plan = state.get("plan")

    # 超过最大迭代次数 → 强制进入 synthesize
    if iteration >= max_iter:
        return "synthesize"

    # 所有工具都已完成 → 进入 synthesize
    all_done = all(
        tc.status in ("done", "failed") for tc in (plan.tool_calls if plan else [])
    )
    if all_done:
        return "synthesize"

    # 还有工具未完成 → 回到 plan
    return "plan"


# ============================================================
# 节点 4: Synthesizer — 整合工具结果，生成最终答案
# ============================================================


def synthesize_node(state: dict) -> dict:
    """
    Synthesizer 节点 — 整合所有工具的执行结果，生成最终答案

    输入 state:
      user_query: 用户问题
      tool_results: 所有工具的执行结果

    处理:
      1. 收集所有工具的成功结果
      2. 调用 LLM，将工具结果整合为自然语言答案
      3. 添加条款引用和免责声明

    输出:
      final_answer: 最终答案文本
    """
    query = state.get("user_query", "")
    tool_results = state.get("tool_results", [])

    # 收集成功结果
    success_results = [r for r in tool_results if r["status"] == "done"]

    if not success_results:
        return {
            "final_answer": "抱歉，暂时无法获取相关信息。请稍后再试或联系人工客服。",
        }

    # ── 拼接工具结果作为 LLM 的参考上下文 ──
    context_parts = []
    for r in success_results:
        context_parts.append(f"[{r['tool']}]\n{r['result']}")
    context = "\n\n".join(context_parts)

    # ── 调用 LLM 整合结果 ──
    synthesize_prompt = f"""你是一个保险智能客服。请根据工具返回的结果，
用清晰、专业的语言回答用户的问题。

用户问题: {query}

工具返回结果:
{context}

要求:
1. 直接回答用户的问题，不要复述工具结果
2. 如有金额信息，标注"仅供参考，以条款为准"
3. 引用具体条款时使用"根据条款第X条"的格式
4. 如果工具结果不足以回答，明确告知用户
"""

    # 调用 LLM 生成最终答案
    client = get_llm_client()
    response = client.chat(
        messages=[
            {
                "role": "system",
                "content": "你是专业的保险智能客服，基于工具返回的结果回答用户问题。",
            },
            {"role": "user", "content": synthesize_prompt},
        ],
        temperature=0.3,
        max_tokens=2048,
    )

    if response.error:
        logger.error(f"[Synthesizer] LLM 调用失败: {response.error}")
        answer = (
            f"工具结果汇总:\n\n{context}\n\n（答案生成失败，请直接参考以上工具结果）"
        )
    else:
        answer = response.content
        logger.info(
            f"[Synthesizer] 答案生成完成: {len(answer)} 字符, "
            f"tokens={response.tokens_used.get('total', '?')}, "
            f"latency={response.latency_ms:.0f}ms"
        )

    return {
        "final_answer": answer,
    }


# ============================================================
# 构建 LangGraph 状态图
# ============================================================


def build_agent_graph():
    """
    构建 Agent 的 LangGraph 状态图

    图结构:
      START → plan → exec → reflect ─(条件边)─→ plan / synthesize
                                                synthesize → END

    返回编译后的 graph，可以直接调用:
      result = graph.invoke({"user_query": "肺炎住院能赔吗？"})
    """

    # 创建状态图，指定状态类型
    graph = StateGraph(dict)  # 用 dict 而不是 dataclass，兼容性更好

    # ── 添加节点 ──
    graph.add_node("plan", plan_node)
    graph.add_node("exec", exec_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("synthesize", synthesize_node)

    # ── 添加边 ──

    # START → plan (入口)
    graph.add_edge(START, "plan")

    # plan → exec (规划完就执行)
    graph.add_edge("plan", "exec")

    # exec → reflect (执行完就检查)
    graph.add_edge("exec", "reflect")

    # reflect → plan / synthesize (条件分支)
    graph.add_conditional_edges(
        "reflect",
        should_continue,
        {
            "plan": "plan",  # 结果不足 → 回到 Planner
            "synthesize": "synthesize",  # 结果充分 → 生成答案
        },
    )

    # synthesize → END (完成)
    graph.add_edge("synthesize", END)

    # ── 编译图 ──
    # 可以传入 checkpointer 实现状态持久化
    # from langgraph.checkpoint.memory import MemorySaver
    # checkpointer = MemorySaver()
    # app = graph.compile(checkpointer=checkpointer)

    app = graph.compile()

    logger.info("Agent Graph 构建完成")
    return app


# ============================================================
# 便捷入口: 运行 Agent
# ============================================================


def run_agent(
    query: str, intent: str = "", entities: dict = None, session_id: str = ""
) -> dict:
    """
    运行 Agent — 便捷入口函数

    Args:
        query: 用户问题 (已脱敏)
        intent: 意图分类结果
        entities: 提取的实体
        session_id: 会话 ID

    Returns:
        dict: 包含 final_answer 和中间过程信息
    """
    graph = build_agent_graph()

    # 构造初始状态
    initial_state = {
        "user_query": query,
        "intent": intent,
        "entities": entities or {},
        "session_id": session_id,
        "messages": [],
        "iteration": 0,
        "max_iterations": 5,
        "plan": None,
        "current_tool_index": 0,
        "tool_results": [],
        "final_answer": "",
        "sources": [],
        "error": "",
        "pipeline": {},
    }

    # 运行图
    try:
        result = graph.invoke(initial_state)
        return result
    except Exception as e:
        logger.error(f"Agent 运行失败: {e}", exc_info=True)
        return {
            "final_answer": "抱歉，系统暂时繁忙，请稍后再试或联系人工客服。",
            "error": str(e),
        }
