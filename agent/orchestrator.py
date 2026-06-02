# -*- coding: utf-8 -*-
"""
Orchestrator — 多 Agent 调度中心

职责:
  1. 意图路由: 识别用户意图 → 决定走单 Agent / 多 Agent / RAG 降级
  2. 复杂度判断: 简单查询走客服 Agent，复杂理赔走理赔 Agent + 核保 Agent
  3. 任务拆解: 将一个用户请求拆成多个子 Agent 任务
  4. 任务分发: 按依赖顺序串行/并行调用子 Agent
  5. 结果聚合: 收集所有子 Agent 结果 → 合成最终答案
  6. 合规终审: 全局合规检查（跨 Agent 一致性校验）

通信模式: Orchestrator 中心化调度
  - 子 Agent 之间不直接通信
  - Orchestrator 持有全局上下文
  - 上游 Agent 的输出作为下游 Agent 的 context 传入

使用方式:
    orch = Orchestrator(agents={
        "insurance": InsuranceAgent(...),
        "underwriting": UnderwritingAgent(...),
        "claim": ClaimAgent(...),
        "service": ServiceAgent(...),
    })

    result = orch.process(
        query="平安e生保肺炎住院能赔吗",
        intent="理赔咨询",
        entities={"insurer": "平安", "product_name": "e生保"},
        session_id="sess_123",
    )
"""

import time

from agent.state import SubAgentTask
from agent.sub_agents.base import SubAgentResult
from base.llm_client import get_llm_client
from base.logger import logger

# ================================================================
# 意图 → Agent 路由表
# ================================================================
# 每种意图对应的主 Agent、辅助 Agent、路由模式。
# 新增意图类型时只需在此表中添加一行。

INTENT_ROUTING = {
    # ── 投保相关 → 投保 Agent ──
    "产品咨询": {
        "primary": "insurance",
        "secondary": None,
        "mode": "single_agent",
        "description": "产品推荐/保障范围解释/免赔额咨询",
    },
    "产品对比": {
        "primary": "insurance",
        "secondary": None,
        "mode": "single_agent",
        "description": "多产品维度对比",
    },
    "保费试算": {
        "primary": "insurance",
        "secondary": None,
        "mode": "single_agent",
        "description": "保费计算",  # 简单试算，不需要核保
    },
    "投保流程": {
        "primary": "insurance",
        "secondary": "underwriting",
        "mode": "multi_agent",
        "description": "投保引导 + 核保审核",
    },
    # ── 核保相关 → 核保 Agent ──
    "核保咨询": {
        "primary": "underwriting",
        "secondary": None,
        "mode": "single_agent",
        "description": "健康告知/核保规则/风险评估",
    },
    # ── 理赔相关 → 理赔 Agent (+ 核保 Agent 辅助) ──
    "理赔咨询": {
        "primary": "claim",
        "secondary": "underwriting",
        "mode": "multi_agent",
        "description": "理赔资格判断 (需要核保确认保障范围/等待期)",
    },
    "理赔进度": {
        "primary": "claim",
        "secondary": "service",
        "mode": "multi_agent",
        "description": "查询理赔进度 + 提供客服支持",
    },
    "条款解读": {
        "primary": "claim",
        "secondary": None,
        "mode": "single_agent",
        "description": "条款原文检索与解释",
    },
    # ── 客服相关 → 客服 Agent ──
    "保单查询": {
        "primary": "service",
        "secondary": None,
        "mode": "single_agent",
        "description": "查询保单状态/保障信息",
    },
    # ── 降级场景 → 不走 Agent ──
    "投诉建议": {
        "primary": "service",
        "secondary": None,
        "mode": "rag_fallback",
        "description": "投诉直接转人工，不走 Agent 推理",
    },
    "闲聊": {
        "primary": "service",
        "secondary": None,
        "mode": "rag_fallback",
        "description": "闲聊直接 RAG FAQ 匹配",
    },
}


class Orchestrator:
    """
    Orchestrator — 多 Agent 调度中心

    工作流:
      1. process(query, intent, entities) — 主入口
      2. _route_intent(intent, query) — 查路由表 + 复杂度判断
      3. _build_tasks(query, route) — 拆解为 SubAgentTask 列表
      4. _execute_tasks(tasks) — 按依赖顺序调用子 Agent
      5. _aggregate_results(query, results, tasks) — 合成最终答案
      6. _compliance_review(answer) — 全局合规检查
    """

    def __init__(self, agents: dict, rag_system=None):
        """
        Args:
            agents: {"insurance": SubAgent, "underwriting": SubAgent, ...}
                    至少包含 claim 和 service（最常用的两个）
            rag_system: RAG 系统实例 (降级时使用)
        """
        self.agents = agents
        self.rag_system = rag_system
        self._validate_agents()

    def _validate_agents(self):
        """验证已注册的 Agent 是否覆盖路由表中的引用"""
        registered = set(self.agents.keys())
        logger.info(f"[Orchestrator] 已注册 Agent: {registered}")

        # 检查路由表引用的 Agent 是否都已注册
        for intent, route in INTENT_ROUTING.items():
            primary = route["primary"]
            secondary = route.get("secondary")
            if primary not in registered:
                logger.warning(
                    f"[Orchestrator] 路由表引用未注册 Agent: '{primary}' (意图: {intent})"
                )
            if secondary and secondary not in registered:
                logger.warning(
                    f"[Orchestrator] 路由表引用未注册 Agent: '{secondary}' (意图: {intent})"
                )

    # ================================================================
    # 主入口
    # ================================================================

    def process(
        self,
        query: str,
        intent: str = "",
        entities: dict = None,
        session_id: str = "",
        user_profile: dict = None,
    ) -> dict:
        """
        处理用户请求 — 完整的多 Agent 协作流程

        Args:
            query: 用户问题 (已脱敏)
            intent: 意图分类结果 (如 "理赔咨询")
            entities: 提取的实体 (如 {"insurer": "平安", "product_name": "e生保"})
            session_id: 会话 ID
            user_profile: 长期记忆中的用户画像

        Returns:
            dict: {
                "answer": str,                # 最终答案
                "route_mode": str,            # "single_agent" | "multi_agent" | "rag_fallback"
                "agents_used": [str, ...],    # 使用了哪些 Agent
                "sources": [dict, ...],       # 条款引用
                "pipeline": {str: float},     # 各阶段耗时 (ms)
                "latency_ms": float,          # 总耗时
            }
        """
        t0 = time.time()
        pipeline = {}

        # ── Step 1: 意图路由 ──
        t1 = time.time()
        route = self._route_intent(intent, query)
        pipeline["route"] = round((time.time() - t1) * 1000, 1)
        logger.info(
            f"[Orchestrator] 路由完成: intent={intent} → "
            f"primary={route['primary']} mode={route['mode']} "
            f"({pipeline['route']}ms)"
        )

        # ── Step 2: 降级判断 ──
        if route["mode"] == "rag_fallback":
            logger.info(f"[Orchestrator] 降级为 RAG: intent={intent}")
            result = self._rag_fallback(query, intent, entities)
            result["pipeline"] = {
                **pipeline,
                "total": round((time.time() - t0) * 1000, 1),
            }
            return result

        # ── Step 3: 构建任务列表 ──
        t2 = time.time()
        tasks = self._build_tasks(
            query, intent, entities or {}, route, user_profile or {}
        )
        pipeline["task_build"] = round((time.time() - t2) * 1000, 1)

        # ── Step 4: 执行任务 (按依赖顺序) ──
        t3 = time.time()
        results = self._execute_tasks(tasks, session_id)
        pipeline["execution"] = round((time.time() - t3) * 1000, 1)

        # ── Step 5: 聚合结果 ──
        t4 = time.time()
        final = self._aggregate_results(query, intent, results, tasks)
        pipeline["aggregate"] = round((time.time() - t4) * 1000, 1)

        # ── Step 6: 合规终审 ──
        t5 = time.time()
        final = self._compliance_review(final, intent)
        pipeline["compliance"] = round((time.time() - t5) * 1000, 1)

        pipeline["total"] = round((time.time() - t0) * 1000, 1)

        logger.info(
            f"[Orchestrator] 完成: mode={route['mode']} agents={[t.agent_name for t in tasks]} "
            f"total={pipeline['total']}ms "
            f"(route={pipeline['route']}ms build={pipeline['task_build']}ms "
            f"exec={pipeline['execution']}ms agg={pipeline['aggregate']}ms "
            f"compliance={pipeline['compliance']}ms)"
        )

        return {
            "answer": final,
            "route_mode": route["mode"],
            "agents_used": [t.agent_name for t in tasks],
            "sources": self._collect_sources(results),
            "pipeline": pipeline,
            "latency_ms": pipeline["total"],
        }

    # ================================================================
    # 意图路由
    # ================================================================

    def _route_intent(self, intent: str, query: str) -> dict:
        """
        根据意图决定路由策略

        路由规则:
          1. 查 INTENT_ROUTING 表 → 命中则直接返回
          2. 意图未知 → LLM 辅助分类
          3. 简单 query (≤8字且非复杂意图) → 强制降级 RAG

        Returns:
            dict: {"primary": str, "secondary": str|None, "mode": str}
        """
        # ── 查路由表 ──
        if intent in INTENT_ROUTING:
            route = INTENT_ROUTING[intent].copy()

            # 简单 query 降级: 短问题且非复杂意图 → 直接 FAQ
            complex_intents = {"理赔咨询", "产品对比", "核保咨询", "投保流程"}
            if len(query) <= 8 and intent not in complex_intents:
                route["mode"] = "rag_fallback"
                logger.info(
                    f"[Orchestrator] 简单 query '{query[:20]}' (intent={intent}) → RAG 降级"
                )

            return route

        # ── 未知意图 → LLM 分类 ──
        logger.info(f"[Orchestrator] 未知意图 '{intent}'，LLM 辅助分类")
        return self._llm_classify_intent(query)

    def _llm_classify_intent(self, query: str) -> dict:
        """
        LLM 辅助意图分类 (路由表未命中时)

        当意图分类器的输出不在 INTENT_ROUTING 表中时，
        用 LLM 零样本判断应该路由到哪个 Agent。
        """
        client = get_llm_client()

        valid_intents = "\n".join(
            f"  - {k}: {v['description']}" for k, v in INTENT_ROUTING.items()
        )

        prompt = f"""将用户问题分类到以下保险业务场景之一:

{valid_intents}

用户问题: {query}

输出 JSON:
{{
    "intent": "场景名",
    "confidence": 0.0-1.0
}}

注意:
- 只输出 JSON，不要其他内容
- 如果无法确定，用 "闲聊" 且 confidence=0.1
"""

        try:
            response = client.chat_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )

            intent = response.get("intent", "闲聊")
            confidence = response.get("confidence", 0.5)

            # 低置信度 → 降级 FAQ
            if confidence < 0.6:
                logger.info(
                    f"[Orchestrator] LLM 分类低置信度 ({confidence}) → RAG 降级"
                )
                return {"primary": "service", "secondary": None, "mode": "rag_fallback"}

            # 查路由表
            if intent in INTENT_ROUTING:
                return INTENT_ROUTING[intent].copy()

            logger.warning(
                f"[Orchestrator] LLM 分类 '{intent}' 不在路由表中 → RAG 降级"
            )

        except Exception as e:
            logger.error(f"[Orchestrator] LLM 分类失败: {e}")

        # 兜底
        return {"primary": "service", "secondary": None, "mode": "rag_fallback"}

    # ================================================================
    # 任务构建
    # ================================================================

    def _build_tasks(
        self,
        query: str,
        intent: str,
        entities: dict,
        route: dict,
        user_profile: dict,
    ) -> list[SubAgentTask]:
        """
        将用户请求拆解为子 Agent 任务列表

        策略:
          single_agent → 1 个任务 (主 Agent)
          multi_agent  → 2 个任务 (主 Agent + 辅助 Agent，辅助依赖主)
        """
        tasks = []

        # ── 主 Agent 任务 ──
        primary = SubAgentTask(
            task_id="task_0",
            agent_name=route["primary"],
            intent=intent,
            user_query=query,
            entities=entities,
            context={"user_profile": user_profile},
            dependencies=[],  # 无依赖，第一个执行
            priority=10,
        )
        tasks.append(primary)
        logger.info(
            f"[Orchestrator] 主任务 task_0: {primary.agent_name} (intent={intent})"
        )

        # ── 辅助 Agent 任务 (仅 multi_agent 模式) ──
        if route.get("secondary") and route["mode"] == "multi_agent":
            secondary = SubAgentTask(
                task_id="task_1",
                agent_name=route["secondary"],
                intent=self._derive_secondary_intent(intent, route["secondary"]),
                user_query=query,
                entities=entities,
                context={
                    "user_profile": user_profile,
                    "awaiting_result_from": "task_0",
                },
                dependencies=["task_0"],  # 依赖主 Agent 完成
                priority=5,
            )
            tasks.append(secondary)
            logger.info(
                f"[Orchestrator] 辅助任务 task_1: {secondary.agent_name} (依赖 task_0)"
            )

        return tasks

    def _derive_secondary_intent(
        self, primary_intent: str, secondary_agent: str
    ) -> str:
        """
        根据主意图推导辅助 Agent 的子意图

        例如: "理赔咨询"(主) + "underwriting"(辅) → "核保咨询"
             因为理赔需要核保确认等待期和保障范围。
        """
        mapping = {
            ("投保流程", "underwriting"): "核保咨询",
            ("理赔咨询", "underwriting"): "核保咨询",
            ("理赔进度", "service"): "保单查询",
        }
        derived = mapping.get((primary_intent, secondary_agent), primary_intent)
        logger.info(
            f"[Orchestrator] 辅助意图推导: {primary_intent}+{secondary_agent} → {derived}"
        )
        return derived

    # ================================================================
    # 任务执行
    # ================================================================

    def _execute_tasks(self, tasks: list[SubAgentTask], session_id: str) -> dict:
        """
        按依赖顺序执行任务

        执行策略:
          1. 无依赖的任务 → 直接执行
          2. 有依赖的任务 → 等上游完成后，将上游结果注入 context
          3. 任务失败 → 标记但不阻塞后续无强依赖的任务

        Returns:
            dict: {task_id: SubAgentResult}
        """
        results = {}
        completed = set()

        for task in tasks:
            # ── 检查依赖是否满足 ──
            deps_ready = all(dep in completed for dep in task.dependencies)
            if not deps_ready:
                logger.warning(
                    f"[Orchestrator] {task.task_id} 依赖未满足: "
                    f"need={task.dependencies} done={completed}"
                )
                # 仍然执行，但记录警告
                task.context["dependency_warning"] = "部分依赖未完成"

            # ── 注入上游结果 ──
            for dep_id in task.dependencies:
                dep_result = results.get(dep_id)
                if dep_result and dep_result.status in ("success", "degraded"):
                    task.context["upstream_result"] = dep_result.result
                    task.context["upstream_data"] = dep_result.structured_data
                    task.context["upstream_agent"] = dep_result.agent_name
                    logger.info(
                        f"[Orchestrator] {task.task_id} 接收上游 {dep_id} "
                        f"({dep_result.agent_name}) 结果: {dep_result.result[:80]}..."
                    )

            # ── 执行 ──
            agent = self.agents.get(task.agent_name)
            if agent is None:
                logger.error(
                    f"[Orchestrator] Agent '{task.agent_name}' 未注册! "
                    f"可用: {list(self.agents.keys())}"
                )
                results[task.task_id] = SubAgentResult(
                    agent_name=task.agent_name,
                    task_id=task.task_id,
                    status="failed",
                    result=f"Agent '{task.agent_name}' 未注册，请联系管理员。",
                    error="agent_not_found",
                )
                completed.add(task.task_id)
                continue

            logger.info(
                f"[Orchestrator] 执行 {task.task_id}: {task.agent_name} "
                f"(query={task.user_query[:50]})"
            )

            result = agent.invoke(
                {
                    "task_id": task.task_id,
                    "intent": task.intent,
                    "user_query": task.user_query,
                    "context": task.context,
                    "entities": task.entities,
                    "session_id": session_id,
                }
            )

            results[task.task_id] = result
            completed.add(task.task_id)

            status_icon = {"success": "✓", "degraded": "△", "failed": "✗"}.get(
                result.status, "?"
            )
            logger.info(
                f"[Orchestrator] {status_icon} {task.task_id} ({task.agent_name}): "
                f"status={result.status} latency={result.latency_ms:.0f}ms "
                f"confidence={result.confidence:.2f}"
            )

            if result.status == "failed":
                logger.warning(
                    f"[Orchestrator] {task.task_id} 失败: {result.error[:100]}"
                )

        return results

    # ================================================================
    # 结果聚合
    # ================================================================

    def _aggregate_results(
        self,
        query: str,
        intent: str,
        results: dict,
        tasks: list[SubAgentTask],
    ) -> str:
        """
        聚合所有子 Agent 的结果为最终答案

        策略:
          单 Agent → 直接返回该 Agent 的结果
          多 Agent → LLM 整合多个 Agent 的结果
        """
        # ── 收集成功结果 ──
        success = {
            tid: r for tid, r in results.items() if r.status in ("success", "degraded")
        }
        failed = {tid: r for tid, r in results.items() if r.status == "failed"}

        # ── 全部失败 ──
        if not success:
            logger.warning("[Orchestrator] 所有 Agent 失败，返回降级回复")
            failed_names = [r.agent_name for r in failed.values()]
            return (
                f"抱歉，{'、'.join(failed_names)} 服务当前不可用。\n"
                "建议您联系人工客服获取帮助。"
            )

        # ── 单 Agent → 直接返回 ──
        if len(success) == 1:
            result = list(success.values())[0]
            answer = result.result
            if result.disclaimer:
                answer += f"\n\n---\n{result.disclaimer}"
            return answer

        # ── 多 Agent → LLM 聚合 ──
        return self._llm_aggregate(query, intent, success, tasks)

    def _llm_aggregate(
        self,
        query: str,
        intent: str,
        results: dict,
        tasks: list[SubAgentTask],
    ) -> str:
        """LLM 整合多个 Agent 的结果"""
        # 拼接各 Agent 结果
        context_parts = []
        for tid, r in results.items():
            task = next((t for t in tasks if t.task_id == tid), None)
            label = f"{task.agent_name} Agent" if task else "Unknown Agent"
            context_parts.append(f"## [{label}]\n{r.result}")

        context = "\n\n".join(context_parts)

        # ── 聚合 Prompt ──
        client = get_llm_client()
        prompt = f"""你是保险智能客服。以下多个专业 Agent 分别给出了分析结果，
请整合为一个连贯、专业的回答。

用户问题: {query}
意图: {intent}

{context}

要求:
1. 整合各 Agent 的结果，自然过渡，不要简单拼接
2. 如果各 Agent 结果有冲突，标注差异并向用户说明
3. 保持专业、清晰、有温度的语气
4. 涉及数字和条款必须引用来源
5. 答案长度控制在 300 字以内（简洁优先）
"""

        response = client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2048,
        )

        if response.error:
            logger.error(f"[Orchestrator] 聚合 LLM 失败: {response.error}")
            # 降级: 直接拼接
            parts = [f"【{tid}】{r.result}" for tid, r in results.items()]
            return "\n\n---\n\n".join(parts)

        return response.content

    # ================================================================
    # 合规终审
    # ================================================================

    def _compliance_review(self, answer: str, intent: str) -> str:
        """
        全局合规终审

        检查所有 Agent 的输出是否符合保险行业合规要求。
        这是最后一道防线，任何子 Agent 的输出都会经过此处。

        检查项:
          1. 确定性承诺 (如 "一定能赔" → 替换为"以审核为准")
          2. 医疗建议检测 (如 "建议你吃药" → 追加警告)
          3. 缺失免责声明补偿
        """
        replacements = [
            ("一定能赔", "最终理赔结果以保险公司审核为准"),
            ("肯定可以赔", "理赔资格需经保险公司正式审核"),
            ("100%能理赔", "理赔结果以最终审核为准"),
            ("保证承保", "承保结果以核保审核为准"),
            ("一定承保", "核保结果以正式审核为准"),
            ("肯定能买", "投保结果以核保审核为准"),
        ]

        for phrase, replacement in replacements:
            if phrase in answer:
                answer = answer.replace(phrase, replacement)
                logger.warning(
                    f"[Orchestrator] 合规: 替换禁止表述 '{phrase}' → '{replacement}'"
                )

        return answer

    # ================================================================
    # RAG 降级
    # ================================================================

    def _rag_fallback(
        self, query: str, intent: str, entities: dict, precomputed_answer: str = None
    ) -> dict:
        """
        降级为 RAG 模式

        如果调用方已经通过 RAG pipeline 获得了答案，
        传入 precomputed_answer 避免重复调用。

        Args:
            query: 用户问题
            intent: 意图
            entities: 实体 (未使用)
            precomputed_answer: 预计算的 RAG 答案 (None 则实时调用)
        """
        logger.info(f"[Orchestrator] RAG 降级: query='{query[:50]}' intent={intent}")

        # 如果调用方已预计算答案，直接复用
        if precomputed_answer:
            logger.info("[Orchestrator] RAG 降级: 复用预计算结果")
            return {
                "answer": precomputed_answer,
                "route_mode": "rag_fallback",
                "agents_used": [],
                "sources": [],
                "latency_ms": 0,
            }

        if self.rag_system:
            try:
                rag_result = self.rag_system.process_query(query)
                answer = rag_result.get("answer", "抱歉，无法处理您的请求。")
                sources = rag_result.get("sources", [])
            except Exception as e:
                logger.error(f"[Orchestrator] RAG 降级失败: {e}")
                answer = "抱歉，当前服务不可用。请联系人工客服。"
                sources = []
        else:
            answer = (
                f"您好！关于「{query[:30]}…」，建议您:\n"
                "1. 查看产品条款中的相关说明\n"
                "2. 联系人工客服获取详细解答\n"
                "3. 拨打客服热线 400-XXX-XXXX"
            )
            sources = []

        return {
            "answer": answer,
            "route_mode": "rag_fallback",
            "agents_used": [],
            "sources": sources,
            "latency_ms": 0,
        }

    def _collect_sources(self, results: dict) -> list[dict]:
        """收集所有子 Agent 的条款引用（去重，最多 10 条）"""
        sources = []
        seen = set()
        for r in results.values():
            for s in r.sources:
                if not isinstance(s, dict):
                    continue
                key = s.get("clause_no", "") + s.get("text", "")[:80]
                if key not in seen:
                    sources.append(s)
                    seen.add(key)
        return sources[:10]
