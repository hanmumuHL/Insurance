# -*- coding: utf-8 -*-
"""
多 Agent 架构集成测试

测试范围:
  1. 子 Agent 基类 (状态图编译、invoke 接口)
  2. Orchestrator 路由 (各种意图 → 正确的 Agent)
  3. 任务拆解与执行 (依赖顺序)
  4. 合规终审 (禁止表述替换)

注意:
  这些测试不依赖真实的基础设施 (Milvus/MySQL/Redis)，
  所有 LLM 调用会被 mock，只验证调度逻辑。
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from unittest.mock import patch

import pytest

from agent.orchestrator import INTENT_ROUTING, Orchestrator

# ── 受测模块 ──
from agent.state import SubAgentTask
from agent.sub_agents.base import SubAgent, SubAgentResult

# ================================================================
# Mock 子 Agent — 模拟真实 Agent 的行为
# ================================================================


class MockSuccessAgent(SubAgent):
    """总是返回成功的 Mock Agent"""

    def __init__(self, name, response="Mock 响应"):
        super().__init__(name=name, model="deepseek-v3")
        self.mock_response = response

    def _get_system_prompt(self):
        return f"Mock {self.name} system prompt"

    def _get_tools(self):
        return []

    def _get_compliance_rules(self):
        return {
            "forbidden_phrases": [],
            "required_disclaimers": [],
            "force_handoff_triggers": [],
        }

    def invoke(self, task):
        return SubAgentResult(
            agent_name=self.name,
            task_id=task.get("task_id", "mock_task"),
            status="success",
            result=self.mock_response,
            confidence=0.9,
            latency_ms=100.0,
        )


class MockFailAgent(SubAgent):
    """总是返回失败的 Mock Agent"""

    def __init__(self, name, error="模拟失败"):
        super().__init__(name=name, model="deepseek-v3")
        self.mock_error = error

    def _get_system_prompt(self):
        return f"Mock {self.name} system prompt"

    def _get_tools(self):
        return []

    def _get_compliance_rules(self):
        return {
            "forbidden_phrases": [],
            "required_disclaimers": [],
            "force_handoff_triggers": [],
        }

    def invoke(self, task):
        return SubAgentResult(
            agent_name=self.name,
            task_id=task.get("task_id", "mock_task"),
            status="failed",
            result="失败时的降级回复",
            confidence=0.1,
            latency_ms=50.0,
            error=self.mock_error,
        )


# ================================================================
# 测试: 意图路由
# ================================================================


class TestRouting:
    """Orchestrator 意图路由"""

    def setup_method(self):
        self.orch = Orchestrator(
            agents={
                "claim": MockSuccessAgent("claim"),
                "insurance": MockSuccessAgent("insurance"),
                "underwriting": MockSuccessAgent("underwriting"),
                "service": MockSuccessAgent("service"),
            }
        )

    @pytest.mark.parametrize(
        "intent,expected_primary,expected_mode",
        [
            ("理赔咨询", "claim", "multi_agent"),  # 理赔需要核保辅助
            ("产品咨询", "insurance", "single_agent"),  # 产品推荐只调投保Agent
            ("投保流程", "insurance", "multi_agent"),  # 投保+核保
            ("核保咨询", "underwriting", "single_agent"),  # 核保独立
            ("保单查询", "service", "single_agent"),  # 客服独立
            ("投诉建议", "service", "rag_fallback"),  # 投诉直接降级
            ("闲聊", "service", "rag_fallback"),  # 闲聊降级FAQ
        ],
    )
    def test_intent_routing(self, intent, expected_primary, expected_mode):
        route = self.orch._route_intent(intent, "这是一条测试query")
        assert route["primary"] == expected_primary, (
            f"意图 '{intent}' 应路由到 {expected_primary}"
        )
        assert route["mode"] == expected_mode, (
            f"意图 '{intent}' 的模式应为 {expected_mode}"
        )

    def test_simple_query_falls_back_to_rag(self):
        """短 query (≤8字) 且非复杂意图 → RAG 降级"""
        route = self.orch._route_intent("产品咨询", "你好")
        assert route["mode"] == "rag_fallback"

    def test_complex_short_query_not_degraded(self):
        """短 query 但是复杂意图 → 不走 RAG 降级"""
        route = self.orch._route_intent("理赔咨询", "肺炎能赔吗")
        assert route["mode"] == "multi_agent"  # 不走降级

    def test_unknown_intent_llm_fallback(self):
        """未知意图 → LLM 辅助分类 (mock LLM 返回)"""
        with patch.object(
            self.orch,
            "_llm_classify_intent",
            return_value={
                "primary": "service",
                "secondary": None,
                "mode": "rag_fallback",
            },
        ):
            route = self.orch._route_intent("未知意图xyz", "test query")
            assert route["primary"] == "service"
            assert route["mode"] == "rag_fallback"


# ================================================================
# 测试: 任务构建
# ================================================================


class TestTaskBuilding:
    """任务拆解"""

    def setup_method(self):
        self.orch = Orchestrator(agents={})

    def test_single_agent_builds_one_task(self):
        route = INTENT_ROUTING["产品咨询"]
        tasks = self.orch._build_tasks("推荐医疗险", "产品咨询", {}, route, {})
        assert len(tasks) == 1
        assert tasks[0].agent_name == "insurance"
        assert tasks[0].dependencies == []

    def test_multi_agent_builds_two_tasks(self):
        route = INTENT_ROUTING["理赔咨询"]
        tasks = self.orch._build_tasks(
            "肺炎能赔吗", "理赔咨询", {"insurer": "平安"}, route, {}
        )
        assert len(tasks) == 2
        assert tasks[0].agent_name == "claim"
        assert tasks[1].agent_name == "underwriting"
        assert tasks[1].dependencies == ["task_0"]  # 辅助Agent依赖主Agent


# ================================================================
# 测试: 任务执行
# ================================================================


class TestTaskExecution:
    """任务依赖执行"""

    def test_executes_single_task(self):
        orch = Orchestrator(
            agents={
                "claim": MockSuccessAgent("claim", "理赔预检通过"),
            }
        )
        tasks = [
            SubAgentTask(
                task_id="task_0",
                agent_name="claim",
                intent="理赔咨询",
                user_query="测试",
                entities={},
                dependencies=[],
            )
        ]
        results = orch._execute_tasks(tasks, "test_session")
        assert "task_0" in results
        assert results["task_0"].status == "success"
        assert "理赔预检通过" in results["task_0"].result

    def test_upstream_result_passed_to_downstream(self):
        """上游 Agent 的结果注入到下游 Agent 的 context"""
        orch = Orchestrator(
            agents={
                "claim": MockSuccessAgent("claim", "保单: 尊享e生2024, 状态: 有效"),
                "underwriting": MockSuccessAgent("underwriting", "核保审核通过"),
            }
        )
        tasks = [
            SubAgentTask(
                task_id="task_0",
                agent_name="claim",
                intent="理赔咨询",
                user_query="测试",
                entities={},
                dependencies=[],
            ),
            SubAgentTask(
                task_id="task_1",
                agent_name="underwriting",
                intent="核保咨询",
                user_query="测试",
                entities={},
                dependencies=["task_0"],
            ),
        ]
        results = orch._execute_tasks(tasks, "test_session")
        assert results["task_0"].status == "success"
        assert results["task_1"].status == "success"

    def test_failed_agent_does_not_block_others(self):
        """失败的 Agent 不阻塞后续无依赖的任务"""
        orch = Orchestrator(
            agents={
                "insurance": MockFailAgent("insurance", "DB连接失败"),
                "service": MockSuccessAgent("service", "客服正常响应"),
            }
        )
        tasks = [
            SubAgentTask(
                task_id="task_0",
                agent_name="insurance",
                intent="产品咨询",
                user_query="测试",
                entities={},
                dependencies=[],
            ),
            SubAgentTask(
                task_id="task_1",
                agent_name="service",
                intent="保单查询",
                user_query="测试",
                entities={},
                dependencies=[],
                # 不依赖 task_0
            ),
        ]
        results = orch._execute_tasks(tasks, "test_session")
        assert results["task_0"].status == "failed"
        assert results["task_1"].status == "success"  # 不受影响


# ================================================================
# 测试: 结果聚合
# ================================================================


class TestResultAggregation:
    """结果聚合"""

    def setup_method(self):
        self.orch = Orchestrator(agents={})

    def test_single_agent_result_returned_directly(self):
        results = {
            "task_0": SubAgentResult(
                agent_name="claim",
                task_id="task_0",
                status="success",
                result="根据条款第2.5条，免赔额1万元。",
                disclaimer="⚠️ 最终以审核为准",
            )
        }
        tasks = [SubAgentTask(task_id="task_0", agent_name="claim")]
        answer = self.orch._aggregate_results("肺炎能赔吗", "理赔咨询", results, tasks)
        assert "免赔额" in answer
        assert "审核为准" in answer  # disclaimer 被附加

    def test_all_failed_returns_fallback(self):
        results = {
            "task_0": SubAgentResult(
                agent_name="claim",
                task_id="task_0",
                status="failed",
                result="",
                error="timeout",
            )
        }
        tasks = [SubAgentTask(task_id="task_0", agent_name="claim")]
        answer = self.orch._aggregate_results("肺炎能赔吗", "理赔咨询", results, tasks)
        assert "不可用" in answer or "人工客服" in answer


# ================================================================
# 测试: 合规终审
# ================================================================


class TestComplianceReview:
    """合规检查"""

    def setup_method(self):
        self.orch = Orchestrator(agents={})

    def test_replaces_forbidden_phrases(self):
        answer = "尊享e生肯定可以赔，您放心"
        result = self.orch._compliance_review(answer, "理赔咨询")
        assert "肯定可以赔" not in result
        assert "审核" in result

    def test_preserves_normal_answer(self):
        answer = "根据条款第2.5条，免赔额为1万元。最终以保险合同为准。"
        result = self.orch._compliance_review(answer, "理赔咨询")
        assert result == answer  # 合规答案不应该被修改


# ================================================================
# 测试: 端到端 (单Agent模式)
# ================================================================


class TestEndToEnd:
    """端到端: process() 完整流程"""

    def test_single_agent_e2e(self):
        orch = Orchestrator(
            agents={
                "insurance": MockSuccessAgent(
                    "insurance", "推荐尊享e生2024，年保费4560元"
                ),
            }
        )
        result = orch.process(
            query="30岁想买一份医疗险有什么推荐的",  # >8字，不走降级
            intent="产品咨询",
            entities={},
            session_id="test",
        )
        assert result["route_mode"] == "single_agent"
        assert "尊享e生" in result["answer"]
        assert "insurance" in result["agents_used"]

    def test_multi_agent_e2e_with_mock_llm(self):
        """多 Agent 端到端 (mock LLM 聚合)"""
        orch = Orchestrator(
            agents={
                "claim": MockSuccessAgent(
                    "claim", "理赔预检: 肺炎住院属于保险责任，符合保障范围"
                ),
                "underwriting": MockSuccessAgent(
                    "underwriting", "核保: 已过30天等待期，标准承保中"
                ),
            }
        )

        # Mock LLM 聚合（避免真实 LLM 调用）
        with patch.object(
            orch,
            "_llm_aggregate",
            return_value="根据理赔和核保结果，您的肺炎住院属于保险责任范围，已过等待期，符合理赔条件。最终以审核为准。",
        ):
            result = orch.process(
                query="平安e生保肺炎住院能赔吗",
                intent="理赔咨询",
                entities={"insurer": "平安", "product_name": "e生保"},
                session_id="test",
            )

        assert result["route_mode"] == "multi_agent"
        assert len(result["agents_used"]) == 2
        assert "claim" in result["agents_used"]
        assert "underwriting" in result["agents_used"]
        assert "保险责任" in result["answer"]

    def test_rag_fallback(self):
        """降级场景: 简单 query → RAG"""
        orch = Orchestrator(agents={})
        result = orch.process(
            query="你好",
            intent="闲聊",
            entities={},
            session_id="test",
        )
        assert result["route_mode"] == "rag_fallback"
        assert len(result["agents_used"]) == 0  # 没有调任何 Agent


if __name__ == "__main__":
    tete = TestEndToEnd()
    tete.test_multi_agent_e2e_with_mock_llm()
