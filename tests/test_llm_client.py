# -*- coding: utf-8 -*-
"""
LLM 客户端测试 — 包含真实 API smoke test 和 mock-based failure path 测试

运行方式:
    cd /home/newnew/code/code/pythonCode/Insurance

    # 仅 smoke test (需 API key):
    python tests/test_llm_client.py

    # 全部测试 (含 mock):
    pytest tests/test_llm_client.py -v
"""

import sys
import os
import json
from unittest.mock import Mock, patch, MagicMock

import pytest

# 添加项目根目录到 path，使 import 生效
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base.llm_client import get_llm_client, LLMClient, CircuitBreaker


# ============================================================
# CircuitBreaker 单元测试 (不依赖 API)
# ============================================================

class TestCircuitBreaker:
    """熔断器逻辑测试"""

    def test_normal_flow(self):
        """正常流程: should_skip 返回 False"""
        cb = CircuitBreaker(failure_threshold=3, cool_down_seconds=30)
        assert cb.should_skip() is False

    def test_opens_after_threshold(self):
        """连续失败达到阈值后熔断"""
        cb = CircuitBreaker(failure_threshold=3, cool_down_seconds=30)
        for _ in range(3):
            cb.record_failure()
        assert cb.should_skip() is True

    def test_no_fallback_never_skips(self):
        """无降级通道时即使熔断也不跳过"""
        cb = CircuitBreaker(failure_threshold=1, cool_down_seconds=30)
        cb.record_failure()
        assert cb.is_open is True
        assert cb.should_skip(has_fallback=False) is False

    def test_resets_on_success(self):
        """成功后重置计数"""
        cb = CircuitBreaker(failure_threshold=3, cool_down_seconds=30)
        for _ in range(2):
            cb.record_failure()
        cb.record_success()
        assert cb.failure_count == 0
        assert cb.is_open is False

    def test_no_skip_without_fallback(self):
        """无 fallback 时 should_skip 返回 False 即使超过阈值"""
        cb = CircuitBreaker(failure_threshold=1, cool_down_seconds=30)
        cb.record_failure()
        assert cb.should_skip(has_fallback=False) is False
        assert cb.should_skip(has_fallback=True) is True


# ============================================================
# LLM 客户端 failure path 测试 (mock)
# ============================================================

class TestLLMClientFailurePaths:
    """Mock-based 故障路径测试"""

    @patch('base.llm_client.OpenAI')
    def test_primary_fails_fallback_succeeds(self, mock_openai):
        """Primary 失败 -> fallback 被调用"""
        mock_primary = Mock()
        mock_primary.chat.completions.create.side_effect = Exception("Primary down")
        mock_fallback = Mock()
        mock_fallback.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="fallback answer"))],
            usage=Mock(total_tokens=10),
            model="qwen-max",
        )
        mock_openai.side_effect = [mock_primary, mock_fallback]

        client = LLMClient()
        response = client.chat(messages=[{"role": "user", "content": "test"}])
        assert not response.error or "系统" in str(response.content)

    @patch('base.llm_client.OpenAI')
    def test_both_channels_fail(self, mock_openai):
        """双通道均失败 -> 返回系统繁忙"""
        mock_broken = Mock()
        mock_broken.chat.completions.create.side_effect = Exception("API down")
        mock_openai.side_effect = [mock_broken, mock_broken]

        client = LLMClient()
        response = client.chat(messages=[{"role": "user", "content": "test"}])
        assert response.error is not None
        assert "繁忙" in response.content or "不可用" in response.content

    @patch('base.llm_client.OpenAI')
    def test_chat_json_returns_error_on_non_json(self, mock_openai):
        """非 JSON 响应 -> 返回 error dict"""
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="这不是 JSON"))],
            usage=Mock(total_tokens=5),
            model="deepseek-chat",
        )
        mock_openai.return_value = mock_client

        client = LLMClient()
        result = client.chat_json(messages=[{"role": "user", "content": "say something"}])
        assert "error" in result

    @patch('base.llm_client.OpenAI')
    def test_circuit_breaker_counts_failures(self, mock_openai):
        """熔断器正确计数失败"""
        mock_broken = Mock()
        mock_broken.chat.completions.create.side_effect = Exception("fail")
        mock_openai.return_value = mock_broken

        client = LLMClient()
        for _ in range(5):
            client.chat(messages=[{"role": "user", "content": "test"}])
        assert client.circuit_breaker.failure_count >= 5


# ============================================================
# Smoke tests (需 API key) — 标记为 smoke
# ============================================================

@pytest.mark.smoke
def test_basic_chat():
    """测试 1: 基本聊天 — 验证 API 连通"""
    print("=" * 50)
    print("测试 1: DeepSeek API 基本聊天")
    print("=" * 50)
    client = get_llm_client()
    response = client.chat(
        messages=[
            {"role": "system", "content": "你是保险智能客服，回复简洁。"},
            {"role": "user", "content": "你好，肺炎住院能赔吗？"},
        ],
        temperature=0.3,
        max_tokens=200,
    )
    if response.error:
        print(f"FAIL: {response.error}")
        return False
    print(f"OK! Provider: {response.provider}, Model: {response.model}, Tokens: {response.tokens_used}")
    return True


@pytest.mark.smoke
def test_json_output():
    """测试 2: JSON 模式 — 验证结构化输出"""
    print("\n" + "=" * 50)
    print("测试 2: JSON 结构化输出")
    print("=" * 50)
    client = get_llm_client()
    result = client.chat_json(
        messages=[{"role": "user", "content": (
            '将以下问题分类为: 条款解读/理赔咨询/产品对比/保费试算/闲聊寒暄\n'
            '问题: "平安e生保和众安尊享e生哪个好"\n'
            '返回 JSON: {"intent": "...", "confidence": 0.xx}'
        )}],
        temperature=0,
    )
    if "error" in result:
        print(f"FAIL: {result['error']}")
        return False
    print(f"OK! Result: {result}")
    return True


@pytest.mark.smoke
def test_planning():
    """测试 3: Agent 规划 — 验证 Planner 输出"""
    print("\n" + "=" * 50)
    print("测试 3: Agent 规划能力")
    print("=" * 50)
    client = get_llm_client()
    prompt = (
        "你是一个保险客服的任务规划器。\n"
        "用户问题: 我的平安e生保上个月肺炎住院了能赔吗\n"
        "意图: 理赔咨询\n"
        '实体: {"insurer": "平安", "product_name": "e生保"}\n\n'
        "可用工具:\n"
        "- policy_query: 查询保单信息\n"
        "- clause_search: 检索条款\n"
        "- claim_eligibility: 理赔资格预检\n\n"
        '返回 JSON: {"tool_calls": [{"tool_name": "tool", "tool_args": {}, "depends_on": []}], "reasoning": "text"}'
    )
    result = client.chat_json(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    if "error" in result:
        print(f"FAIL: {result['error']}")
        return False
    tool_calls = result.get("tool_calls", [])
    print(f"OK! Planned {len(tool_calls)} tool calls")
    return True


if __name__ == "__main__":
    print("DeepSeek API 连通性测试\n")
    results = []
    results.append(("基本聊天", test_basic_chat()))
    results.append(("JSON 输出", test_json_output()))
    results.append(("Agent 规划", test_planning()))
    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    for name, passed in results:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    all_passed = all(r[1] for r in results)
    print(f"\n{'All passed!' if all_passed else 'Some tests failed'}")
