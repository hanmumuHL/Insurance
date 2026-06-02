# -*- coding: utf-8 -*-
"""
快速测试 — 验证 DeepSeek API 连通性

运行方式:
    cd /home/newnew/code/code/pythonCode/Insurance
    python tests/test_llm_client.py
"""

import sys
import os

# 添加项目根目录到 path，使 import 生效
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base.llm_client import get_llm_client


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
        print(f"❌ 失败: {response.error}")
        return False

    print(f"✅ 成功!")
    print(f"   Provider: {response.provider}")
    print(f"   Model:    {response.model}")
    print(f"   Tokens:   {response.tokens_used}")
    print(f"   Latency:  {response.latency_ms:.0f}ms")
    print(f"   Fallback: {response.is_fallback}")
    print(f"\n   回复:\n   {response.content}")
    return True


def test_json_output():
    """测试 2: JSON 模式 — 验证结构化输出"""
    print("\n" + "=" * 50)
    print("测试 2: JSON 结构化输出")
    print("=" * 50)

    client = get_llm_client()

    result = client.chat_json(
        messages=[
            {"role": "user", "content": '将以下问题分类为: 条款解读/理赔咨询/产品对比/保费试算/闲聊寒暄\n问题: "平安e生保和众安尊享e生哪个好"\n返回 JSON: {"intent": "...", "confidence": 0.xx}'},
        ],
        temperature=0,
    )

    if "error" in result:
        print(f"❌ 失败: {result['error']}")
        return False

    print(f"✅ 成功!")
    print(f"   解析结果: {result}")
    return True


def test_planning():
    """测试 3: Agent 规划 — 验证 Planner 输出"""
    print("\n" + "=" * 50)
    print("测试 3: Agent 规划能力")
    print("=" * 50)

    client = get_llm_client()

    result = client.chat_json(
        messages=[
            {"role": "user", "content": """你是一个保险客服的任务规划器。
用户问题: 我的平安e生保上个月肺炎住院了能赔吗
意图: 理赔咨询
实体: {"insurer": "平安", "product_name": "e生保"}

可用工具:
- policy_query: 查询保单信息
- clause_search: 检索条款
- claim_eligibility: 理赔资格预检

返回 JSON: {"tool_calls": [{"tool_name": "...", "tool_args": {}, "depends_on": []}], "reasoning": "..."}"""},
        ],
        temperature=0.1,
    )

    if "error" in result:
        print(f"❌ 失败: {result['error']}")
        return False

    tool_calls = result.get("tool_calls", [])
    print(f"✅ 成功! 规划了 {len(tool_calls)} 个工具调用:")
    for i, tc in enumerate(tool_calls):
        print(f"   [{i}] {tc.get('tool_name', '?')}({tc.get('tool_args', {})})")
    print(f"   推理: {result.get('reasoning', '')}")
    return True


if __name__ == "__main__":
    print("🚀 DeepSeek API 连通性测试\n")

    results = []
    results.append(("基本聊天", test_basic_chat()))
    results.append(("JSON 输出", test_json_output()))
    results.append(("Agent 规划", test_planning()))

    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")

    all_passed = all(r[1] for r in results)
    print(f"\n{'🎉 全部通过!' if all_passed else '⚠️ 部分测试失败'}")
