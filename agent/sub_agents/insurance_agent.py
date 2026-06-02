# -*- coding: utf-8 -*-
"""
投保 Agent — 产品推荐、保费试算、投保流程引导

职责边界:
  - ✅ 推荐产品、解释保障范围、试算保费、介绍投保流程
  - ❌ 不做核保结论（那是 UnderwritingAgent 的工作）
  - ❌ 不做理赔判断（那是 ClaimAgent 的工作）
  - ❌ 不处理投诉（直接转人工）

模型选择: deepseek-v3
  投保推荐不需要强推理，主要是信息检索和对比展示。
  V3 速度更快 (延迟更低)，适合交互式推荐场景。
"""

from agent.sub_agents.base import SubAgent
from agent.tools.all_tools import get_tools_by_domain


class InsuranceAgent(SubAgent):
    """
    投保领域 Agent

    典型对话:
      用户: "30岁想买医疗险，预算3000-5000，有什么推荐"
      → product_compare: 筛选医疗险 + 保费试算
      → 返回推荐列表 + 保障范围对比表

      用户: "尊享e生2024免赔额多少"
      → clause_search: 检索尊享e生免赔额条款
      → 返回具体条款原文
    """

    def __init__(
        self,
        vector_store=None,
        mysql_session=None,
        redis_session=None,
        llm_client=None,
        model="deepseek-v3",
    ):
        super().__init__(name="insurance", model=model)
        self._vector_store = vector_store
        self._mysql_session = mysql_session
        self._redis_session = redis_session
        self._llm_client = llm_client

    # ================================================================
    # 抽象方法实现
    # ================================================================

    def _get_system_prompt(self) -> str:
        return """## 角色
你是保险投保顾问，帮助用户选择最合适的保险产品。

## 职责
1. 根据用户需求（年龄、预算、保障偏好）推荐合适的产品
2. 解释产品的保障范围、免赔额、等待期、保费等关键信息
3. 协助用户完成保费试算
4. 引导用户进入投保流程（健康告知 → 核保 → 支付）

## 禁止
- 承诺承保结果（"一定能通过核保"、"肯定能买"）
- 做医疗建议（"你这个情况建议买XX"、"你不需要XX"）
- 贬低或比较不同保险公司产品的优劣（只说差异，不做好坏评价）
- 推荐用户不了解的产品（必须先确认需求再推荐）

## 要求
- 每次产品推荐后提示"以上推荐仅供参考，具体以投保时核保结果为准"
- 保费试算结果后提示"实际保费以最终核保为准"
- 提及健康告知是投保的必要环节
- 推荐时给出 2-3 个选项，附带核心差异对比"""

    def _get_tools(self) -> list:
        return get_tools_by_domain(
            "insurance",
            vector_store=self._vector_store,
            mysql_session=self._mysql_session,
            redis_session=self._redis_session,
            llm_client=self._llm_client,
        )

    def _get_compliance_rules(self) -> dict:
        return {
            "forbidden_phrases": [
                "保证承保",
                "一定能过",
                "肯定能买",
                "肯定能通过",
                "包赔",
                "必赔",
                "无风险",
                "稳赚",
                "绝对划算",
                "最好的",
                "第一名",
                "最强",
            ],
            "required_disclaimers": [
                "⚠️ 以上推荐仅供参考，实际承保结果以核保审核为准。",
                "⚠️ 投保前请仔细阅读保险条款和健康告知，确保理解保障内容和免责条款。",
            ],
            "force_handoff_triggers": [
                "投诉",
                "举报",
                "退保纠纷",
                "理赔纠纷",
                "我要告",
            ],
        }
