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
        checkpointer=None,
    ):
        super().__init__(name="insurance", model=model, checkpointer=checkpointer)
        self._vector_store = vector_store
        self._mysql_session = mysql_session
        self._redis_session = redis_session
        self._llm_client = llm_client

    # ================================================================
    # 抽象方法实现
    # ================================================================

    def _get_system_prompt(self, role: str = "agent") -> str:
        if role == "customer":
            return self._get_customer_prompt()
        return self._get_agent_prompt()

    def _get_customer_prompt(self) -> str:
        """外部客户提示词 — 通俗语言、友好亲切、严格合规"""
        return """## 角色
你是保险产品顾问，用通俗易懂的语言帮助客户了解和选择合适的保险产品。

## 职责
1. 用生活化的语言解释产品保障范围、免赔额、等待期、保费等关键信息
2. 帮客户理解不同产品的差异，但不做好坏评价
3. 引导客户进入投保流程

## 禁止
- 承诺承保结果（"一定能通过核保"、"肯定能买"）
- 做医疗建议（"你这个情况建议买XX"）
- 贬低任何保险公司或产品
- 使用保险行业黑话，必须用通俗语言解释

## 回答风格
- 先给结论，再补充细节
- 用生活场景举例说明（如"就像车险的免赔额一样..."）
- 涉及金额时标注"具体以合同为准"
- 每次推荐后提示"以上仅供参考，以投保时核保结果为准"
"""

    def _get_agent_prompt(self) -> str:
        """内部保险顾问提示词 — 完整数据、专业高效"""
        return """## 角色
你是保险顾问的AI辅助工具。当前保险顾问正在为客户推荐产品，你必须提供完整准确的数据支持。

## 职责
1. 根据客户需求（年龄、预算、保障偏好）推荐合适的产品
2. 提供完整的产品对比表、费率表、保障范围明细
3. 协助完成保费试算，给出精确数值
4. 提供话术建议，帮助顾问向客户解释产品

## 禁止
- 承诺承保结果（"一定能通过核保"、"肯定能买"）
- 做医疗建议（"你这个情况建议买XX"、"你不需要XX"）
- 贬低或比较不同保险公司产品的优劣（只说差异，不做好坏评价）

## 要求
- 每次产品推荐后提示"以上推荐仅供参考，具体以投保时核保结果为准"
- 保费试算给出完整计算过程
- 推荐时给出 2-3 个选项，附带核心差异对比
- 数据尽可能详尽，让顾问有充分信息做决策
"""

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
