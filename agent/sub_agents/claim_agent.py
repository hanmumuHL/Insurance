# -*- coding: utf-8 -*-
"""
理赔 Agent — 理赔资格判断、条款检索、赔付计算

职责边界:
  - ✅ 判断疾病/事件是否在保障范围内
  - ✅ 检索相关条款（保险责任 + 责任免除）
  - ✅ 估算赔付金额
  - ✅ 追踪理赔进度
  - ❌ 不做最终理赔决定（那是保司的工作）
  - ❌ 不承诺一定能赔付

模型选择: deepseek-r1
  理赔判断是 Agent 中最复杂的场景——需要多步推理:
  查保单 → 查条款 → 等待期计算 → 免责检查 → 赔付计算。
  R1 的思维链在这个场景下准确率显著高于 V3。
"""

from agent.sub_agents.base import SubAgent
from agent.tools.all_tools import get_tools_by_domain


class ClaimAgent(SubAgent):
    """
    理赔领域 Agent

    典型对话:
      用户: "平安e生保肺炎住院能赔吗"
      → policy_query: 查用户的平安e生保保单
      → clause_search: 检索保险责任 + 责任免除中关于肺炎/住院的规定
      → claim_eligibility: 预检理赔资格
      → 返回: "根据条款第2.3条，肺炎住院属于保险责任范围。
               但需确认: 1) 是否已过30天等待期 2) 是否在保障期限内"

      用户: "上周流感住院花了5000，能赔多少"
      → policy_query → clause_search → premium_calc(试算赔付)
      → 返回: "根据条款第2.5条免赔额1万元，本次5000元未超过免赔额，无法赔付。
               但年度累计超过1万元后可按100%赔付。"
    """

    def __init__(self, vector_store=None, mysql_session=None, redis_session=None,
                 llm_client=None, model="deepseek-r1"):
        super().__init__(name="claim", model=model)
        self._vector_store = vector_store
        self._mysql_session = mysql_session
        self._redis_session = redis_session
        self._llm_client = llm_client

    # ================================================================
    # 抽象方法实现
    # ================================================================

    def _get_system_prompt(self) -> str:
        return """## 角色
你是保险理赔顾问，帮助用户解答理赔相关的问题。

## 职责
1. 根据用户描述的事件，判断是否在保险产品的保障范围内
2. 检索对应产品的保险责任条款和责任免除条款
3. 估算可能的赔付金额
4. 说明理赔流程和所需材料
5. 追踪已有理赔申请的进度

## 严格禁止
- 承诺一定能赔付（"肯定能赔"、"100%能理赔"）
- 做最终理赔结论（这是保险公司的职责，你只能做预检）
- 建议用户伪造理赔材料
- 对条款做超出原文的解释

## 理赔判断流程
1. 先确认用户购买了哪个产品 (policy_query)
2. 检索该产品的保险责任条款 (clause_search, clause_type="保险责任")
3. 检索该产品的责任免除条款 (clause_search, clause_type="责任免除")
4. 判断事件是否在保险责任中且不在责任免除中 (claim_eligibility)
5. 如果符合，估算赔付金额 (premium_calc)
6. 说明理赔流程和所需材料

## 要求
- 每个判断必须引用具体条款编号和原文
- 赔付金额估算必须标注"仅供参考，以最终审核为准"
- 如果信息不足以判断，明确告知用户需要补充什么信息
- 涉及金额时标注币种和计算方式"""

    def _get_tools(self) -> list:
        return get_tools_by_domain(
            "claim",
            vector_store=self._vector_store,
            mysql_session=self._mysql_session,
            redis_session=self._redis_session,
            llm_client=self._llm_client,
        )

    def _get_compliance_rules(self) -> dict:
        return {
            "forbidden_phrases": [
                "一定能赔", "肯定可以赔", "100%能理赔", "绝对能赔",
                "没问题肯定赔", "这个肯定在保障范围内",
                "你们保险公司", "你们不赔我就", "骗人的",
            ],
            "required_disclaimers": [
                "⚠️ 以上为理赔预检结果，最终理赔结论以保险公司正式审核为准。",
                "⚠️ 赔付金额为估算值，实际金额以理赔审核结果为准。",
                "⚠️ 请确保理赔申请时提供真实完整的材料。",
            ],
            "force_handoff_triggers": [
                "投诉", "拒赔申诉", "理赔纠纷", "我要告",
                "律师函", "保监会", "银保监",
            ],
        }
