# -*- coding: utf-8 -*-
"""
核保 Agent — 健康告知审核、风险评估、核保结论

职责边界:
  - ✅ 引导健康告知、评估承保风险、给出核保结论
  - ❌ 不做产品推荐（那是 InsuranceAgent 的工作）
  - ❌ 不做医疗诊断
  - ❌ 不能建议用户隐瞒病史

模型选择: deepseek-r1
  核保需要多步推理: 健康告知 → 匹配核保规则 → 风险评估 → 结论。
  R1 的思维链能更好地处理 "多个条件组合判断" 的场景。
"""

from agent.sub_agents.base import SubAgent
from agent.tools import get_tools_by_domain


class UnderwritingAgent(SubAgent):
    """
    核保领域 Agent

    典型对话:
      用户: "有高血压能买医疗险吗"
      → clause_search: 检索核保规则中关于高血压的规定
      → 规则引擎推理: 高血压一级 → 可能条件承保，二级以上 → 可能拒保
      → 返回: "高血压一级一般可条件承保（加费或除外），具体需提供近半年血压记录"

      用户: "去年做过甲状腺结节手术"
      → 需要更多信息: 良性/恶性? 术后多久? 有无复发?
      → 标注"需进一步提供医疗报告"
    """

    def __init__(self, vector_store=None, mysql_session=None, redis_session=None,
                 llm_client=None, model="deepseek-r1", checkpointer=None):
        super().__init__(name="underwriting", model=model, checkpointer=checkpointer)
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
        """外部客户提示词 — 引导正确填写健康告知、不替代核保"""
        return """## 角色
你是健康告知辅助工具，帮助客户正确理解和填写健康告知。

## 核保结论说明（仅四种，告知客户含义）
1. **标准承保**: 健康情况良好，按正常价格投保
2. **有条件承保**: 可能需要加费或不保某些疾病
3. **延期**: 需要等一段时间再评估（比如手术后恢复期）
4. **拒保**: 目前情况不适合投保该产品

## 职责
1. 帮客户理解健康告知里的医学名词
2. 告知客户哪些情况可能影响核保结果
3. 解释四种核保结论的含义

## 严格禁止
- 做医疗诊断（"你这个症状可能是XX病"）
- 承诺一定承保（"高血压没事，肯定能过"）
- 建议客户隐瞒病史（"这个不用说"、"可以不填"）
- 对不确定的病史妄下结论

## 回答风格
- 用通俗语言解释医学术语
- 安抚客户对核保的焦虑
- 必须提示"以上评估仅供参考，最终以正式核保为准"
- 鼓励如实告知，解释隐瞒病史的后果
"""

    def _get_agent_prompt(self) -> str:
        """内部核保人员提示词 — 完整核保规则、精准风险评估"""
        return """## 角色
你是核保人员的AI辅助工具。当前核保人员正在评估客户的健康风险，请提供完整的核保规则和评估建议。

## 核保结论（仅限四种）
1. **标准承保**: 健康告知无异常，按标准费率承保
2. **有条件承保**: 加费承保 / 除外责任 / 增加等待期
3. **延期**: 需观察一段时间后重新评估（如术后恢复期）
4. **拒保**: 风险过高，无法承保

## 职责
1. 根据客户健康告知匹配核保规则
2. 提供详细的核保规则原文和判例参考
3. 评估风险等级并给出核保建议

## 严格禁止
- 做医疗诊断
- 承诺一定承保
- 建议客户隐瞒病史
- 对不确定的病史妄下结论

## 要求
- 每种结论必须附带核保规则依据
- 标注核保规则的适用范围和例外情况
- 对不确定的情况标注"需进一步提供医疗报告"
- 核保结论后提示"以上评估基于提供的信息，最终以正式核保为准"
"""

    def _get_tools(self) -> list:
        return get_tools_by_domain(
            "underwriting",
            vector_store=self._vector_store,
            mysql_session=self._mysql_session,
            redis_session=self._redis_session,
            llm_client=self._llm_client,
        )

    def _get_compliance_rules(self) -> dict:
        return {
            "forbidden_phrases": [
                "你的病不严重", "没什么大问题", "小毛病",
                "不用告知", "可以不填", "可以不写", "不用写",
                "这个病史不影响", "这个不用管",
                "肯定能通过", "保证承保",
            ],
            "required_disclaimers": [
                "⚠️ 以上核保结论基于您提供的信息，最终以正式核保结果为准。",
                "⚠️ 投保时请如实告知健康状况。隐瞒病史可能导致未来理赔被拒赔。",
            ],
            "force_handoff_triggers": [
                "投诉", "拒保申诉", "核保争议", "不公平",
            ],
        }
