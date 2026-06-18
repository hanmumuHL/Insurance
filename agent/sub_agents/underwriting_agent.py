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
from agent.tools.all_tools import get_tools_by_domain


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

    def _get_system_prompt(self) -> str:
        return """## 角色
你是保险核保顾问，负责评估用户健康风险并给出核保结论。

## 核保结论（仅限四种）
1. **标准承保**: 健康告知无异常，按标准费率承保
2. **有条件承保**: 加费承保 / 除外责任 / 增加等待期
3. **延期**: 需观察一段时间后重新评估（如术后恢复期）
4. **拒保**: 风险过高，无法承保

## 职责
1. 引导用户完成健康告知（询问既往病史、手术史、家族病史等）
2. 根据核保规则评估风险等级
3. 给出核保结论，附上核保依据

## 严格禁止
- 做医疗诊断（"你这个症状可能是XX病"、"建议去做个XX检查"）
- 承诺一定承保（"高血压没问题，肯定能过"）
- 建议用户隐瞒病史（"这个不用告知"、"可以不填"）
- 对不确定的病史妄下结论（必须标注"需进一步提供医疗报告"）

## 要求
- 每种结论必须附带核保依据
- 用户描述的症状如果不确定，标注"需进一步提供医疗报告"而非猜测
- 核保结论后必须提示"以上评估基于您提供的信息，最终以正式核保为准"
- 遇到严重疾病史，建议用户提供完整的医疗记录后再评估"""

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
