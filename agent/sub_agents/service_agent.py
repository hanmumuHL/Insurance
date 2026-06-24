# -*- coding: utf-8 -*-
"""
客服 Agent — 保单查询、FAQ、转人工

职责边界:
  - ✅ 查询保单信息（状态/保额/有效期）
  - ✅ 回答常见问题（FAQ）
  - ✅ 在必要时转接人工客服
  - ❌ 不做产品推荐（那是 InsuranceAgent 的工作）
  - ❌ 不做理赔判断（那是 ClaimAgent 的工作）
  - ❌ 不处理投诉（直接转人工）

模型选择: deepseek-v3
  客服场景以信息查询和 FAQ 匹配为主，不需要强推理。
  V3 速度快，适合高频、低延迟的客服交互。
"""

from agent.sub_agents.base import SubAgent
from agent.tools.all_tools import get_tools_by_domain


class ServiceAgent(SubAgent):
    """
    客服领域 Agent

    典型对话:
      用户: "帮我查一下我的保单"
      → policy_query: 查保单状态
      → 返回保单详情

      用户: "退保怎么操作"
      → FAQ 检索 + 流程说明
      → 如果需要具体操作 → human_handoff 转人工

      用户: "我要投诉"
      → 直接 human_handoff 转人工
    """

    def __init__(self, vector_store=None, mysql_session=None, redis_session=None,
                 llm_client=None, model="deepseek-v3", checkpointer=None):
        super().__init__(name="service", model=model, checkpointer=checkpointer)
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
        """外部客户提示词 — 简洁友好、快速解决问题"""
        return """## 角色
你是保险客服助手，帮助客户处理保单查询和日常问题。

## 职责
1. 查询保单基本信息（保单号、状态、保额、有效期）
2. 回答常见问题（退保流程、续保方式、信息变更等）
3. 遇到复杂问题时引导客户联系人工客服

## 禁止
- 擅自修改保单信息
- 对理赔结果做出判断
- 对产品优劣做出评价
- 透露其他客户的信息

## 转人工场景
遇到以下情况应引导客户联系人工客服:
- 需要修改保单信息
- 涉及退保金额计算
- 投诉、举报、纠纷
- 客户明确要求转人工

## 回答风格
- 简洁直接，不要长篇大论
- 先给出结论，再补充细节
- 涉及数字时标注单位和来源
- 结尾提示"如有其他问题请随时咨询"
"""

    def _get_agent_prompt(self) -> str:
        """内部客服人员提示词 — 全量客户信息、高效处理"""
        return """## 角色
你是客服人员的AI辅助工具。当前客服人员正在处理客户事务，请提供完整准确的信息。

## 职责
1. 查询保单基本信息（保单号、状态、保额、有效期、保费）
2. 获取客户全量信息（保单列表、理赔记录、互动历史）
3. 回答常见FAQ（退保流程、续保方式、信息变更等）
4. 记录工单并转接人工客服

## 禁止
- 擅自修改保单信息
- 承诺未经验证的保单变更结果
- 对理赔结果做出判断
- 对产品优劣做出评价

## 转人工规则
遇到以下情况应立即转人工:
- 投诉、举报、纠纷
- 需要修改保单信息（需身份验证）
- 涉及退保金额计算（需人工核算）
- 客户连续3轮表达不满

## 回答风格
- 简洁直接，信息完整
- 返回客户全量数据供客服人员参考
- 涉及数字时标注单位和来源
"""

    def _get_tools(self) -> list:
        return get_tools_by_domain(
            "service",
            vector_store=self._vector_store,
            mysql_session=self._mysql_session,
            redis_session=self._redis_session,
            llm_client=self._llm_client,
        )

    def _get_compliance_rules(self) -> dict:
        return {
            "forbidden_phrases": [
                "你的密码是", "你的身份证号是", "你的银行卡号是",  # 不应暴露敏感信息
            ],
            "required_disclaimers": [
                "⚠️ 如需修改保单信息，请通过官方渠道或联系人工客服核实身份后办理。",
            ],
            "force_handoff_triggers": [
                "投诉", "举报", "退保", "我要告",
                "经理", "领导", "上级", "监管部门",
                "不满意", "太差了", "什么破",
            ],
        }
