# -*- coding: utf-8 -*-
"""多 Agent 子 Agent 模块

统一接口:
  SubAgent.invoke(task: dict) → SubAgentResult

所有领域 Agent 通过此模块注册。
"""

from agent.sub_agents.base import SubAgent, SubAgentResult
from agent.sub_agents.insurance_agent import InsuranceAgent
from agent.sub_agents.underwriting_agent import UnderwritingAgent
from agent.sub_agents.claim_agent import ClaimAgent
from agent.sub_agents.service_agent import ServiceAgent

__all__ = [
    "SubAgent",
    "SubAgentResult",
    "InsuranceAgent",
    "UnderwritingAgent",
    "ClaimAgent",
    "ServiceAgent",
]
