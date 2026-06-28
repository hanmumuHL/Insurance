# -*- coding: utf-8 -*-
"""
Agent 工具集 — 7 个 LangChain Tool

每个工具继承 LangChain 的 BaseTool，分布在各独立文件中:
  1. policy_query       — 保单查询        (policy_query.py)
  2. claim_eligibility  — 理赔资格预检    (claim_eligibility.py)
  3. clause_search      — 条款检索        (clause_search.py)
  4. premium_calc       — 保费试算        (premium_calc.py)
  5. product_compare    — 多产品对比      (product_compare.py)
  6. claim_tracking     — 理赔进度追踪    (claim_tracking.py)
  7. human_handoff      — 人工转接        (human_handoff.py)

注册表:
  - get_all_tools():      全部工具 + 依赖注入
  - get_tools_by_domain(): 按领域+角色过滤（双通道核心函数）
  - TOOL_ROLE_MAP / TOOL_DOMAIN_MAP: 权限与领域映射
"""

# ── 工具类 ──
from agent.tools.policy_query import PolicyQueryTool, PolicyQueryInput
from agent.tools.claim_eligibility import ClaimEligibilityTool, ClaimEligibilityInput
from agent.tools.clause_search import ClauseSearchTool, ClauseSearchInput
from agent.tools.premium_calc import PremiumCalcTool, PremiumCalcInput
from agent.tools.product_compare import ProductCompareTool, ProductCompareInput
from agent.tools.claim_tracking import ClaimTrackingTool, ClaimTrackingInput
from agent.tools.human_handoff import HumanHandoffTool, HumanHandoffInput

# ── 注册表 ──
from agent.tools.registry import (
    get_all_tools,
    get_tool_by_name,
    get_tools_by_domain,
    TOOL_ROLE_MAP,
    TOOL_DOMAIN_MAP,
)

# ── 工具函数 ──
from agent.tools._escape import escape_like_pattern

__all__ = [
    # 工具类
    "PolicyQueryTool", "PolicyQueryInput",
    "ClaimEligibilityTool", "ClaimEligibilityInput",
    "ClauseSearchTool", "ClauseSearchInput",
    "PremiumCalcTool", "PremiumCalcInput",
    "ProductCompareTool", "ProductCompareInput",
    "ClaimTrackingTool", "ClaimTrackingInput",
    "HumanHandoffTool", "HumanHandoffInput",
    # 注册表
    "get_all_tools",
    "get_tool_by_name",
    "get_tools_by_domain",
    "TOOL_ROLE_MAP",
    "TOOL_DOMAIN_MAP",
    # 工具函数
    "escape_like_pattern",
]
