# -*- coding: utf-8 -*-
"""
工具注册表 — 依赖注入 + 领域角色映射

提供:
  - get_all_tools(): 返回全部 7 个已注入依赖的工具
  - get_tool_by_name(): 按名称查找单个工具
  - get_tools_by_domain(): 按业务领域 + 角色返回工具集（双通道核心函数）
"""

from typing import Optional
from langchain_core.tools import BaseTool

from base.logger import logger

# ── 工具 → 角色映射表（双通道权限隔离）──
# 外部客户 (customer): 只能查看自己的数据，不可使用内部工具
# 内部人员 (agent/underwriter): 可使用全部工具
TOOL_ROLE_MAP = {
    "policy_query":       ["customer", "agent", "underwriter"],
    "claim_eligibility":  ["agent", "underwriter"],
    "clause_search":      ["customer", "agent", "underwriter"],
    "premium_calc":       ["agent", "underwriter"],
    "product_compare":    ["customer", "agent"],
    "claim_tracking":     ["customer", "agent"],
    "human_handoff":      ["agent", "underwriter"],
}

# ── 工具 → 领域映射表 ──
TOOL_DOMAIN_MAP = {
    "product_compare":  ["insurance"],
    "premium_calc":     ["insurance", "claim"],
    "policy_query":      ["claim", "service"],
    "clause_search":     ["claim", "insurance", "underwriting"],
    "claim_eligibility": ["claim"],
    "claim_tracking":    ["claim"],
    "human_handoff":     ["claim", "insurance", "underwriting", "service"],
}

# 全局依赖（用于降级场景）
_global_deps: dict = {}


def get_all_tools(
    vector_store=None,
    mysql_session=None,
    redis_session=None,
    llm_client=None,
) -> list[BaseTool]:
    """
    返回所有 Agent 工具实例，注入真实服务依赖

    Args:
        vector_store: VectorStore 实例
        mysql_session: MySQL 会话
        redis_session: Redis 客户端
        llm_client: LLM 客户端

    Returns:
        7 个已注入依赖的工具实例列表
    """
    from agent.tools.policy_query import PolicyQueryTool
    from agent.tools.claim_eligibility import ClaimEligibilityTool
    from agent.tools.clause_search import ClauseSearchTool
    from agent.tools.premium_calc import PremiumCalcTool
    from agent.tools.product_compare import ProductCompareTool
    from agent.tools.claim_tracking import ClaimTrackingTool
    from agent.tools.human_handoff import HumanHandoffTool

    return [
        PolicyQueryTool(mysql_session=mysql_session),
        ClaimEligibilityTool(vector_store=vector_store, mysql_session=mysql_session),
        ClauseSearchTool(vector_store=vector_store),
        PremiumCalcTool(mysql_session=mysql_session),
        ProductCompareTool(vector_store=vector_store, mysql_session=mysql_session, llm_client=llm_client),
        ClaimTrackingTool(mysql_session=mysql_session),
        HumanHandoffTool(mysql_session=mysql_session, redis_session=redis_session),
    ]


def get_tool_by_name(name: str, **deps) -> Optional[BaseTool]:
    """按名称查找工具（支持依赖注入）"""
    for tool in get_all_tools(**deps):
        if tool.name == name:
            return tool
    return None


def get_tools_by_domain(
    domain: str,
    vector_store=None,
    mysql_session=None,
    redis_session=None,
    llm_client=None,
    user_context=None,
    user_role: str = "agent",
) -> list:
    """
    按业务领域 + 角色返回工具集 — 双通道架构核心函数

    每个子 Agent 调用此函数获取本领域 + 本角色可用的工具。

    Args:
        domain: 领域名称 ("insurance", "underwriting", "claim", "service")
        vector_store: VectorStore 实例
        mysql_session: MySQL 会话
        redis_session: Redis 客户端
        llm_client: LLM 客户端
        user_context: UserContext 实例（用于数据隔离）
        user_role: 用户角色 (customer / agent / underwriter / admin)

    Returns:
        list[BaseTool]: 该领域 + 该角色可用的工具列表
    """
    all_tools = get_all_tools(
        vector_store=vector_store,
        mysql_session=mysql_session,
        redis_session=redis_session,
        llm_client=llm_client,
    )

    # ── 按领域过滤 ──
    domain_tools = []
    for tool in all_tools:
        tool_name = tool.name
        if tool_name in TOOL_DOMAIN_MAP and domain in TOOL_DOMAIN_MAP[tool_name]:
            domain_tools.append(tool)

    # ── 按角色过滤（双通道权限隔离）──
    role_tools = []
    for tool in domain_tools:
        tool_name = tool.name
        if tool_name in TOOL_ROLE_MAP and user_role in TOOL_ROLE_MAP[tool_name]:
            if user_context is not None and hasattr(tool, 'user_context'):
                tool.user_context = user_context
            role_tools.append(tool)

    logger.info(
        f"[Tools] 领域 '{domain}' + 角色 '{user_role}' → "
        f"{len(role_tools)}/{len(domain_tools)} 个工具: "
        f"{[t.name for t in role_tools]}"
    )

    return role_tools
