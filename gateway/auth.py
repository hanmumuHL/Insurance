# -*- coding: utf-8 -*-
"""
双通道认证与角色模型

支持四种角色:
  - CUSTOMER:    外部客户 — 只能查自己的保单，走 RAG Pipeline，严格合规
  - AGENT:        保险顾问 — 可查任意客户数据，走 Multi-Agent 编排
  - UNDERWRITER: 核保人员 — 同 AGENT，可访问核保敏感字段
  - ADMIN:       管理员 — 文档导入、系统配置

生产环境: 由上游 API Gateway (Spring Gateway / Nginx) 完成 SSO 认证后，
          在请求头中注入 X-User-Id、X-User-Role、X-Org-Id。
          本模块只做提取和校验，不做密码/Token 验证。

开发环境: 不传 header 时默认为 agent 角色（向后兼容）。
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

from fastapi import Header, HTTPException, Depends

from base.logger import logger
from config.settings import settings


class UserRole(str, Enum):
    """用户角色枚举"""
    CUSTOMER = "customer"           # 外部客户
    AGENT = "agent"                 # 保险顾问/销售
    UNDERWRITER = "underwriter"     # 核保人员
    ADMIN = "admin"                 # 管理员


# 角色分组
INTERNAL_ROLES = {UserRole.AGENT, UserRole.UNDERWRITER, UserRole.ADMIN}
EXTERNAL_ROLES = {UserRole.CUSTOMER}
ADMIN_ROLES = {UserRole.ADMIN}


@dataclass
class UserContext:
    """
    用户上下文 — 从请求头提取，贯穿整个请求生命周期

    Attributes:
        user_id: 用户唯一标识（客户 ID 或员工工号）
        role: 用户角色
        org_id: 所属机构（保险公司/经纪公司代码）
        display_name: 用户显示名（用于日志）
    """
    user_id: str = ""
    role: UserRole = UserRole.AGENT
    org_id: str = ""
    display_name: str = ""

    @property
    def is_internal(self) -> bool:
        """是否为内部人员"""
        return self.role in INTERNAL_ROLES

    @property
    def is_customer(self) -> bool:
        """是否为外部客户"""
        return self.role == UserRole.CUSTOMER

    @property
    def is_admin(self) -> bool:
        """是否为管理员"""
        return self.role == UserRole.ADMIN

    @property
    def channel(self) -> str:
        """返回通道标识: 'customer' 或 'agent'"""
        return "customer" if self.is_customer else "agent"


async def get_current_user(
    x_user_id: str = Header("", alias="X-User-Id"),
    x_user_role: str = Header(settings.auth.default_role, alias="X-User-Role"),
    x_org_id: str = Header("", alias="X-Org-Id"),
    x_display_name: str = Header("", alias="X-Display-Name"),
) -> UserContext:
    """
    FastAPI Dependency — 从请求头提取用户信息

    生产环境中由上游 API Gateway 完成 SSO 后注入这些 header。
    开发/测试环境中不传则默认 agent 角色（向后兼容）。

    Raises:
        HTTPException 401: 外部客户未提供 X-User-Id
        HTTPException 400: 非法的 X-User-Role 值
    """
    # ── 解析角色 ──
    valid_roles = {r.value for r in UserRole}
    if x_user_role not in valid_roles:
        logger.warning(f"非法的 X-User-Role: '{x_user_role}'，降级为 agent")
        role = UserRole.AGENT
    else:
        role = UserRole(x_user_role)

    # ── 外部客户必须有 user_id ──
    if role == UserRole.CUSTOMER and not x_user_id:
        if settings.auth.auth_enabled:
            raise HTTPException(
                status_code=401,
                detail="外部客户必须提供 X-User-Id",
            )
        logger.warning("外部客户未提供 X-User-Id，auth_enabled=False 放行")

    # ── Admin 端点校验 ──
    # 注意: 具体端点的 admin 校验在路由层做，这里只提取上下文

    user = UserContext(
        user_id=x_user_id,
        role=role,
        org_id=x_org_id,
        display_name=x_display_name or x_user_id,
    )

    logger.debug(
        f"[Auth] user={user.display_name} role={user.role.value} "
        f"channel={user.channel}"
    )

    return user
