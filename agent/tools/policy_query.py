# -*- coding: utf-8 -*-
"""
工具 1: 保单查询 — 查 MySQL policy_cache 表
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from base.logger import logger
from agent.tools._escape import escape_like_pattern


class PolicyQueryInput(BaseModel):
    insurer: str = Field(default="", description="保险公司名称，如 '平安健康'")
    product_name: str = Field(default="", description="产品名称，如 '平安e生保'")
    user_id: str = Field(default="", description="用户 ID 或证件号后4位")


class PolicyQueryTool(BaseTool):
    """
    保单查询工具 — 查 MySQL policy_cache 表

    数据来源: MySQL policy_cache 表 (从保司 API 定期同步的脱敏副本)
    返回: 保单状态、保额、保障期限、保费等信息
    """
    name: str = "policy_query"
    description: str = (
        "查询用户的保险保单信息。可以查询保单状态、保额、保障期限、"
        "保费金额等。需要提供保险公司名称和/或产品名称。"
        "当用户询问自己的保单情况时使用此工具。"
    )
    args_schema: type = PolicyQueryInput

    # ── 依赖注入 ──
    mysql_session: Optional[object] = None
    user_context: Optional[object] = None

    def _run(self, insurer: str = "", product_name: str = "", user_id: str = "") -> str:
        """
        查询保单信息

        数据隔离:
          - customer 角色: 强制使用自己的 user_id，只能查自己的保单
          - agent/underwriter 角色: 可以指定任意 user_id 查询
        """
        logger.info(f"工具调用: policy_query(insurer={insurer}, product={product_name})")

        if self.user_context is not None and hasattr(self.user_context, 'is_customer'):
            if self.user_context.is_customer:
                user_id = self.user_context.user_id
                logger.debug(f"[Security] Customer 数据隔离: 强制 user_id={user_id}")

        if self.mysql_session is None:
            return self._fallback(insurer, product_name)

        try:
            conditions = []
            params = []

            if insurer:
                conditions.append("insurer LIKE %s")
                params.append(f"%{escape_like_pattern(insurer)}%")
            if product_name:
                conditions.append("product_name LIKE %s")
                params.append(f"%{escape_like_pattern(product_name)}%")
            if user_id is not None and user_id != "":
                conditions.append("user_id = %s")
                params.append(user_id)

            where = " AND ".join(conditions) if conditions else "1=1"

            sql = f"""
                SELECT policy_no_masked, insurer, product_name,
                       status, sum_insured, premium,
                       effective_date, expire_date
                FROM policy_cache
                WHERE {where}
                  AND is_valid = 1
                ORDER BY effective_date DESC
                LIMIT 5
            """

            rows = self.mysql_session.execute(sql, tuple(params)).fetchall()

            if not rows:
                return (
                    f"保单查询结果:\n"
                    f"- 未找到匹配的保单\n"
                    f"- 查询条件: 保司={insurer or '不限'}, 产品={product_name or '不限'}\n"
                    f"- 如确认有保单，请联系人工客服核实"
                )

            lines = [f"保单查询结果: 找到 {len(rows)} 条保单"]
            for row in rows:
                lines.append(
                    f"\n- 保单号: {row[0]} | {row[1]} {row[2]}\n"
                    f"  状态: {row[3]} | 保额: {row[4]}元 | 保费: {row[5]}元\n"
                    f"  有效期: {str(row[6])[:10]} 至 {str(row[7])[:10]}"
                )

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"保单查询失败: {e}")
            return f"保单查询异常: {e}"

    def _fallback(self, insurer: str, product_name: str) -> str:
        logger.warning("保单查询: MySQL 未连接，返回提示")
        return (
            f"保单查询暂不可用（数据库未连接）。\n"
            f"查询条件: 保司={insurer or '不限'}, 产品={product_name or '不限'}\n"
            f"请联系人工客服查询保单信息。"
        )
