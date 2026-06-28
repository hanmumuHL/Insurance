# -*- coding: utf-8 -*-
"""
工具 4: 保费试算 — 查 MySQL 费率表
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from base.logger import logger
from agent.tools._escape import escape_like_pattern


class PremiumCalcInput(BaseModel):
    product_name: str = Field(default="", description="产品名称")
    age: int = Field(default=30, description="被保险人年龄")
    sum_insured: str = Field(default="", description="保额档位，如 '50万'")
    riders: str = Field(default="", description="附加险，逗号分隔")


class PremiumCalcTool(BaseTool):
    """
    保费试算工具 — 查 MySQL 费率表

    数据来源: MySQL rate_table (保司提供的费率表，定期同步)
    """
    name: str = "premium_calc"
    description: str = (
        "计算保险产品的保费。需要提供产品名称、被保险人年龄、"
        "保额等信息。返回年保费金额和缴费方式。"
        "当用户询问保险产品的价格或保费时使用此工具。"
    )
    args_schema: type = PremiumCalcInput

    # ── 依赖注入 ──
    mysql_session: Optional[object] = None

    def _run(self, product_name: str = "", age: int = 30, sum_insured: str = "", riders: str = "") -> str:
        logger.info(f"工具调用: premium_calc(product={product_name}, age={age})")

        if self.mysql_session is None:
            return self._fallback(product_name, age, sum_insured, riders)

        try:
            params = []
            conditions = ["product_name LIKE %s"]
            params.append(f"%{escape_like_pattern(product_name)}%")

            conditions.append("min_age <= %s AND max_age >= %s")
            params.extend([age, age])

            if sum_insured:
                conditions.append("sum_insured = %s")
                params.append(sum_insured)

            where = " AND ".join(conditions)

            sql = f"""
                SELECT product_name, premium_yearly, premium_monthly,
                       payment_methods, sum_insured, age
                FROM rate_table
                WHERE {where}
                ORDER BY premium_yearly ASC
                LIMIT 1
            """

            rows = self.mysql_session.execute(sql, tuple(params))
            row = rows.fetchone()

            if not row:
                return (
                    f"保费试算结果:\n"
                    f"- 产品: {product_name}\n"
                    f"- 年龄: {age}岁\n"
                    f"- 保额: {sum_insured or '默认'}\n"
                    f"- ⚠️ 未找到匹配的费率数据\n"
                    f"- 建议: 确认产品名和年龄是否正确"
                )

            return (
                f"保费试算结果:\n"
                f"- 产品: {row[0]}\n"
                f"- 被保险人年龄: {row[5]}岁\n"
                f"- 保额: {row[4]}\n"
                f"- 年保费: {row[1]}元\n"
                f"- 月保费: {row[2]}元\n"
                f"- 缴费方式: {row[3]}\n"
                f"- 附加险: {'无' if not riders else riders}\n"
                f"- ⚠️ 实际保费以投保时的核保结果为准\n"
            )

        except Exception as e:
            logger.error(f"保费试算失败: {e}")
            return f"保费试算异常: {e}"

    def _fallback(self, product_name: str, age: int, sum_insured: str, riders: str) -> str:
        logger.warning("保费试算: MySQL 未连接")
        return (
            f"保费试算暂不可用（费率数据库未连接）。\n"
            f"查询条件: 产品={product_name}, 年龄={age}岁\n"
            f"请联系人工客服获取准确报价。"
        )
