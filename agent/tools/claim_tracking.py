# -*- coding: utf-8 -*-
"""
工具 6: 理赔追踪 — 查 MySQL claim_records 表
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from base.logger import logger


class ClaimTrackingInput(BaseModel):
    report_no: str = Field(default="", description="理赔报案号")


class ClaimTrackingTool(BaseTool):
    """
    理赔进度追踪工具 — 查 MySQL claim_records 表

    数据来源: MySQL claim_records 表 (从保司理赔系统定期同步)
    """
    name: str = "claim_tracking"
    description: str = (
        "查询理赔案件的处理进度。需要提供理赔报案号。"
        "返回当前处理阶段、预计时效和需要补充的材料。"
        "当用户询问理赔进度时使用此工具。"
    )
    args_schema: type = ClaimTrackingInput

    # ── 依赖注入 ──
    mysql_session: Optional[object] = None

    def _run(self, report_no: str = "", user_id: str = "") -> str:
        """
        理赔进度查询

        数据隔离:
          - customer 角色: 强制按自己的 user_id 过滤
          - agent/underwriter: 若传入了 user_id 则过滤，否则查全部
        """
        logger.info(f"工具调用: claim_tracking(report_no={report_no})")

        if not report_no:
            return "请提供理赔报案号以查询进度。"

        if self.mysql_session is None:
            return self._fallback(report_no)

        try:
            effective_user_id = user_id
            if hasattr(self, 'user_context') and self.user_context is not None:
                if self.user_context.is_customer:
                    effective_user_id = self.user_context.user_id

            if effective_user_id is not None and effective_user_id != "":
                sql = """
                    SELECT report_no, status, stage, submitted_at,
                           estimated_days, need_materials, remarks
                    FROM claim_records
                    WHERE report_no = %s AND user_id = %s
                """
                rows = self.mysql_session.execute(sql, (report_no, effective_user_id))
            else:
                sql = """
                    SELECT report_no, status, stage, submitted_at,
                           estimated_days, need_materials, remarks
                    FROM claim_records
                    WHERE report_no = %s
                """
                rows = self.mysql_session.execute(sql, (report_no,))

            row = rows.fetchone()

            if not row:
                return (
                    f"理赔进度查询:\n"
                    f"- 报案号: {report_no}\n"
                    f"- ⚠️ 未找到该报案号对应的理赔记录\n"
                    f"- 请确认报案号是否正确，或拨打客服热线查询"
                )

            return (
                f"理赔进度查询:\n"
                f"- 报案号: {row[0]}\n"
                f"- 状态: {row[1]}\n"
                f"- 当前阶段: {row[2]}\n"
                f"- 提交时间: {str(row[3])[:10]}\n"
                f"- 预计完成: {row[4]}个工作日\n"
                f"- 补充材料: {row[5] or '无'}\n"
                f"- 备注: {row[6] or '无'}\n"
                f"- 如有疑问请拨打客服热线"
            )

        except Exception as e:
            logger.error(f"理赔追踪失败: {e}")
            return f"理赔追踪查询异常: {e}"

    def _fallback(self, report_no: str) -> str:
        logger.warning("理赔追踪: MySQL 未连接")
        return (
            f"理赔追踪暂不可用（数据库未连接）。\n"
            f"报案号: {report_no}\n"
            f"请拨打客服热线查询理赔进度。"
        )
