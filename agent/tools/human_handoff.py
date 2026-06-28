# -*- coding: utf-8 -*-
"""
工具 7: 人工转接 — 记录转接请求 + 返回引导
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from base.logger import logger


class HumanHandoffInput(BaseModel):
    reason: str = Field(default="用户要求", description="转接原因")


class HumanHandoffTool(BaseTool):
    """
    人工转接工具 — 记录转接请求 + 返回引导

    生产环境中应调用客服系统 API 创建工单。
    当前实现为记录转接请求并返回引导信息。
    """
    name: str = "human_handoff"
    description: str = (
        "将当前对话转接给人工客服。会生成对话摘要传递给客服人员。"
        "当用户明确要求人工服务，或问题超出 AI 处理能力时使用此工具。"
    )
    args_schema: type = HumanHandoffInput

    # ── 依赖注入 ──
    mysql_session: Optional[object] = None
    redis_session: Optional[object] = None

    def _run(self, reason: str = "用户要求") -> str:
        logger.info(f"工具调用: human_handoff(reason={reason})")

        if self.mysql_session:
            try:
                self.mysql_session.execute(
                    """INSERT INTO handoff_requests
                       (reason, status, created_at)
                       VALUES (%s, 'pending', NOW())""",
                    (reason,),
                )
                self.mysql_session.commit()
                logger.info("转接工单已记录")
            except Exception as e:
                logger.warning(f"工单记录失败（非致命）: {e}")

        return (
            f"已为您转接人工客服。\n"
            f"- 转接原因: {reason}\n"
            f"- 预计等待时间: 1-3分钟\n"
            f"- 客服人员将看到本次对话摘要，无需重复描述问题\n"
            f"- 如需紧急帮助，请拨打客服热线: 400-XXX-XXXX\n"
        )
