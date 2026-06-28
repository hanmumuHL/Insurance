# -*- coding: utf-8 -*-
"""
统一合规审查 Stage — 所有路径的最终必经阶段

统一当前两套合规实现:
  - ComplianceGuard.check() (5 规则，角色感知) — RAG 管道使用
  - _compliance_review() (弱字符串替换)       — Agent 管道使用

改造后: 所有路径统一走 ComplianceGuard.check()，
        agent 通道的弱审查被替换为完整的 5 规则检查。
"""

import time
from rag_qa.core.pipeline.stage import Stage
from rag_qa.core.pipeline.context import PipelineContext, StageResult
from base.logger import logger


class ComplianceStage(Stage):
    """
    统一合规审查 Stage

    5 规则检查: 医疗建议 | 监管敏感词 | 贬低 | 金额引用 | 角色感知严格度
    所有路径（customer RAG、agent multi-agent）统一走此阶段。
    """

    name = "compliance"

    def __init__(self, compliance_guard=None):
        self._guard = compliance_guard

    def can_execute(self, ctx: PipelineContext) -> bool:
        # 有生成答案时才做合规检查
        return bool(ctx.generated_answer)

    def execute(self, ctx: PipelineContext) -> StageResult:
        t0 = time.time()
        try:
            if self._guard is None:
                from rag_qa.core.compliance_guard import ComplianceGuard
                self._guard = ComplianceGuard()

            result = self._guard.check(
                response=ctx.generated_answer,
                context_chunks=ctx.retrieved_chunks,
                intent=ctx.intent,
                user_role=ctx.user_role,
            )

            timing = round((time.time() - t0) * 1000, 1)
            ctx.pipeline[self.name] = timing
            ctx.compliance_passed = result.passed
            ctx.compliance_violations = result.violated_rules or []

            if not result.passed:
                logger.warning(
                    f"ComplianceStage: 不通过 — {result.violated_rules}"
                )
                ctx.generated_answer = (
                    "很抱歉，您的问题我暂时无法直接回答。"
                    "建议您通过官方渠道或联系人工客服获取更准确的信息。"
                )
                return StageResult(
                    status="skip",
                    data={"violations": result.violated_rules},
                    timing_ms=timing,
                )

            # 合规通过但有修改（如追加了医疗免责声明）
            if result.modified_response:
                ctx.compliance_modified = result.modified_response
                ctx.generated_answer = result.modified_response
                logger.info("ComplianceStage: 答案被修改（追加免责声明等）")

            return StageResult(status="success", timing_ms=timing)

        except Exception as e:
            timing = round((time.time() - t0) * 1000, 1)
            ctx.pipeline[self.name] = timing
            logger.error(f"ComplianceStage 异常: {e}", exc_info=True)
            # 合规检查异常时返回安全兜底话术（不放行原始答案）
            ctx.generated_answer = "抱歉，系统暂时繁忙，请稍后再试或联系人工客服。"
            return StageResult(status="failed", timing_ms=timing,
                               error=f"合规检查异常: {e}")
