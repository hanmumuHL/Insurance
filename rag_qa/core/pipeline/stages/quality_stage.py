# -*- coding: utf-8 -*-
"""检索质量检查 Stage — 质量不足时拒绝生成答案（<1ms）"""

import time
from rag_qa.core.pipeline.stage import Stage
from rag_qa.core.pipeline.context import PipelineContext, StageResult
from base.logger import logger


class QualityStage(Stage):
    """
    检索质量检查 Stage

    检查检索结果的 Top-1 分数、Top-5 平均分、好 chunk 数量。
    质量不足时拒绝生成答案，返回兜底话术。
    """

    name = "quality"

    def __init__(self, quality_guard=None):
        self._guard = quality_guard

    def can_execute(self, ctx: PipelineContext) -> bool:
        return ctx.complexity >= 2

    def execute(self, ctx: PipelineContext) -> StageResult:
        t0 = time.time()
        try:
            if self._guard is None:
                from rag_qa.core.retrieval_quality_guard import \
                    RetrievalQualityGuard
                self._guard = RetrievalQualityGuard()

            quality = self._guard.check(ctx.retrieved_chunks, ctx.intent)
            timing = round((time.time() - t0) * 1000, 1)
            ctx.pipeline[self.name] = timing

            if not quality.passed:
                ctx.quality_passed = False
                ctx.quality_fallback = quality.fallback_response
                ctx.generated_answer = quality.fallback_response
                logger.warning(f"QualityStage: 不通过 — {quality.reason}")
                return StageResult(
                    status="skip",
                    data={"reason": quality.reason},
                    timing_ms=timing,
                )

            return StageResult(status="success", timing_ms=timing)

        except Exception as e:
            timing = round((time.time() - t0) * 1000, 1)
            ctx.pipeline[self.name] = timing
            logger.warning(f"QualityStage 失败: {e}")
            return StageResult(status="degraded", timing_ms=timing,
                               data={"warning": str(e)})
