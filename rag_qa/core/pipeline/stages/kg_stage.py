# -*- coding: utf-8 -*-
"""KG 推理 Stage — 知识图谱实体链接 + 推理（可选，~15-30ms）

仅在 complexity>=2 且 entities 非空时执行。
KG 不可用时静默降级（返回 degraded 而非 failed）。
"""

import time
from rag_qa.core.pipeline.stage import Stage
from rag_qa.core.pipeline.context import PipelineContext, StageResult
from base.logger import logger


class KGStage(Stage):
    """
    KG 推理增强 Stage

    为复杂查询（理赔/对比/核保）提供知识图谱推理上下文。
    推理结果注入 ctx.kg_context，后续 GenerateStage 会拼入 LLM prompt。

    降级策略: KG 不可用或未命中时返回空字符串，不影响主流程。
    """

    name = "kg"

    def __init__(self, kg_service=None):
        self._kg = kg_service

    def can_execute(self, ctx: PipelineContext) -> bool:
        return ctx.complexity >= 2 and bool(ctx.entities)

    def execute(self, ctx: PipelineContext) -> StageResult:
        t0 = time.time()
        try:
            if self._kg is None:
                from rag_qa.core.kg.service import KGService
                self._kg = KGService()

            ctx.kg_context = self._kg.get_reasoning(ctx.query, ctx.intent)
            timing = round((time.time() - t0) * 1000, 1)
            ctx.pipeline[self.name] = timing

            if ctx.kg_context:
                logger.info(f"KGStage: 推理成功 ({len(ctx.kg_context)} chars)")
                return StageResult(status="success", timing_ms=timing)
            else:
                logger.debug("KGStage: 无推理结果")
                return StageResult(status="degraded", timing_ms=timing,
                                   data={"reason": "no reasoning paths"})

        except Exception as e:
            timing = round((time.time() - t0) * 1000, 1)
            ctx.pipeline[self.name] = timing
            logger.debug(f"KGStage 降级: {e}")
            return StageResult(status="degraded", timing_ms=timing,
                               data={"warning": str(e)})
