# -*- coding: utf-8 -*-
"""FAQ 缓存检查 Stage — 精确命中直接返回（<1ms）"""

import time
from rag_qa.core.pipeline.stage import Stage
from rag_qa.core.pipeline.context import PipelineContext, StageResult


class FAQStage(Stage):
    """检查 FAQ 缓存，命中则短路返回"""

    name = "faq"

    def __init__(self, faq_cache=None):
        self._faq_cache = faq_cache

    def execute(self, ctx: PipelineContext) -> StageResult:
        t0 = time.time()
        try:
            if self._faq_cache is None:
                from cache.faq_cache import FAQCache
                from cache.redis_client import RedisClient
                self._faq_cache = FAQCache(RedisClient())

            answer = self._faq_cache.try_hit(ctx.query)
            timing = round((time.time() - t0) * 1000, 1)
            ctx.pipeline[self.name] = timing

            if answer:
                ctx.faq_answer = answer
                ctx.complexity = 0
                return StageResult(
                    status="skip", data={"answer": answer}, timing_ms=timing
                )

            return StageResult(status="success", timing_ms=timing)

        except Exception as e:
            timing = round((time.time() - t0) * 1000, 1)
            ctx.pipeline[self.name] = timing
            return StageResult(status="degraded", timing_ms=timing,
                               data={"warning": str(e)})
