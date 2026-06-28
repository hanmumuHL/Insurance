# -*- coding: utf-8 -*-
"""缓存写入 Stage — 将结果写入查询缓存（<1ms）"""

import time
from rag_qa.core.pipeline.stage import Stage
from rag_qa.core.pipeline.context import PipelineContext, StageResult


class CacheWriteStage(Stage):
    """将生成的答案写入 QueryResultCache，供后续相同查询复用"""

    name = "cache_write"

    def __init__(self, query_cache=None):
        self._query_cache = query_cache

    def can_execute(self, ctx: PipelineContext) -> bool:
        # 只在成功生成答案且非缓存命中时写入
        return (
            ctx.complexity >= 2
            and bool(ctx.generated_answer)
            and not ctx.cache_hit
        )

    def execute(self, ctx: PipelineContext) -> StageResult:
        t0 = time.time()
        try:
            if self._query_cache is None:
                from cache.query_result_cache import QueryResultCache
                from cache.redis_client import RedisClient
                from config.settings import settings
                self._query_cache = QueryResultCache(
                    RedisClient(), ttl=settings.query_result_cache_ttl
                )

            self._query_cache.set(
                ctx.query,
                ctx.intent,
                {"answer": ctx.generated_answer, "intent": ctx.intent},
            )

            timing = round((time.time() - t0) * 1000, 1)
            ctx.pipeline[self.name] = timing
            return StageResult(status="success", timing_ms=timing)

        except Exception as e:
            timing = round((time.time() - t0) * 1000, 1)
            ctx.pipeline[self.name] = timing
            # 缓存写入失败不影响主流程
            return StageResult(status="degraded", timing_ms=timing,
                               data={"warning": str(e)})
