# -*- coding: utf-8 -*-
"""查询缓存 Stage — 检查 QueryResultCache（<1ms）"""

import time
from rag_qa.core.pipeline.stage import Stage
from rag_qa.core.pipeline.context import PipelineContext, StageResult


class CacheStage(Stage):
    """检查查询结果缓存，命中则短路返回"""

    name = "cache"

    def __init__(self, query_cache=None):
        self._query_cache = query_cache

    def can_execute(self, ctx: PipelineContext) -> bool:
        return ctx.complexity >= 1

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

            cached = self._query_cache.try_get(ctx.query, ctx.intent)
            timing = round((time.time() - t0) * 1000, 1)
            ctx.pipeline[self.name] = timing

            if cached:
                ctx.cache_hit = True
                ctx.generated_answer = cached.get("answer", "")
                return StageResult(
                    status="skip",
                    data={"cached_intent": cached.get("intent", "")},
                    timing_ms=timing,
                )

            return StageResult(status="success", timing_ms=timing)

        except Exception as e:
            timing = round((time.time() - t0) * 1000, 1)
            ctx.pipeline[self.name] = timing
            return StageResult(status="degraded", timing_ms=timing,
                               data={"warning": str(e)})
