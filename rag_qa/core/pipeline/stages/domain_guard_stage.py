# -*- coding: utf-8 -*-
"""领域边界守卫 Stage — 拦截非保险查询（<1ms）"""

import time
from rag_qa.core.pipeline.stage import Stage
from rag_qa.core.pipeline.context import PipelineContext, StageResult


class DomainGuardStage(Stage):
    """检查查询是否属于保险领域，非保险查询直接拒绝"""

    name = "domain_guard"

    def __init__(self, guard=None):
        self._guard = guard

    def can_execute(self, ctx: PipelineContext) -> bool:
        # complexity=0 已在 FAQStage 短路，此 Stage 仅在 >=1 时启用
        return ctx.complexity >= 1

    def execute(self, ctx: PipelineContext) -> StageResult:
        t0 = time.time()
        try:
            if self._guard is None:
                from rag_qa.core.domain_guard import DomainBoundaryGuard
                self._guard = DomainBoundaryGuard()

            result = self._guard.check(ctx.query, user_role=ctx.user_role)
            timing = round((time.time() - t0) * 1000, 1)
            ctx.pipeline[self.name] = timing

            if not result.passed:
                ctx.domain_guard_passed = False
                ctx.generated_answer = result.fallback_response
                ctx.intent = "out_of_domain"
                return StageResult(
                    status="skip",
                    data={"reason": result.reason},
                    timing_ms=timing,
                )

            return StageResult(status="success", timing_ms=timing)

        except Exception as e:
            timing = round((time.time() - t0) * 1000, 1)
            ctx.pipeline[self.name] = timing
            return StageResult(status="degraded", timing_ms=timing,
                               data={"warning": str(e)})
