# -*- coding: utf-8 -*-
"""检索策略选择 Stage — 根据意图和实体选择检索策略（<1ms）"""

import time
from rag_qa.core.pipeline.stage import Stage
from rag_qa.core.pipeline.context import PipelineContext, StageResult
from base.logger import logger


class StrategyStage(Stage):
    """选择检索策略（direct / hyde / sub_query / compare / conditional）"""

    name = "strategy"

    def __init__(self, strategy_selector=None):
        self._selector = strategy_selector

    def can_execute(self, ctx: PipelineContext) -> bool:
        return ctx.complexity >= 2

    def execute(self, ctx: PipelineContext) -> StageResult:
        t0 = time.time()
        try:
            if self._selector is None:
                from rag_qa.core.strategy_selector import StrategySelector
                self._selector = StrategySelector()

            strategy_plan = self._selector.select(
                ctx.intent, ctx.query, ctx.entities
            )
            timing = round((time.time() - t0) * 1000, 1)
            ctx.pipeline[self.name] = timing
            ctx.strategy_plan = strategy_plan

            logger.info(
                f"StrategyStage: {strategy_plan.strategy.value} — "
                f"{strategy_plan.reason}"
            )
            return StageResult(status="success", timing_ms=timing)

        except Exception as e:
            timing = round((time.time() - t0) * 1000, 1)
            ctx.pipeline[self.name] = timing
            logger.warning(f"StrategyStage 失败: {e}")
            return StageResult(status="degraded", timing_ms=timing,
                               data={"warning": str(e)})
