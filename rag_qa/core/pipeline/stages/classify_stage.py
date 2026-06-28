# -*- coding: utf-8 -*-
"""意图分类 Stage — 三层路由: 规则 → BERT → LLM（~5ms）"""

import time
from rag_qa.core.pipeline.stage import Stage
from rag_qa.core.pipeline.context import PipelineContext, StageResult
from base.logger import logger


class ClassifyStage(Stage):
    """三层意图分类: 关键词规则 → BERT 分类 → LLM 兜底"""

    name = "classify"

    def __init__(self, classifier=None):
        self._classifier = classifier

    def can_execute(self, ctx: PipelineContext) -> bool:
        return ctx.complexity >= 1

    def execute(self, ctx: PipelineContext) -> StageResult:
        t0 = time.time()
        try:
            if self._classifier is None:
                from rag_qa.core.query_classifier import QueryClassifier
                from rag_qa.core.bert_intent_classifier import get_bert_classifier
                from base.llm_client import get_llm_client
                bert_model = get_bert_classifier()
                llm_client = get_llm_client()
                self._classifier = QueryClassifier(
                    bert_model=bert_model, llm_client=llm_client
                )

            intent_result = self._classifier.classify(ctx.query)
            timing = round((time.time() - t0) * 1000, 1)
            ctx.pipeline[self.name] = timing

            ctx.intent = intent_result.intent
            ctx.entities = intent_result.entities
            ctx.classifier_source = intent_result.source

            # 意图被拒绝 → 短路
            if intent_result.reject:
                ctx.generated_answer = intent_result.fallback_response
                ctx.complexity = 0
                logger.info(
                    f"ClassifyStage: 意图拒绝 reason={intent_result.reject_reason}"
                )
                return StageResult(
                    status="skip",
                    data={"reason": intent_result.reject_reason},
                    timing_ms=timing,
                )

            logger.info(
                f"ClassifyStage: intent={ctx.intent} "
                f"confidence={intent_result.confidence:.2f} "
                f"source={intent_result.source}"
            )
            return StageResult(status="success", timing_ms=timing)

        except Exception as e:
            timing = round((time.time() - t0) * 1000, 1)
            ctx.pipeline[self.name] = timing
            logger.warning(f"ClassifyStage 失败: {e}")
            return StageResult(status="degraded", timing_ms=timing,
                               data={"warning": str(e)})
