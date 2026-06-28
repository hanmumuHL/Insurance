# -*- coding: utf-8 -*-
"""检索 Stage — 统一检索入口（~30-50ms）

委托 RetrievalInterface 执行 encode → Milvus search → rerank 整条链。
所有检索需求（RAG 管道、Agent Tools）通过此 Stage 统一获取结果。
"""

import time
from rag_qa.core.pipeline.stage import Stage
from rag_qa.core.pipeline.context import PipelineContext, StageResult
from base.logger import logger


class RetrievalStage(Stage):
    """
    统一检索 Stage

    封装 BGE-M3 编码 → Milvus 混合检索 → Cross-Encoder 精排 整条链。
    使用 RetrievalInterface，确保 RAG 管道和 Agent Tools 走同一检索入口。
    """

    name = "retrieval"

    def __init__(self, retrieval_interface=None, vector_store=None):
        """
        Args:
            retrieval_interface: RetrievalInterface 实例（依赖注入）
            vector_store: VectorStore 实例（创建默认 RetrievalInterface 时使用）
        """
        self._retrieval = retrieval_interface
        self._vector_store = vector_store

    def can_execute(self, ctx: PipelineContext) -> bool:
        return ctx.complexity >= 1

    def execute(self, ctx: PipelineContext) -> StageResult:
        t0 = time.time()
        try:
            if self._retrieval is None:
                from rag_qa.core.pipeline.retrieval_interface import \
                    RetrievalInterface
                self._retrieval = RetrievalInterface(
                    vector_store=self._vector_store
                )

            # 从 strategy_plan 提取 filters（如果有）
            filters = {}
            if ctx.strategy_plan is not None:
                filters = getattr(ctx.strategy_plan, "filters", {})

            chunks = self._retrieval.retrieve(
                query=ctx.query,
                filters=filters,
            )

            timing = round((time.time() - t0) * 1000, 1)
            ctx.pipeline[self.name] = timing
            ctx.retrieved_chunks = chunks

            logger.info(f"RetrievalStage: {len(chunks)} chunks")
            return StageResult(
                status="success",
                data={"chunk_count": len(chunks)},
                timing_ms=timing,
            )

        except Exception as e:
            timing = round((time.time() - t0) * 1000, 1)
            ctx.pipeline[self.name] = timing
            logger.error(f"RetrievalStage 失败: {e}")
            return StageResult(status="failed", timing_ms=timing, error=str(e))
