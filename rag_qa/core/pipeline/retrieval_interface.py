# -*- coding: utf-8 -*-
"""
统一检索接口 — 封装 encoder → Milvus search → reranker 整条检索链

解决重复问题:
  当前有 4 处独立调用检索链:
    - RAGSystem._do_retrieval()         (rag_system.py:343)
    - ClauseSearchTool._run()           (all_tools.py:341)
    - ClaimEligibilityTool._run()       (all_tools.py:197)
    - ProductCompareTool._run()         (all_tools.py:573, 间接)
  每处都独立 import encoder、调 encode()、调 vector_store.search()、调 reranker.rerank()。

  RetrievalInterface 将这条链封装为单个 retrieve() 方法，
  所有调用方改为委托，消除重复。

使用方式:
    from rag_qa.core.pipeline.retrieval_interface import RetrievalInterface

    ri = RetrievalInterface()          # 使用全局单例
    # 或
    ri = RetrievalInterface(vector_store=my_vs)  # 依赖注入

    chunks = ri.retrieve("肺炎住院能赔吗", filters={"insurer": "平安"})
"""

from base.logger import logger
from base.encoder import get_encoder
from base.reranker import get_reranker
from config.settings import settings


class RetrievalInterface:
    """
    统一检索接口

    封装 BGE-M3 编码 → Milvus 混合检索 → Cross-Encoder 精排 整条链。
    所有需要检索的组件（RAG 管道、Agent Tools）都通过此接口获取结果。

    Attributes:
        _vector_store: Milvus 向量库（可选，默认使用 RAGSystem 的实例）
    """

    def __init__(self, vector_store=None):
        """
        Args:
            vector_store: VectorStore 实例（依赖注入）。
                          为 None 时从 RAGSystem 获取全局单例。
        """
        self._vector_store = vector_store
        self._default_vs = None  # 懒加载的默认 VectorStore

    def _get_vector_store(self):
        """获取 VectorStore 实例（懒加载 + 自动连接）"""
        if self._vector_store:
            return self._vector_store

        if self._default_vs is None:
            from rag_qa.core.vector_store import VectorStore
            self._default_vs = VectorStore()

        if not self._default_vs._connected:
            self._default_vs.connect()

        return self._default_vs

    def retrieve(
        self,
        query: str,
        filters: dict = None,
        top_k_retrieve: int = None,
        top_k_rerank: int = None,
    ) -> list:
        """
        执行完整检索链: encode → search → rerank

        Args:
            query: 检索查询文本
            filters: Milvus 标量过滤条件
                     {"insurer": "平安", "clause_type": "保险责任"}
            top_k_retrieve: Milvus 粗排返回数量（默认 settings.top_k_retrieve=30）
            top_k_rerank: Reranker 精排后保留数量（默认 settings.top_k_rerank=5）

        Returns:
            list[ChunkResult]: 精排后的 Top-K chunks
        """
        vs = self._get_vector_store()

        # ── Step 1: BGE-M3 编码 → Dense + Sparse 向量 ──
        encoder = get_encoder()
        dense_vector, sparse_vector = encoder.encode(query)

        # ── Step 2: Milvus 混合检索（Dense 语义 + Sparse 关键词）──
        chunks = vs.search(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            filters=filters or {},
            top_k=top_k_retrieve or settings.top_k_retrieve,
            include_parent=True,
        )

        if not chunks:
            logger.info("RetrievalInterface: Milvus 检索无结果")
            return []

        # ── Step 3: Cross-Encoder 精排（Top-N → Top-K）──
        reranker = get_reranker()
        reranked = reranker.rerank(
            query=query,
            chunks=chunks,
            top_k=top_k_rerank or settings.top_k_rerank,
        )

        logger.info(
            f"RetrievalInterface: {len(chunks)} → {len(reranked)} chunks"
        )
        return reranked
