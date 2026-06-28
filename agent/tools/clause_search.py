# -*- coding: utf-8 -*-
"""
工具 3: 条款检索 — 直接调用 VectorStore 检索
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from base.logger import logger


class ClauseSearchInput(BaseModel):
    product_name: str = Field(default="", description="产品名称")
    keywords: str = Field(default="", description="检索关键词，如 '免赔额 住院'")
    clause_type: str = Field(default="", description="条款类型，如 '保险责任'")


class ClauseSearchTool(BaseTool):
    """
    条款检索工具 — 直接调用 VectorStore 检索

    这是 Agent 中最常用的工具，Planner 用它来获取条款信息。
    复用 RAG 系统的 VectorStore，不重复造轮子。
    """
    name: str = "clause_search"
    description: str = (
        "从保险条款知识库中检索相关条款原文。"
        "可以按产品名称和关键词检索，返回条款原文和章节引用。"
        "当用户询问具体的条款内容、保障范围、免责条款时使用此工具。"
    )
    args_schema: type = ClauseSearchInput

    # ── 依赖注入 ──
    vector_store: Optional[object] = None
    retrieval: Optional[object] = None

    def _run(self, product_name: str = "", keywords: str = "", clause_type: str = "") -> str:
        logger.info(f"工具调用: clause_search(product={product_name}, keywords={keywords})")

        if self.retrieval is None and self.vector_store is None:
            return self._fallback(keywords, product_name, clause_type)

        try:
            search_query = f"{product_name} {keywords}".strip()

            filters = {}
            if product_name:
                filters["product_name"] = product_name
            if clause_type:
                filters["clause_type"] = clause_type

            if self.retrieval is not None:
                chunks = self.retrieval.retrieve(
                    query=search_query,
                    filters=filters,
                    top_k_retrieve=10,
                    top_k_rerank=5,
                )
            else:
                chunks = self._retrieve_legacy(search_query, filters)

            if not chunks:
                return (
                    f"条款检索结果 (关键词: {keywords}):\n"
                    f"- 未找到相关条款\n"
                    f"- 产品: {product_name or '不限'}\n"
                    f"- 建议: 确认产品名或换一个更通用的关键词"
                )

            lines = [f"条款检索结果 (关键词: {keywords or '不限'}):"]
            for i, chunk in enumerate(chunks, 1):
                text = chunk.parent_text or chunk.text
                source = f"[条款{i}: {chunk.product_name} - {chunk.clause_type or '其他'}]"
                lines.append(f"\n{source}")
                display_text = text[:300] + ("..." if len(text) > 300 else "")
                lines.append(display_text)

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"条款检索失败: {e}", exc_info=True)
            return f"条款检索异常: {e}"

    def _retrieve_legacy(self, search_query: str, filters: dict) -> list:
        """旧的检索路径（RetrievalInterface 不可用时的降级方案）"""
        from base.encoder import get_encoder
        from base.reranker import get_reranker

        encoder = get_encoder()
        dense, sparse = encoder.encode(search_query)

        chunks = self.vector_store.search(
            dense_vector=dense,
            sparse_vector=sparse,
            filters=filters,
            top_k=10,
            include_parent=True,
        )

        if not chunks:
            return []

        try:
            reranker = get_reranker()
            return reranker.rerank(search_query, chunks, top_k=5)
        except Exception as e:
            logger.warning(f"Reranker 不可用，跳过精排: {e}")
            return chunks[:5]

    def _fallback(self, keywords: str, product_name: str, clause_type: str) -> str:
        logger.warning("条款检索: VectorStore 未连接")
        return (
            f"条款检索暂不可用（知识库未连接）。\n"
            f"检索条件: 产品={product_name or '不限'}, 关键词={keywords or '不限'}\n"
            f"请联系人工客服查询条款信息。"
        )
