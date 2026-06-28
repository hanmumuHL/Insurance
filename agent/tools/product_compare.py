# -*- coding: utf-8 -*-
"""
工具 5: 多产品对比 — 检索各产品条款后交叉对比
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from base.logger import logger


class ProductCompareInput(BaseModel):
    products: str = Field(default="", description="要对比的产品列表，逗号分隔")
    dimensions: str = Field(default="保障范围,保费,免赔额", description="对比维度")


class ProductCompareTool(BaseTool):
    """
    多产品对比工具 — 调用 ClauseSearchTool 分别检索各产品后交叉对比
    """
    name: str = "product_compare"
    description: str = (
        "对比多个保险产品。可以按保障范围、保费、免赔额、"
        "等待期等维度进行对比。返回结构化的对比结果。"
        "当用户要求对比两个或多个保险产品时使用此工具。"
    )
    args_schema: type = ProductCompareInput

    # ── 依赖注入 ──
    vector_store: Optional[object] = None
    mysql_session: Optional[object] = None
    llm_client: Optional[object] = None
    retrieval: Optional[object] = None

    def _run(self, products: str = "", dimensions: str = "保障范围,保费,免赔额") -> str:
        logger.info(f"工具调用: product_compare(products={products})")

        product_list = [p.strip() for p in products.split(",") if p.strip()]

        if len(product_list) < 2:
            return "对比至少需要 2 个产品，请提供更多产品名称。"

        if self.retrieval is None and self.vector_store is None:
            return self._fallback(products, dimensions)

        try:
            dim_list = [d.strip() for d in dimensions.split(",")]

            product_results = {}
            for product in product_list:
                results = []
                for dim in dim_list:
                    search_query = f"{product} {dim}"

                    if self.retrieval is not None:
                        chunks = self.retrieval.retrieve(
                            query=search_query,
                            filters={"product_name": product},
                            top_k_retrieve=5,
                            top_k_rerank=3,
                        )
                        lines = [f"条款检索结果 (关键词: {dim}):"]
                        for j, chunk in enumerate(chunks, 1):
                            text = getattr(chunk, 'parent_text', None) or getattr(chunk, 'text', '')
                            clause = getattr(chunk, 'clause_type', '其他') or '其他'
                            lines.append(
                                f"[条款{j}: {product} - {clause}]\n"
                                f"{text[:300]}"
                            )
                        results.append("\n".join(lines))
                    else:
                        from agent.tools.clause_search import ClauseSearchTool
                        searcher = ClauseSearchTool(vector_store=self.vector_store)
                        result = searcher._run(
                            product_name=product,
                            keywords=dim,
                        )
                        results.append(result)

                product_results[product] = results

            # ── LLM 汇总对比 ──
            if self.llm_client:
                summary_prompt = self._build_compare_prompt(
                    product_list, dim_list, product_results
                )
                response = self.llm_client.chat(
                    messages=[{"role": "user", "content": summary_prompt}],
                    temperature=0.3,
                    max_tokens=1024,
                )
                if not response.error:
                    return response.content

            # ── 没有 LLM → 拼接原始结果 ──
            lines = ["产品对比结果:\n"]
            lines.append(f"对比维度: {dimensions}\n")
            for product in product_list:
                lines.append(f"## {product}")
                for result in product_results.get(product, []):
                    lines.append(result[:200])
                lines.append("")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"产品对比失败: {e}", exc_info=True)
            return f"产品对比异常: {e}"

    def _fallback(self, products: str, dimensions: str) -> str:
        logger.warning("产品对比: VectorStore 未连接")
        return (
            f"产品对比功能暂不可用（知识库未连接）。\n"
            f"对比产品: {products}\n"
            f"请联系人工客服进行产品对比。"
        )

    def _build_compare_prompt(self, products, dims, results) -> str:
        parts = [f"请对比以下 {len(products)} 个保险产品:"]
        parts.append(f"对比维度: {', '.join(dims)}\n")
        for product in products:
            parts.append(f"--- {product} ---")
            for r in results.get(product, []):
                parts.append(r[:500])
            parts.append("")
        parts.append("请用表格形式给出对比结论，保持中立客观。")
        return "\n".join(parts)
