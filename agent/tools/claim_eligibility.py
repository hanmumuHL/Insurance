# -*- coding: utf-8 -*-
"""
工具 2: 理赔资格预检 — 检索条款 + 规则引擎判断
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from base.logger import logger


class ClaimEligibilityInput(BaseModel):
    disease_or_event: str = Field(default="", description="疾病名称或保险事件，如 '肺炎住院'")
    product_name: str = Field(default="", description="产品名称")
    insurer: str = Field(default="", description="保险公司名称")


class ClaimEligibilityTool(BaseTool):
    """
    理赔资格预检工具 — 调用 RAG 检索条款 + 规则引擎判断

    流程:
      1. 检索相关条款（保险责任 + 责任免除）
      2. 规则引擎判断: 疾病/事件是否在保障范围内
      3. 返回预检结论 + 条款引用

    注意: 预检结论不是最终理赔决定，仅供参考。
    """
    name: str = "claim_eligibility"
    description: str = (
        "预检某个疾病或事件是否在保险产品的保障范围内。"
        "会返回是否符合赔付条件、相关的条款引用、以及预计的赔付方式。"
        "当用户询问某个情况能否理赔时使用此工具。"
    )
    args_schema: type = ClaimEligibilityInput

    # ── 依赖注入 ──
    vector_store: Optional[object] = None
    mysql_session: Optional[object] = None
    retrieval: Optional[object] = None

    def _run(self, disease_or_event: str = "", product_name: str = "", insurer: str = "") -> str:
        logger.info(f"工具调用: claim_eligibility(event={disease_or_event}, product={product_name})")

        if self.retrieval is None and self.vector_store is None:
            return self._fallback(disease_or_event, product_name, insurer)

        try:
            search_query = f"{product_name} {disease_or_event}"

            if self.retrieval is not None:
                chunks = self.retrieval.retrieve(
                    query=search_query,
                    filters={
                        "insurer": insurer,
                        "product_name": product_name,
                        "clause_type": ["保险责任", "责任免除", "释义"],
                    },
                    top_k_retrieve=10,
                    top_k_rerank=5,
                )
            else:
                from base.encoder import get_encoder
                encoder = get_encoder()
                dense, sparse = encoder.encode(search_query)
                chunks = self.vector_store.search(
                    dense_vector=dense,
                    sparse_vector=sparse,
                    filters={
                        "insurer": insurer,
                        "product_name": product_name,
                        "clause_type": ["保险责任", "责任免除", "释义"],
                    },
                    top_k=10,
                    include_parent=True,
                )

            if not chunks or len(chunks) == 0:
                return (
                    f"理赔资格预检结果:\n"
                    f"- 产品: {insurer} {product_name}\n"
                    f"- 事件: {disease_or_event}\n"
                    f"- 预检结论: ⚠️ 未找到相关条款，无法判断\n"
                    f"- 建议: 联系人工客服核实条款内容"
                )

            # ── 规则引擎判断 ──
            in_coverage = False
            in_exclusion = False
            coverage_citations = []
            exclusion_citations = []

            event_keywords = [disease_or_event]
            if disease_or_event and len(disease_or_event) > 2:
                for i in range(len(disease_or_event) - 1):
                    sub = disease_or_event[i:i+2]
                    if sub not in event_keywords:
                        event_keywords.append(sub)

            for chunk in chunks:
                text = chunk.parent_text or chunk.text
                citation = f"条款[{chunk.product_name} - {chunk.clause_type}]"

                if chunk.clause_type == "保险责任":
                    if any(kw in text for kw in event_keywords):
                        in_coverage = True
                        coverage_citations.append(f"{citation}: {text[:100]}...")
                elif chunk.clause_type == "责任免除":
                    if any(kw in text for kw in event_keywords):
                        in_exclusion = True
                        exclusion_citations.append(f"{citation}: {text[:100]}...")

            if in_exclusion:
                conclusion = "❌ 不符合保障范围（属于责任免除）"
            elif in_coverage:
                conclusion = "✅ 初步判断符合保障范围"
            else:
                conclusion = "⚠️ 无法自动判断，建议人工核实"

            lines = [
                f"理赔资格预检结果:",
                f"- 产品: {insurer} {product_name}",
                f"- 事件: {disease_or_event}",
                f"- 预检结论: {conclusion}",
            ]

            if coverage_citations:
                lines.append("\n📋 相关保障条款:")
                for c in coverage_citations[:3]:
                    lines.append(f"  {c}")

            if exclusion_citations:
                lines.append("\n🚫 相关免责条款:")
                for c in exclusion_citations[:3]:
                    lines.append(f"  {c}")

            lines.append("\n⚠️ 最终理赔结果以保险公司审核为准")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"理赔资格预检失败: {e}", exc_info=True)
            return f"理赔资格预检异常: {e}"

    def _fallback(self, disease_or_event: str, product_name: str, insurer: str) -> str:
        logger.warning("理赔预检: VectorStore 未连接")
        return (
            f"理赔资格预检暂不可用（知识库未连接）。\n"
            f"查询条件: 产品={product_name}, 事件={disease_or_event}\n"
            f"请联系人工客服核实理赔条件。"
        )
