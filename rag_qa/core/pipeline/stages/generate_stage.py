# -*- coding: utf-8 -*-
"""LLM 答案生成 Stage — 组装 prompt + 调用 LLM（~150-200ms）"""

import time
from rag_qa.core.pipeline.stage import Stage
from rag_qa.core.pipeline.context import PipelineContext, StageResult
from base.logger import logger


class GenerateStage(Stage):
    """
    LLM 答案生成 Stage

    委托 RAGSystem 的 _generate_answer() 方法（保留现有生成逻辑不变）。
    仅做 Stage 包装——从 ctx 提取参数，调用生成，写回 ctx。
    """

    name = "generate"

    def __init__(self, rag_system=None, llm_client=None):
        """
        Args:
            rag_system: RAGSystem 实例（用于调用 _generate_answer 等内部方法）
            llm_client: LLMClient 实例（备用，推荐通过 rag_system）
        """
        self._rag = rag_system
        self._llm_client = llm_client

    def can_execute(self, ctx: PipelineContext) -> bool:
        return ctx.complexity >= 1

    def build_messages(self, ctx: PipelineContext) -> list[dict]:
        """
        构建 LLM 调用消息（不实际调用 LLM）

        将 PipelineContext 中的检索结果、KG 上下文等组装为
        OpenAI 格式的消息列表，供同步 chat() 或异步 astream_chat() 使用。
        """
        if self._rag is not None:
            # 委托 RAGSystem 的内部方法组装
            context = self._rag._build_context(ctx.retrieved_chunks)
            system_prompt = self._rag._build_system_prompt(ctx.intent)

            user_parts = [f"参考条款:\n{context}"]
            if ctx.kg_context:
                user_parts.insert(0, f"知识图谱推理:\n{ctx.kg_context}")
            user_parts.append(f"用户问题: {ctx.query}")
            user_message = "\n\n".join(user_parts)

            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
        else:
            # 独立模式：用自身的方法组装
            context = self._build_context(ctx.retrieved_chunks)
            system_prompt = self._build_system_prompt(ctx.intent)

            user_parts = [f"参考条款:\n{context}"]
            if ctx.kg_context:
                user_parts.insert(0, f"知识图谱推理:\n{ctx.kg_context}")
            user_parts.append(f"用户问题: {ctx.query}")
            user_message = "\n\n".join(user_parts)

            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

    def execute(self, ctx: PipelineContext) -> StageResult:
        t0 = time.time()
        try:
            # 构建消息 → 调用 LLM
            messages = self.build_messages(ctx)

            if self._rag is not None:
                answer = self._rag._llm_chat_with_fallback(messages)
            else:
                answer = self._llm_chat_with_fallback(messages)

            timing = round((time.time() - t0) * 1000, 1)
            ctx.pipeline[self.name] = timing
            ctx.generated_answer = answer

            # 提取来源信息
            ctx.sources = self._extract_sources(ctx.retrieved_chunks)

            logger.info(
                f"GenerateStage: answer={len(answer)} chars"
            )
            return StageResult(status="success", timing_ms=timing)

        except Exception as e:
            timing = round((time.time() - t0) * 1000, 1)
            ctx.pipeline[self.name] = timing
            logger.error(f"GenerateStage 失败: {e}")
            ctx.generated_answer = "抱歉，系统暂时繁忙，请稍后再试。"
            return StageResult(status="degraded", timing_ms=timing,
                               data={"warning": str(e)})

    # ============================================================
    # 独立生成（当没有 RAGSystem 实例时）
    # ============================================================

    def _generate_standalone(self, ctx: PipelineContext) -> str:
        """独立生成答案——不依赖 RAGSystem 实例"""
        context = self._build_context(ctx.retrieved_chunks)
        system_prompt = self._build_system_prompt(ctx.intent)

        user_parts = [f"参考条款:\n{context}"]
        if ctx.kg_context:
            user_parts.insert(0, f"知识图谱推理:\n{ctx.kg_context}")
        user_parts.append(f"用户问题: {ctx.query}")
        user_message = "\n\n".join(user_parts)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        return self._llm_chat_with_fallback(messages)

    def _build_context(self, chunks: list) -> str:
        """组装检索条款为 LLM 参考上下文"""
        parts = []
        for i, chunk in enumerate(chunks[:5], 1):
            text = getattr(chunk, "parent_text", None) or getattr(chunk, "text", "")
            product = getattr(chunk, "product_name", "") or ""
            clause = getattr(chunk, "clause_type", "") or ""
            parts.append(f"[条款{i}: {product} - {clause}]\n{text}")
        return "\n\n".join(parts) if parts else "暂无相关条款"

    def _build_system_prompt(self, intent: str) -> str:
        """根据意图构造 system prompt"""
        base = (
            "你是一个专业的保险智能客服。基于提供的条款内容回答用户问题。\n"
            "要求:\n"
            "1. 必须基于条款原文回答，不要编造信息\n"
            "2. 引用条款时标注来源（如 '根据条款第X.X条'）\n"
            "3. 不确定的信息明确告知用户\n"
            "4. 不提供医疗建议\n"
        )
        additions = {
            "条款解读": "回答要严谨准确，逐条引用相关条款。",
            "理赔咨询": "先表达同理心，再清晰说明理赔流程和所需材料。",
            "产品对比": "用表格形式对比，保持中立客观，不贬低任何产品。",
            "退保咨询": "说明退保流程和可能的损失（现金价值 vs 已交保费）。",
            "保费试算": "给出具体金额，注明计算依据和假设条件。",
        }
        return base + "\n" + additions.get(intent, "")

    def _llm_chat_with_fallback(self, messages: list) -> str:
        """LLM 主备切换调用"""
        if self._llm_client is None:
            from base.llm_client import get_llm_client
            self._llm_client = get_llm_client()

        response = self._llm_client.chat(messages=messages)
        if response.error:
            logger.error(f"LLM 调用失败: {response.error}")
            return "抱歉，系统暂时繁忙，请稍后再试。"
        return response.content

    def _extract_sources(self, chunks: list) -> list[dict]:
        """从检索结果提取来源信息"""
        sources = []
        seen = set()
        for chunk in chunks[:5]:
            product = getattr(chunk, "product_name", "") or ""
            clause = getattr(chunk, "clause_type", "") or ""
            chunk_id = getattr(chunk, "chunk_id", "") or ""
            key = f"{product}:{clause}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "product": product,
                    "clause": clause,
                    "chunk_id": chunk_id,
                })
        return sources
