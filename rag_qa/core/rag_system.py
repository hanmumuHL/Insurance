# -*- coding: utf-8 -*-
"""
RAG 系统主编排器 — 串联所有 RAG 模块，处理一次完整的问答请求

一次完整请求的流转:
  用户 query
  → QueryResultCache 检查 (命中则秒回)
  → FAQCache 检查 (命中则秒回)
  → PII 脱敏
  → DomainBoundaryGuard 领域守卫 (不相关的直接拒绝)
  → QueryClassifier 三层意图路由
  → 意图被拒绝 → 返回兜底话术
  → StrategySelector 选择检索策略
  → VectorStore Milvus 混合检索
  → BGE-M3 编码 (Dense + Sparse) → Milvus 混合检索
  → Reranker 精排 (Cross-Encoder, Top-30→Top-5)
  → RetrievalQualityGuard 检索质量检查 (质量差则拒绝)
  → LLM 生成答案 (DeepSeek 主 + Qwen 降级)
  → ComplianceGuard 合规检查
  → PII 还原
  → QueryResultCache 写入缓存
  → 返回用户

这个模块不实现具体逻辑，只做编排和异常处理。
每个子模块都是独立可测试的。
"""

import time
from dataclasses import dataclass, field
# from typing import Optional

from base.logger import logger
from config.settings import settings
from base.llm_client import get_llm_client
from base.encoder import get_encoder
from base.reranker import get_reranker

# 导入所有子模块
from cache.redis_client import RedisClient
from cache.faq_cache import FAQCache
from cache.query_result_cache import QueryResultCache
from rag_qa.core.pii_desensitizer import PIIDesensitizer
from rag_qa.core.domain_guard import DomainBoundaryGuard
from rag_qa.core.query_classifier import QueryClassifier
from rag_qa.core.strategy_selector import StrategySelector
from rag_qa.core.vector_store import VectorStore
from rag_qa.core.retrieval_quality_guard import RetrievalQualityGuard
from rag_qa.core.compliance_guard import ComplianceGuard


# ============================================================
# 响应数据结构
# ============================================================


@dataclass
class RAGResponse:
    """
    RAG 系统的完整响应

    包含答案文本和所有中间过程信息（用于调试、日志、监控）

    Attributes:
        answer: 最终返回给用户的答案
        intent: 识别的意图
        strategy: 使用的检索策略
        sources: 引用的条款来源列表 (产品名 + 条款章节)
        cache_hit: 是否命中缓存（FAQ/Query结果）
        latency_ms: 端到端延迟（毫秒）
        pipeline: 各阶段的耗时明细
        error: 如果有错误，记录错误信息
    """

    answer: str
    intent: str = ""
    strategy: str = ""
    sources: list[dict] = field(default_factory=list)
    cache_hit: bool = False
    latency_ms: float = 0.0
    pipeline: dict = field(default_factory=dict)
    error: str = ""


# ============================================================
# RAG 系统主体
# ============================================================


class RAGSystem:
    """
    RAG 智能客服系统 — 主编排器

    职责: 串联所有子模块，处理一次完整的问答请求。
    不包含任何具体的业务逻辑，只做编排和异常兜底。
    """

    def __init__(self):
        """
        初始化所有子模块

        注意: 这里只做对象创建，不做网络连接。
        网络连接（Milvus、Redis 等）在第一次请求时延迟建立，
        这样即使某个服务不可用，系统也能启动（只是对应功能不可用）。
        """
        # ── 缓存层 ──
        self.redis = RedisClient()
        self.faq_cache = FAQCache(self.redis)
        self.query_cache = QueryResultCache(
            self.redis, ttl=settings.query_result_cache_ttl
        )

        # ── 安全层 ──
        self.pii_desensitizer = PIIDesensitizer()
        self.domain_guard = DomainBoundaryGuard()

        # ── 路由层 ──
        # bert_model 和 llm_client 可选传入，None 时跳过对应层级
        self.classifier = QueryClassifier(bert_model=None, llm_client=None)
        self.strategy_selector = StrategySelector()

        # ── 检索层 ──
        self.vector_store = VectorStore()

        # ── 质量检查层 ──
        self.retrieval_guard = RetrievalQualityGuard()
        self.compliance_guard = ComplianceGuard()

        # ── LLM 客户端 (延迟初始化) ──
        self._llm_client = None

        logger.info("RAG 系统初始化完成")

    # ============================================================
    # 主入口: 处理一次完整的问答请求
    # ============================================================

    def query(self, raw_query: str, session_id: str = "") -> RAGResponse:
        """
        处理一次用户查询 — 完整 pipeline

        Args:
            raw_query: 用户原始输入（未脱敏）
            session_id: 会话 ID（用于上下文关联）

        Returns:
            RAGResponse: 包含答案和中间过程信息的完整响应
        """
        start_time = time.time()
        pipeline = {}  # 记录各阶段耗时

        try:
            # ── 阶段 1: FAQ 精确命中 (<1ms) ──
            t0 = time.time()
            faq_answer = self.faq_cache.try_hit(raw_query)
            pipeline["faq_cache"] = round((time.time() - t0) * 1000, 1)
            if faq_answer:
                return RAGResponse(
                    answer=faq_answer,
                    intent="FAQ",
                    cache_hit=True,
                    latency_ms=round((time.time() - start_time) * 1000, 1),
                    pipeline=pipeline,
                )

            # ── 阶段 2: PII 脱敏 (<1ms) ──
            # 在 query 进入任何处理流程之前先脱敏
            # 脱敏后的文本用于后续所有流程
            # mapping 保留，最后用于还原
            t0 = time.time()
            pii_result = self.pii_desensitizer.desensitize(raw_query)
            desensitized_query = pii_result.text
            pii_mapping = pii_result.mapping
            pipeline["pii_desensitize"] = round((time.time() - t0) * 1000, 1)

            # ── 阶段 3: 领域边界检查 (<1ms) ──
            # 不相关的问题直接拒绝，不走后续任何流程
            t0 = time.time()
            guard_result = self.domain_guard.check(desensitized_query)
            pipeline["domain_guard"] = round((time.time() - t0) * 1000, 1)
            if not guard_result.passed:
                logger.info(f"领域守卫拦截: {guard_result.reason}")
                return RAGResponse(
                    answer=guard_result.fallback_response,
                    intent="out_of_domain",
                    latency_ms=round((time.time() - start_time) * 1000, 1),
                    pipeline=pipeline,
                )

            # ── 阶段 4: 三层意图路由 (~5ms) ──
            # 关键词规则 → BERT 分类 → LLM 兜底
            t0 = time.time()
            intent_result = self.classifier.classify(desensitized_query)
            pipeline["classify"] = round((time.time() - t0) * 1000, 1)

            # 意图被拒绝（OOD 或置信度不足）→ 返回兜底话术
            if intent_result.reject:
                logger.info(
                    f"意图拒绝: {intent_result.reject_reason} "
                    f"source={intent_result.source}"
                )
                return RAGResponse(
                    answer=intent_result.fallback_response,
                    intent=intent_result.intent,
                    latency_ms=round((time.time() - start_time) * 1000, 1),
                    pipeline=pipeline,
                )

            intent = intent_result.intent
            entities = intent_result.entities
            logger.info(
                f"意图: {intent} | 置信度: {intent_result.confidence:.2f} "
                f"| 来源: {intent_result.source} | 实体: {entities}"
            )

            # ── 阶段 5: Query 结果缓存检查 (<1ms) ──
            # 已知意图后再查缓存，key 对齐避免 miss
            t0 = time.time()
            cached = self.query_cache.try_get(raw_query, intent)
            pipeline["query_cache"] = round((time.time() - t0) * 1000, 1)
            if cached:
                return RAGResponse(
                    answer=cached.get("answer", ""),
                    intent=cached.get("intent", ""),
                    cache_hit=True,
                    latency_ms=round((time.time() - start_time) * 1000, 1),
                    pipeline=pipeline,
                )

            # ── 阶段 6: 闲聊意图直接 LLM 回复（不走检索）──
            if intent == "闲聊寒暄":
                answer = self._llm_chat(desensitized_query)
                answer = PIIDesensitizer.restore(answer, pii_mapping)
                return RAGResponse(
                    answer=answer,
                    intent=intent,
                    latency_ms=round((time.time() - start_time) * 1000, 1),
                    pipeline=pipeline,
                )

            # ── 阶段 7: 投诉建议 → 转人工（不走检索）──
            if intent == "投诉建议":
                return RAGResponse(
                    answer=(
                        "非常抱歉给您带来不好的体验。我已记录您的反馈，"
                        "正在为您转接人工客服，请稍候。"
                    ),
                    intent=intent,
                    latency_ms=round((time.time() - start_time) * 1000, 1),
                    pipeline=pipeline,
                )

            # ── 阶段 8: 选择检索策略 ──
            t0 = time.time()
            strategy_plan = self.strategy_selector.select(
                intent, desensitized_query, entities
            )
            pipeline["strategy_select"] = round((time.time() - t0) * 1000, 1)
            logger.info(
                f"检索策略: {strategy_plan.strategy.value} | {strategy_plan.reason}"
            )

            # ── 阶段 9: Milvus 混合检索 (~30ms) ──
            t0 = time.time()
            chunks = self._do_retrieval(desensitized_query, strategy_plan)
            pipeline["retrieval"] = round((time.time() - t0) * 1000, 1)

            # ── 阶段 10: 检索质量检查 ──
            t0 = time.time()
            quality = self.retrieval_guard.check(chunks, intent)
            pipeline["quality_check"] = round((time.time() - t0) * 1000, 1)
            if not quality.passed:
                logger.warning(f"检索质量不通过: {quality.reason}")
                return RAGResponse(
                    answer=quality.fallback_response,
                    intent=intent,
                    strategy=strategy_plan.strategy.value,
                    latency_ms=round((time.time() - start_time) * 1000, 1),
                    pipeline=pipeline,
                )

            # ── 阶段 11: LLM 生成答案 (~200ms) ──
            t0 = time.time()
            answer = self._generate_answer(desensitized_query, chunks, intent)
            pipeline["generation"] = round((time.time() - t0) * 1000, 1)

            # ── 阶段 12: 合规检查 ──
            t0 = time.time()
            compliance = self.compliance_guard.check(answer, chunks, intent)
            pipeline["compliance"] = round((time.time() - t0) * 1000, 1)
            if compliance.modified_response:
                answer = compliance.modified_response

            # ── 阶段 13: PII 还原 ──
            # LLM 回复中可能包含 [姓名_001] 等占位符，还原为真实值
            answer = PIIDesensitizer.restore(answer, pii_mapping)

            # ── 阶段 14: 写入缓存 ──
            self.query_cache.set(
                raw_query, intent, {"answer": answer, "intent": intent}
            )

            # ── 构造来源信息（用于前端展示引用）──
            sources = self._extract_sources(chunks)

            total_ms = round((time.time() - start_time) * 1000, 1)
            pipeline["total"] = total_ms
            logger.info(
                f"RAG 完成: {total_ms}ms | 策略={strategy_plan.strategy.value} | pipeline={pipeline}"
            )

            return RAGResponse(
                answer=answer,
                intent=intent,
                strategy=strategy_plan.strategy.value,
                sources=sources,
                latency_ms=total_ms,
                pipeline=pipeline,
            )

        except Exception as e:
            # 全局异常兜底 — 任何未预期的错误都不能让用户看到报错
            total_ms = round((time.time() - start_time) * 1000, 1)
            logger.error(f"RAG 系统异常: {e}", exc_info=True)
            return RAGResponse(
                answer="抱歉，系统暂时繁忙，请稍后再试或联系人工客服。",
                error=str(e),
                latency_ms=total_ms,
                pipeline=pipeline,
            )

    # ============================================================
    # 内部方法: 检索
    # ============================================================

    def _do_retrieval(self, query: str, plan) -> list:
        """
        根据策略计划执行检索

        目前实现了直接检索和条件检索。
        HyDE、子查询拆分、对比检索需要 LLM 配合，在完整版中实现。

        Args:
            query: 脱敏后的 query
            plan: StrategyPlan 策略计划

        Returns:
            ChunkResult 列表（已按 RRF 分数排序）
        """
        # 确保 Milvus 已连接
        if not self.vector_store._connected:
            self.vector_store.connect()

        # ── BGE-M3 编码: query → Dense + Sparse 向量 ──
        encoder = get_encoder()
        dense_vector, sparse_vector = encoder.encode(query)

        # 执行混合检索
        chunks = self.vector_store.search(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,  # BGE-M3 词汇权重（BM25 式关键词检索）
            filters=plan.filters,
            top_k=settings.top_k_retrieve,
            include_parent=True,  # 启用 Parent-Child 回查
        )

        # ── Reranker 精排: Top-30 → Top-5 ──
        # 用 Cross-Encoder 逐 query-chunk 对计算真实相关度
        # 精度远高于 Milvus 粗排的余弦相似度
        reranker = get_reranker()
        chunks = reranker.rerank(
            query=query,
            chunks=chunks,
            top_k=settings.top_k_rerank,  # 默认 5
        )

        return chunks

    # ============================================================
    # 内部方法: LLM 调用
    # ============================================================

    def _generate_answer(self, query: str, chunks: list, intent: str) -> str:
        """
        调用 LLM 生成答案 — 主备切换模式

        流程:
          1. 组装 prompt (system + context + user query)
          2. 调用 DeepSeek API (主)
          3. 如果 DeepSeek 超时/报错 → 降级到千问 API
          4. 如果千问也失败 → 返回兜底话术

        Args:
            query: 脱敏后的用户 query
            chunks: 检索到的条款 chunks
            intent: 意图类型

        Returns:
            LLM 生成的答案文本
        """
        # ── 组装上下文 ──
        context = self._build_context(chunks)

        # ── 组装 system prompt ──
        system_prompt = self._build_system_prompt(intent)

        # ── 调用 LLM (主备切换) ──
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"参考条款:\n{context}\n\n用户问题: {query}"},
        ]

        answer = self._llm_chat_with_fallback(messages)
        return answer

    def _build_context(self, chunks: list) -> str:
        """
        将检索到的 chunks 组装成 LLM 的参考上下文

        优先使用 parent_text（父块，上下文完整），
        如果 parent_text 为空则用子块 text。
        每个 chunk 标注来源，方便 LLM 引用。
        """
        parts = []
        for i, chunk in enumerate(chunks[:5], 1):  # Top-5
            text = chunk.parent_text or chunk.text
            source = f"[条款{i}: {chunk.product_name} - {chunk.clause_type}]"
            parts.append(f"{source}\n{text}")
        return "\n\n".join(parts)

    def _build_system_prompt(self, intent: str) -> str:
        """
        根据意图类型构造不同的 system prompt

        不同意图的回答风格不同:
          条款解读 → 严谨，必须引用条款原文
          理赔咨询 → 同理心 + 准确
          产品对比 → 中立客观，表格化
        """
        base = (
            "你是一个专业的保险智能客服。基于提供的条款内容回答用户问题。\n"
            "要求:\n"
            "1. 必须基于条款原文回答，不要编造信息\n"
            "2. 引用条款时标注来源（如 '根据条款第X.X条'）\n"
            "3. 不确定的信息明确告知用户\n"
            "4. 不提供医疗建议\n"
        )

        intent_additions = {
            "条款解读": "回答要严谨准确，逐条引用相关条款。",
            "理赔咨询": "先表达同理心，再清晰说明理赔流程和所需材料。",
            "产品对比": "用表格形式对比，保持中立客观，不贬低任何产品。",
            "退保咨询": "说明退保流程和可能的损失（现金价值 vs 已交保费）。",
            "保费试算": "给出具体金额，注明计算依据和假设条件。",
        }

        return base + "\n" + intent_additions.get(intent, "")

    def _llm_chat(self, query: str) -> str:
        """
        闲聊场景的 LLM 直答（无检索上下文）

        用 LLM 直接回复闲聊，不走 RAG 检索流程。
        system prompt 引导用户回到保险话题。
        """
        messages = [
            {
                "role": "system",
                "content": "你是保险智能客服，友好地回应用户的闲聊，并引导到保险话题。回复简洁，不超过100字。",
            },
            {"role": "user", "content": query},
        ]
        return self._llm_chat_with_fallback(messages)

    def _llm_chat_with_fallback(self, messages: list) -> str:
        """
        LLM 调用 — 通过 LLMClient 实现主备切换

        内部流程 (由 LLMClient 管理):
          1. 尝试 DeepSeek API (主)
          2. DeepSeek 失败 → 降级到千问 API
          3. 千问也失败 → 重试一次 DeepSeek
          4. 全部失败 → 返回兜底话术

        熔断机制:
          DeepSeek 连续 5 次失败 → 熔断 30 秒
          熔断期间直接走千问，不等超时

        Args:
            messages: OpenAI 格式的消息列表

        Returns:
            LLM 生成的回复文本
        """
        client = get_llm_client()
        response = client.chat(messages=messages)

        if response.error:
            logger.error(f"LLM 调用最终失败: {response.error}")
            return "抱歉，系统暂时繁忙，请稍后再试。"

        # 记录使用的通道（用于监控降级率）
        if response.is_fallback:
            logger.warning(f"LLM 降级到 {response.provider}/{response.model}")

        logger.info(
            f"LLM 调用完成: provider={response.provider}, "
            f"model={response.model}, "
            f"tokens={response.tokens_used.get('total', '?')}, "
            f"latency={response.latency_ms:.0f}ms"
        )

        return response.content

    # ============================================================
    # 内部方法: 辅助
    # ============================================================

    def _extract_sources(self, chunks: list) -> list[dict]:
        """
        从检索结果中提取来源信息（用于前端展示引用标记）

        返回格式:
          [{"product": "平安e生保", "clause": "保险责任", "chunk_id": "xxx"}]
        """
        sources = []
        seen = set()
        for chunk in chunks[:5]:
            key = f"{chunk.product_name}:{chunk.clause_type}"
            if key not in seen:
                seen.add(key)
                sources.append(
                    {
                        "product": chunk.product_name,
                        "clause": chunk.clause_type,
                        "chunk_id": chunk.chunk_id,
                    }
                )
        return sources
