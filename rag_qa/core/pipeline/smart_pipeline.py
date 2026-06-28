# -*- coding: utf-8 -*-
"""
SmartPipeline — 智能管道主编排器

替代 RAGSystem.query() 的 12 步硬编码流程，提供:
  - 复杂度路由: 0=FAQ/闲聊, 1=简单咨询, 2=复杂业务
  - 按需增强: 简单查询不经过 KG/Agent, 复杂查询动态叠加
  - Stage 接口化: 每个阶段可独立测试
  - 独立降级: KG 不可用不影响主流程

使用方式:
    from rag_qa.core.pipeline.smart_pipeline import SmartPipeline

    pipeline = SmartPipeline()
    response = pipeline.run("肺炎住院能赔吗", user_role="customer")
"""

import time
import asyncio
from typing import AsyncGenerator
from dataclasses import dataclass, field

from rag_qa.core.pipeline.context import PipelineContext
from rag_qa.core.pipeline.stage import Stage
from rag_qa.core.pipeline.query_analyzer import QueryAnalyzer
from rag_qa.core.pipeline.stages import (
    FAQStage,
    DomainGuardStage,
    ClassifyStage,
    CacheStage,
    ChitchatStage,
    StrategyStage,
    KGStage,
    RetrievalStage,
    QualityStage,
    GenerateStage,
    ComplianceStage,
    CacheWriteStage,
)
from base.logger import logger
from rag_qa.core.rag_system import RAGResponse

# ── 管道定义: complexity → stage 名称列表 ──
# complexity=0: FAQ/闲聊/投诉 → FAQStage 直接返回
# complexity=1: 简单咨询 → 轻量管道（分类→缓存→检索→生成→合规）
# complexity=2: 复杂业务 → 完整管道（分类→KG→策略→检索→质量→生成→合规→缓存写入）

PIPELINE_DEFINITIONS = {
    0: ["faq", "chitchat"],  # FAQ 精确命中立即返回，未命中则走闲聊/投诉处理
    1: [  # 轻量管道（~200ms）
        "classify",
        "cache",
        "retrieval",
        "generate",
        "compliance",
    ],
    2: [  # 完整管道（~500ms）
        "classify",
        "cache",
        "kg",
        "strategy",
        "retrieval",
        "quality",
        "generate",
        "compliance",
        "cache_write",
    ],
}


# ── 流式事件: astream() 的 yield 类型 ──


@dataclass
class StreamEvent:
    """
    SmartPipeline.astream() 异步生成器 yield 的事件

    type 取值:
      - "status":  处理进度通知  (data: {stage, message})
      - "answer":  token 级答案片段 (data: {text})
      - "replace": 合规审查修正了答案，替换全文 (data: {text})
      - "sources": 引用来源 (data: {sources})
      - "done":    流式完成 (data: {latency_ms, intent, strategy, cache_hit})
      - "error":   流式异常 (data: {error})
    """

    type: str = ""
    data: dict = field(default_factory=dict)


class SmartPipeline:
    """
    智能管道主编排器

    按复杂度路由请求到不同的 Stage 组合:
      complexity=0: FAQ → 直接返回（~1ms）
      complexity=1: Classify→Cache→Retrieve→Generate→Compliance（~200ms）
      complexity=2: Classify→Cache→KG→Strategy→Retrieve→Quality→Generate→Compliance→CacheWrite（~500ms）

    特性:
      - KG Stage 仅在 complexity>=2 且 entities 非空时执行
      - 所有 Stage 失败/降级都记录在 ctx.pipeline 中
      - 返回 RAGResponse（与当前 RAGSystem.query() 完全兼容）
    """

    def __init__(self, rag_system=None):
        """
        Args:
            rag_system: RAGSystem 实例（可选）。
                       传入时 GenerateStage 可复用 RAGSystem 的 _generate_answer()。
                       不传入时 GenerateStage 独立工作。
        """
        self._rag = rag_system

        # ── 核心服务（共享实例）──
        self._query_analyzer = QueryAnalyzer()

        # ── 创建所有 Stage ──
        # 从 rag_system 提取子组件（如果可用）
        vs = getattr(rag_system, "vector_store", None) if rag_system else None

        from rag_qa.core.pipeline.retrieval_interface import RetrievalInterface

        self._retrieval_interface = RetrievalInterface(vector_store=vs)

        from rag_qa.core.kg.service import KGService

        self._kg_service = KGService()

        self._stages: dict[str, Stage] = {
            "faq": FAQStage(
                faq_cache=getattr(rag_system, "faq_cache", None) if rag_system else None
            ),
            "domain_guard": DomainGuardStage(
                guard=getattr(rag_system, "domain_guard", None) if rag_system else None
            ),
            "classify": ClassifyStage(
                classifier=(
                    getattr(rag_system, "classifier", None) if rag_system else None
                )
            ),
            "cache": CacheStage(
                query_cache=(
                    getattr(rag_system, "query_cache", None) if rag_system else None
                )
            ),
            "chitchat": ChitchatStage(),
            "strategy": StrategyStage(
                strategy_selector=(
                    getattr(rag_system, "strategy_selector", None)
                    if rag_system
                    else None
                )
            ),
            "kg": KGStage(kg_service=self._kg_service),
            "retrieval": RetrievalStage(retrieval_interface=self._retrieval_interface),
            "quality": QualityStage(
                quality_guard=(
                    getattr(rag_system, "retrieval_guard", None) if rag_system else None
                )
            ),
            "generate": GenerateStage(rag_system=rag_system),
            "compliance": ComplianceStage(
                compliance_guard=(
                    getattr(rag_system, "compliance_guard", None)
                    if rag_system
                    else None
                )
            ),
            "cache_write": CacheWriteStage(
                query_cache=(
                    getattr(rag_system, "query_cache", None) if rag_system else None
                )
            ),
        }

        logger.info(
            f"SmartPipeline 初始化完成 — " f"已注册 {len(self._stages)} 个 Stage"
        )

    # ============================================================
    # 预生成阶段: 查询分析 + 在 generate 之前的所有 Stage
    # ============================================================

    def _init_and_pre_generate(
        self,
        query: str,
        session_id: str,
        user_role: str,
    ) -> tuple[PipelineContext, RAGResponse] | tuple[PipelineContext, None]:
        """
        执行 generate 之前的所有阶段

        包括: QueryAnalyzer → FAQ/Domain → Classify → Cache →
              KG → Strategy → Retrieval → Quality

        返回:
          (ctx, early_exit_response_or_None)

          ctx 是构建好的 PipelineContext（内含 intent, chunks, kg_context 等）。
          如果 early_exit_response 非 None，表示管道短路（FAQ/领域拦截等），
          调用方应直接返回该响应，不再执行生成。
        """
        from rag_qa.core.rag_system import RAGResponse

        start_time = time.time()

        ctx = PipelineContext(
            query=query,
            session_id=session_id,
            user_role=user_role,
            start_time=start_time,
        )

        # ── Step 1: 查询分析 ──
        analysis = self._query_analyzer.analyze(query, session_id, user_role)

        # FAQ 命中 → 立即返回
        if analysis.complexity == 0 and analysis.faq_answer:
            total_ms = round((time.time() - start_time) * 1000, 1)
            return ctx, RAGResponse(
                answer=analysis.faq_answer,
                intent="FAQ",
                cache_hit=True,
                latency_ms=total_ms,
                pipeline={"faq": total_ms, "total": total_ms},
            )

        # 领域守卫拦截
        if analysis.domain_blocked:
            total_ms = round((time.time() - start_time) * 1000, 1)
            return ctx, RAGResponse(
                answer=analysis.fallback_response,
                intent="out_of_domain",
                latency_ms=total_ms,
                pipeline={"domain_guard": total_ms, "total": total_ms},
            )

        # 设置上下文
        ctx.complexity = analysis.complexity
        ctx.intent = analysis.intent
        ctx.entities = analysis.entities

        # 闲聊/投诉 → complexity=0
        if analysis.intent in ("闲聊寒暄", "投诉建议"):
            ctx.complexity = 0

        # 意图被拒绝 → 返回兜底话术
        if analysis.fallback_response and not analysis.faq_answer:
            total_ms = round((time.time() - start_time) * 1000, 1)
            return ctx, RAGResponse(
                answer=analysis.fallback_response,
                intent=analysis.intent,
                latency_ms=total_ms,
                pipeline={"classify": total_ms, "total": total_ms},
            )

        logger.info(
            f"SmartPipeline: complexity={ctx.complexity} "
            f"intent={ctx.intent} query='{query[:50]}'"
        )

        # ── Step 2: 选择管道 ──
        stage_names = PIPELINE_DEFINITIONS.get(ctx.complexity, PIPELINE_DEFINITIONS[1])

        # ── Step 3: 执行 generate 之前的 Stage ──
        for name in stage_names:
            if name == "generate":
                break  # 到此为止，不执行 generate

            stage = self._stages.get(name)
            if stage is None:
                logger.warning(f"SmartPipeline: 未知 Stage '{name}'")
                continue

            if not stage.can_execute(ctx):
                logger.debug(f"SmartPipeline: 跳过 {name} (can_execute=False)")
                continue

            result = stage.execute(ctx)

            if result.status == "failed":
                logger.error(f"SmartPipeline: {name} 失败 — {result.error}")
                total_ms = round((time.time() - start_time) * 1000, 1)
                ctx.pipeline["total"] = total_ms
                return ctx, RAGResponse(
                    answer="抱歉，系统暂时繁忙，请稍后再试或联系人工客服。",
                    intent=ctx.intent,
                    error=result.error,
                    latency_ms=total_ms,
                    pipeline=ctx.pipeline,
                )

            if result.status == "skip":
                logger.info(
                    f"SmartPipeline: {name} 短路 — "
                    f"{result.data.get('reason', result.data)}"
                )
                if ctx.generated_answer:
                    total_ms = round((time.time() - start_time) * 1000, 1)
                    ctx.pipeline["total"] = total_ms
                    return ctx, RAGResponse(
                        answer=ctx.generated_answer,
                        intent=ctx.intent,
                        strategy=getattr(ctx.strategy_plan, "strategy", None),
                        sources=ctx.sources,
                        cache_hit=ctx.cache_hit,
                        latency_ms=total_ms,
                        pipeline=ctx.pipeline,
                    )

        return ctx, None  # None → 调用方应继续生成

    # ============================================================
    # 主入口
    # ============================================================

    def run(
        self,
        query: str,
        session_id: str = "",
        user_role: str = "agent",
        on_stage_complete: callable = None,
    ) -> "RAGResponse":
        """
        处理一次完整的问答请求

        Args:
            query: 用户查询
            session_id: 会话 ID
            user_role: 用户角色 (customer / agent / underwriter / admin)
            on_stage_complete: 可选回调 — 每完成一个 Stage 时调用
                              签名为 on_stage_complete(stage_name: str, result: StageResult)

        Returns:
            RAGResponse: 兼容当前 gateway 的响应格式
        """
        from rag_qa.core.rag_system import RAGResponse

        start_time = time.time()

        try:
            # ── Phase 1: 预生成阶段 ──
            ctx, early_exit = self._init_and_pre_generate(query, session_id, user_role)
            if early_exit is not None:
                return early_exit

            # ── Phase 2: 执行 generate 及之后的 Stage ──
            all_names = PIPELINE_DEFINITIONS.get(
                ctx.complexity, PIPELINE_DEFINITIONS[1]
            )

            if "generate" in all_names:
                # 从 generate 开始（_init_and_pre_generate 已执行之前的 stage）
                gen_idx = all_names.index("generate")
                post_gen_names = all_names[gen_idx:]

                for name in post_gen_names:
                    stage = self._stages.get(name)
                    if stage is None:
                        logger.warning(f"SmartPipeline: 未知 Stage '{name}'")
                        continue

                    if not stage.can_execute(ctx):
                        logger.debug(f"SmartPipeline: 跳过 {name} (can_execute=False)")
                        continue

                    result = stage.execute(ctx)

                    if on_stage_complete:
                        try:
                            on_stage_complete(stage.name, result)
                        except Exception:
                            pass

                    if result.status == "failed":
                        logger.error(f"SmartPipeline: {name} 失败 — {result.error}")
                        total_ms = round((time.time() - start_time) * 1000, 1)
                        ctx.pipeline["total"] = total_ms
                        return RAGResponse(
                            answer="抱歉，系统暂时繁忙，请稍后再试或联系人工客服。",
                            intent=ctx.intent,
                            error=result.error,
                            latency_ms=total_ms,
                            pipeline=ctx.pipeline,
                        )

                    if result.status == "skip" and ctx.generated_answer:
                        total_ms = round((time.time() - start_time) * 1000, 1)
                        ctx.pipeline["total"] = total_ms
                        return RAGResponse(
                            answer=ctx.generated_answer,
                            intent=ctx.intent,
                            strategy=getattr(ctx.strategy_plan, "strategy", None),
                            sources=ctx.sources,
                            cache_hit=ctx.cache_hit,
                            latency_ms=total_ms,
                            pipeline=ctx.pipeline,
                        )

            # ── 构建响应 ──
            total_ms = round((time.time() - start_time) * 1000, 1)
            ctx.pipeline["total"] = total_ms

            # ── 兜底保护: 如果走到这里仍无答案，返回兜底话术 ──
            if not ctx.generated_answer:
                logger.warning(
                    f"SmartPipeline: 无答案生成 (complexity={ctx.complexity} "
                    f"intent={ctx.intent})，使用兜底话术"
                )
                ctx.generated_answer = (
                    "您好！我是保险智能客服，有什么可以帮您的吗？"
                    if ctx.intent in ("闲聊寒暄",)
                    else "抱歉，我暂时无法处理您的问题，正在为您转接人工客服，请稍候。"
                    if ctx.intent in ("投诉建议",)
                    else "抱歉，系统暂时繁忙，请稍后再试或联系人工客服。"
                )

            strategy = ""
            if ctx.strategy_plan is not None:
                try:
                    strategy = ctx.strategy_plan.strategy.value
                except Exception:
                    strategy = str(ctx.strategy_plan.strategy)

            logger.info(
                f"SmartPipeline 完成: {total_ms}ms "
                f"complexity={ctx.complexity} pipeline={ctx.pipeline}"
            )

            return RAGResponse(
                answer=ctx.generated_answer,
                intent=ctx.intent,
                strategy=strategy,
                sources=ctx.sources,
                cache_hit=ctx.cache_hit,
                latency_ms=total_ms,
                pipeline=ctx.pipeline,
            )

        except Exception as e:
            total_ms = round((time.time() - start_time) * 1000, 1)
            logger.error(f"SmartPipeline 异常: {e}", exc_info=True)
            return RAGResponse(
                answer="抱歉，系统暂时繁忙，请稍后再试或联系人工客服。",
                error=str(e),
                latency_ms=total_ms,
                pipeline=getattr(ctx, "pipeline", {}) if "ctx" in dir() else {},
            )

    # ============================================================
    # 异步流式入口: token 级 SSE 推送
    # ============================================================

    async def astream(
        self,
        query: str,
        session_id: str = "",
        user_role: str = "agent",
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        处理一次问答请求，逐 token 流式 yield StreamEvent

        流程:
          Phase A: 预生成阶段 (asyncio.to_thread) → status 事件
          Phase B: LLM 流式生成 (async for) → answer 事件
          Phase C: 合规审查 (asyncio.to_thread) → 如需修正则 replace 事件
          Phase D: 缓存写入 (fire-and-forget)
          Phase E: sources + done 事件

        Args:
            query:      用户查询
            session_id: 会话 ID
            user_role:  用户角色

        Yields:
            StreamEvent: 流式事件（status/answer/replace/sources/done/error）
        """
        start_time = time.time()

        try:
            # ── Phase A: 预生成阶段（同步，放到线程池）──
            ctx, early_exit = await asyncio.to_thread(
                self._init_and_pre_generate, query, session_id, user_role
            )

            if early_exit is not None:
                # 短路响应（FAQ/领域拦截等）
                yield StreamEvent(
                    "status",
                    {
                        "stage": "complete",
                        "message": "已找到答案",
                    },
                )
                # 按句拆分推送（短路响应通常较短）
                for sentence in early_exit.answer.split("。"):
                    if sentence.strip():
                        yield StreamEvent(
                            "answer",
                            {
                                "text": sentence.strip() + "。",
                            },
                        )
                if early_exit.sources:
                    yield StreamEvent(
                        "sources",
                        {
                            "sources": early_exit.sources,
                        },
                    )
                yield StreamEvent(
                    "done",
                    {
                        "latency_ms": early_exit.latency_ms,
                        "intent": early_exit.intent,
                        "strategy": getattr(early_exit, "strategy", ""),
                        "cache_hit": early_exit.cache_hit,
                    },
                )
                return

            # 推送检索状态
            yield StreamEvent(
                "status",
                {
                    "stage": "searching",
                    "message": f"检索完成，找到 {len(ctx.retrieved_chunks)} 条相关条款",
                },
            )

            # ── Phase B: 流式 LLM 生成 ──
            generate_stage = self._stages.get("generate")
            if generate_stage is None or not generate_stage.can_execute(ctx):
                yield StreamEvent(
                    "error",
                    {
                        "error": "无法执行生成阶段",
                    },
                )
                return

            messages = generate_stage.build_messages(ctx)

            from base.llm_client import get_llm_client, LLMStreamError

            llm_client = get_llm_client()

            full_answer = ""
            try:
                async for token in llm_client.astream_chat(messages=messages):
                    if token:
                        full_answer += token
                        yield StreamEvent("answer", {"text": token})
            except LLMStreamError as e:
                if e.partial_content:
                    full_answer = e.partial_content
                    yield StreamEvent(
                        "status",
                        {
                            "stage": "warning",
                            "message": "流式输出中断，使用已生成的部分结果",
                        },
                    )
                else:
                    yield StreamEvent(
                        "error",
                        {
                            "error": "生成失败，请稍后再试",
                        },
                    )
                    return

            generate_timing = round((time.time() - start_time) * 1000, 1)
            ctx.pipeline["generate"] = generate_timing
            ctx.generated_answer = full_answer
            ctx.sources = generate_stage._extract_sources(ctx.retrieved_chunks)

            # ── Phase C: 合规审查（同步，放到线程池）──
            compliance_stage = self._stages.get("compliance")
            if compliance_stage and compliance_stage.can_execute(ctx):
                original_answer = full_answer
                compliance_result = await asyncio.to_thread(
                    compliance_stage.execute, ctx
                )
                ctx.pipeline["compliance"] = compliance_result.timing_ms

                # 合规修正了答案 → 推送 replace 事件
                if ctx.generated_answer != original_answer:
                    yield StreamEvent(
                        "replace",
                        {
                            "text": ctx.generated_answer,
                        },
                    )

            # ── Phase D: 缓存写入（fire-and-forget）──
            cache_write_stage = self._stages.get("cache_write")
            if cache_write_stage and cache_write_stage.can_execute(ctx):
                try:
                    asyncio.create_task(
                        asyncio.to_thread(cache_write_stage.execute, ctx)
                    )
                except Exception:
                    pass

            # ── Phase E: 完成 ──
            if ctx.sources:
                yield StreamEvent("sources", {"sources": ctx.sources})

            total_ms = round((time.time() - start_time) * 1000, 1)
            ctx.pipeline["total"] = total_ms

            strategy = ""
            if ctx.strategy_plan is not None:
                try:
                    strategy = ctx.strategy_plan.strategy.value
                except Exception:
                    strategy = str(ctx.strategy_plan.strategy)

            yield StreamEvent(
                "done",
                {
                    "latency_ms": total_ms,
                    "intent": ctx.intent,
                    "strategy": strategy,
                    "cache_hit": ctx.cache_hit,
                },
            )

        except Exception as e:
            logger.error(f"SmartPipeline.astream 异常: {e}", exc_info=True)
            yield StreamEvent(
                "error",
                {
                    "error": "处理失败，请稍后再试",
                },
            )
