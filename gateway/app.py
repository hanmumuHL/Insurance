# -*- coding: utf-8 -*-
"""
FastAPI 网关 — 双通道 API 层入口

双通道架构:
  - CUSTOMER 通道: 外部客户 → RAG Pipeline（轻量、快速、只看自己数据、严格合规）
  - AGENT 通道:   内部人员 → Multi-Agent Orchestrator（完整编排、全量数据、宽松合规）

提供四个端点:
  1. POST /chat         — 同步问答（按角色自动分流）
  2. GET  /chat/stream   — SSE 流式问答（按角色自动分流）
  3. POST /admin/ingest  — 文档摄取（仅 ADMIN）
  4. GET  /health        — 健康检查

SSE (Server-Sent Events) 流式返回:
  用户提问后，服务端一边处理一边推送中间状态:
    event: status   → 处理进度
    event: answer   → 答案片段
    event: sources  → 引用来源
    event: done     → 完成信号
"""

import json
import asyncio
from fastapi import FastAPI, HTTPException, Request, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from typing import Optional

from base.logger import logger
from rag_qa.core.rag_system import RAGSystem

# ── 多 Agent 架构 ──
from agent.sub_agents import (
    InsuranceAgent, UnderwritingAgent, ClaimAgent, ServiceAgent,
)
from agent.orchestrator import Orchestrator
from agent.memory import get_memory_manager

# ── 双通道模型与认证 ──
from gateway.models import ChatRequest, ChatResponse, IngestRequest
from gateway.auth import (
    UserContext, UserRole, INTERNAL_ROLES, ADMIN_ROLES,
    get_current_user,
)


# ============================================================
# FastAPI 应用初始化
# ============================================================

app = FastAPI(
    title="保险智能客服 API — 双通道架构",
    description=(
        "RAG 智能客服 + 多 Agent 协作 (Orchestrator + 4 领域 Agent)\n\n"
        "双通道模式:\n"
        "  - Customer 通道: 外部客户自助问答 (RAG Pipeline)\n"
        "  - Agent 通道:   内部人员 AI 助手 (Multi-Agent Orchestrator)"
    ),
    version="4.0.0",
)

# 跨域配置 (开发环境允许所有来源，生产环境应该限制)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 初始化 RAG 系统 (单例) ──
rag_system = RAGSystem()

# ── 多 Agent 系统 (懒初始化) ──
_orchestrator = None  # type: Optional[Orchestrator]


def init_multi_agent():
    """
    初始化多 Agent 系统（含短记忆 checkpointer）

    在 startup 或首次请求时懒加载，失败则优雅降级。
    """
    global _orchestrator

    if _orchestrator is not None:
        return _orchestrator

    try:
        from base.llm_client import get_llm_client

        llm_client = get_llm_client()

        vector_store = None
        mysql_session = None
        redis_session = None

        try:
            vector_store = rag_system.vector_store
        except Exception:
            pass

        try:
            from cache.redis_client import get_redis_client
            redis_session = get_redis_client()
        except Exception:
            pass

        try:
            from base.database import get_mysql_session
            mysql_session = get_mysql_session()
        except Exception:
            pass

        # ── 短记忆: RedisSaver checkpointer ──
        mm = get_memory_manager()
        checkpointer = mm.get_checkpointer()

        # ── 创建四个子 Agent ──
        agents = {
            "insurance": InsuranceAgent(
                vector_store=vector_store,
                mysql_session=mysql_session,
                redis_session=redis_session,
                llm_client=llm_client,
                checkpointer=checkpointer,
            ),
            "underwriting": UnderwritingAgent(
                vector_store=vector_store,
                mysql_session=mysql_session,
                redis_session=redis_session,
                llm_client=llm_client,
                checkpointer=checkpointer,
            ),
            "claim": ClaimAgent(
                vector_store=vector_store,
                mysql_session=mysql_session,
                redis_session=redis_session,
                llm_client=llm_client,
                checkpointer=checkpointer,
            ),
            "service": ServiceAgent(
                vector_store=vector_store,
                mysql_session=mysql_session,
                redis_session=redis_session,
                llm_client=llm_client,
                checkpointer=checkpointer,
            ),
        }

        _orchestrator = Orchestrator(agents=agents, rag_system=rag_system)

        logger.info(
            f"多 Agent 系统初始化完成 — "
            f"已注册: {list(agents.keys())} | "
            f"checkpointer: {'RedisSaver' if checkpointer else '无状态'} | "
            f"RAG 降级: {'可用' if rag_system else '不可用'}"
        )

        return _orchestrator

    except Exception as e:
        logger.warning(
            f"多 Agent 初始化失败 (RAG 模式仍可用): {e}",
            exc_info=True,
        )
        return None


# ============================================================
# 端点 1: 同步问答 — 双通道自动分流
# ============================================================


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user: UserContext = Depends(get_current_user),
):
    """
    统一问答端点 — 按角色自动分流

    Customer 通道: RAG Pipeline（轻量快速，只看自己的数据）
    Agent 通道:    Multi-Agent Orchestrator（完整编排，全量数据）

    Headers:
      X-User-Id:   用户唯一标识（Spring Gateway 认证后注入）
      X-User-Role: 角色: customer / agent / underwriter / admin
      X-Org-Id:    所属机构（可选）
    """
    logger.info(
        f"[{user.channel.upper()}] 收到请求: user={user.display_name} "
        f"role={user.role.value} query='{request.query[:50]}'"
    )

    if user.is_customer:
        return await _customer_chat(request, user)
    else:
        return await _agent_chat(request, user)


async def _customer_chat(request: ChatRequest, user: UserContext) -> ChatResponse:
    """
    Customer 通道 — RAG Pipeline

    特点: 轻量快速（~200ms），只能查自己的保单，严格合规审查，
          回答通俗易懂，附带免责声明。
    """
    try:
        result = rag_system.query(
            request.query,
            session_id=request.session_id,
            user_role=user.role.value,
        )

        return ChatResponse(
            answer=result.answer,
            intent=result.intent,
            strategy=f"customer_{result.strategy}",
            sources=result.sources,
            latency_ms=result.latency_ms,
            cache_hit=result.cache_hit,
            channel="customer",
            route_mode="rag",
            agents_used=[],
            disclaimer=(
                "⚠️ 以上信息基于保险条款解读，不构成投保或理赔建议。"
                "具体以保险合同约定和保险公司审核为准。"
            ),
        )

    except Exception as e:
        logger.error(f"[CUSTOMER] 处理失败: {e}", exc_info=True)
        return ChatResponse(
            answer="抱歉，系统暂时繁忙，请稍后再试或联系人工客服。",
            strategy="error",
            channel="customer",
            route_mode="rag",
            disclaimer="如有紧急问题，请拨打客服热线。",
        )


async def _agent_chat(request: ChatRequest, user: UserContext) -> ChatResponse:
    """
    Agent 通道 — Multi-Agent Orchestrator

    特点: 完整编排（~500ms），可查任意客户数据，宽松合规，
          返回完整数据（费率表、核保规则、对比表）。
    """
    # ── 初始化多 Agent 系统 (懒加载) ──
    orch = init_multi_agent()
    if orch is None:
        logger.warning("[AGENT] 多 Agent 不可用，降级为纯 RAG")
        try:
            result = rag_system.query(
                request.query,
                session_id=request.session_id,
                user_role=user.role.value,
            )
            return ChatResponse(
                answer=result.answer,
                intent=result.intent,
                strategy="rag_fallback",
                sources=result.sources,
                latency_ms=result.latency_ms,
                channel="agent",
                route_mode="rag",
                agents_used=[],
            )
        except Exception as e:
            logger.error(f"[AGENT] RAG 降级失败: {e}")
            return ChatResponse(
                answer="抱歉，系统暂时繁忙，请稍后再试或联系人工客服。",
                strategy="error",
                channel="agent",
                route_mode="rag",
            )

    try:
        # ── 第 1 步: 构建 user_profile (长记忆，内部人员可查任意客户) ──
        user_profile = {}
        if user.user_id:
            try:
                mm = get_memory_manager()
                user_profile = mm.get_user_profile(
                    user_id=user.user_id,
                    role=user.role.value,
                )
            except Exception as e:
                logger.warning(f"[AGENT] 获取用户画像失败: {e}，降级为空画像")

        # ── 第 2 步: 意图分类 ──
        intent = ""
        entities = {}
        try:
            from rag_qa.core.query_classifier import QueryClassifier
            classifier = QueryClassifier(bert_model=None, llm_client=None)
            intent_result = classifier.classify(request.query)
            intent = intent_result.intent
            entities = intent_result.entities
        except Exception as e:
            logger.warning(f"[AGENT] 意图分类失败: {e}，使用默认路由")

        # ── 第 3 步: Orchestrator 处理 ──
        result = orch.process(
            query=request.query,
            intent=intent,
            entities=entities,
            session_id=request.session_id,
            user_profile=user_profile,
            user_role=user.role.value,
        )

        return ChatResponse(
            answer=result["answer"],
            intent=intent,
            strategy=f"agent_{result['route_mode']}",
            sources=result.get("sources", []),
            latency_ms=result["latency_ms"],
            channel="agent",
            route_mode=result["route_mode"],
            agents_used=result.get("agents_used", []),
        )

    except Exception as e:
        logger.error(f"[AGENT] 处理请求失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="服务器内部错误")


# ============================================================
# 端点 2: SSE 流式问答 — 双通道自动分流
# ============================================================


@app.get("/chat/stream")
async def chat_stream(
    query: str,
    session_id: str = "",
    user: UserContext = Depends(get_current_user),
):
    """
    SSE 流式问答端点 — 按角色自动分流

    Customer 通道: RAG Pipeline 流式输出 + 逐句推送 + 免责声明
    Agent 通道:    RAG Pipeline 流式输出（内部使用，无冗余声明）

    Headers 同 POST /chat。
    """

    async def event_generator():
        try:
            channel = user.channel

            # ── 推送状态: 开始处理 ──
            yield {
                "event": "status",
                "data": json.dumps(
                    {"stage": "analyzing", "message": "正在分析问题...",
                     "channel": channel},
                    ensure_ascii=False,
                ),
            }

            # ── 执行 RAG 查询 ──
            result = await asyncio.to_thread(
                rag_system.query,
                raw_query=query,
                session_id=session_id,
                user_role=user.role.value,
            )

            # ── 推送状态: 检索完成 ──
            yield {
                "event": "status",
                "data": json.dumps(
                    {
                        "stage": "searching",
                        "message": f"检索完成，找到 {len(result.sources)} 条相关条款",
                        "channel": channel,
                    },
                    ensure_ascii=False,
                ),
            }

            # ── 推送答案 (逐句推送) ──
            sentences = result.answer.split("。")
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    text = sentence.strip() + ("。" if i < len(sentences) - 1 else "")
                    yield {
                        "event": "answer",
                        "data": json.dumps({"text": text}, ensure_ascii=False),
                    }
                    await asyncio.sleep(0.05)

            # ── Customer 通道追加免责声明 ──
            if user.is_customer:
                disclaimer = (
                    "⚠️ 以上信息基于保险条款解读，不构成投保或理赔建议。"
                    "具体以保险合同约定和保险公司审核为准。"
                )
                for sentence in disclaimer.split("。"):
                    if sentence.strip():
                        yield {
                            "event": "answer",
                            "data": json.dumps(
                                {"text": sentence.strip() + "。"},
                                ensure_ascii=False,
                            ),
                        }
                        await asyncio.sleep(0.03)

            # ── 推送来源引用 ──
            if result.sources:
                yield {
                    "event": "sources",
                    "data": json.dumps({"sources": result.sources}, ensure_ascii=False),
                }

            # ── 推送完成信号 ──
            yield {
                "event": "done",
                "data": json.dumps(
                    {
                        "latency_ms": result.latency_ms,
                        "intent": result.intent,
                        "strategy": result.strategy,
                        "cache_hit": result.cache_hit,
                        "channel": channel,
                    },
                    ensure_ascii=False,
                ),
            }

        except Exception as e:
            logger.error(f"[{user.channel.upper()}] 流式处理失败: {e}", exc_info=True)
            yield {
                "event": "error",
                "data": json.dumps(
                    {"error": "处理失败，请稍后再试"}, ensure_ascii=False
                ),
            }

    return EventSourceResponse(event_generator())


# ============================================================
# 端点 3: 文档摄取 (管理后台，仅 ADMIN)
# ============================================================


@app.post("/admin/ingest")
async def ingest_documents(
    request: IngestRequest,
    user: UserContext = Depends(get_current_user),
):
    """
    文档摄取端点 — 新产品 PDF 上线

    仅 ADMIN 角色可调用。
    管理后台调用此端点，将新产品的条款 PDF 导入知识库。
    生产环境建议改为异步任务队列 (Celery / RQ)。
    """
    if user.role not in ADMIN_ROLES:
        logger.warning(
            f"[SECURITY] 非 ADMIN 用户尝试调用 /admin/ingest: "
            f"user={user.display_name} role={user.role.value}"
        )
        raise HTTPException(
            status_code=403,
            detail=f"权限不足: 仅 ADMIN 可执行文档导入，当前角色: {user.role.value}",
        )

    logger.info(
        f"[ADMIN] 文档摄取请求: user={user.display_name} "
        f"product={request.product_name} ({len(request.pdf_paths)} 个文件)"
    )

    try:
        from rag_qa.ingestion.ingestion_orchestrator import ingest_local_pdfs

        stats = ingest_local_pdfs(
            pdf_paths=request.pdf_paths,
            insurer=request.insurer,
            product_name=request.product_name,
            product_code=request.product_code,
        )

        return {
            "status": "success",
            "stats": stats,
            "message": (
                f"摄取完成: 成功 {stats['success']} 个, "
                f"跳过 {stats['skipped']} 个, 失败 {stats['failed']} 个"
            ),
        }

    except Exception as e:
        logger.error(f"[ADMIN] 文档摄取失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"摄取失败: {e}")


# ============================================================
# 健康检查
# ============================================================


@app.get("/health")
async def health_check():
    """健康检查 — K8s 探针 / 负载均衡健康检查"""
    return {
        "status": "ok",
        "service": "insurance-rag-agent-dual-channel",
        "version": "4.0.0",
        "channels": ["customer", "agent"],
    }


# ============================================================
# 启动入口
# ============================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("启动保险智能客服 API (双通道架构)...")
    uvicorn.run(
        "gateway.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
