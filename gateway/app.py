# -*- coding: utf-8 -*-
"""
FastAPI 网关 — API 层入口

提供三个主要端点:
  1. POST /chat         — 同步问答 (简单场景)
  2. GET  /chat/stream   — SSE 流式问答 (推荐，用户体验好)
  3. POST /admin/ingest  — 文档摄取 (管理后台调用)

SSE (Server-Sent Events) 流式返回:
  用户提问后，服务端一边处理一边推送中间状态:
    event: thinking  → "正在分析问题..."
    event: searching → "正在检索条款..."
    event: answer    → 逐字推送答案
    event: done      → 完成，附带来源引用

  好处: 用户不用干等 350ms+，立刻看到反馈
"""

import json
import asyncio
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field
from typing import Optional

from base.logger import logger
from rag_qa.core.rag_system import RAGSystem

# ── 多 Agent 架构 ──
from agent.sub_agents import (
    InsuranceAgent, UnderwritingAgent, ClaimAgent, ServiceAgent,
)
from agent.orchestrator import Orchestrator
from agent.memory import get_memory_manager


# ============================================================
# FastAPI 应用初始化
# ============================================================

app = FastAPI(
    title="保险智能客服 API",
    description="RAG 智能客服 + 纯多 Agent 协作 (Orchestrator + 4 领域 Agent)",
    version="3.0.0",
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

# ── 多 Agent 系统 (懒初始化，在首次使用或 startup 事件中初始化) ──
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

        # ── 创建四个子 Agent（注入 checkpointer）──
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
# 请求 / 响应模型
# ============================================================


class ChatRequest(BaseModel):
    """聊天请求"""

    query: str = Field(..., description="用户问题", min_length=1, max_length=2000)
    session_id: str = Field(default="", description="会话 ID (多轮对话时传入)")


class ChatResponse(BaseModel):
    """聊天响应"""

    answer: str = Field(description="回答内容")
    intent: str = Field(default="", description="识别的意图")
    strategy: str = Field(default="", description="使用的检索策略")
    sources: list[dict] = Field(default_factory=list, description="引用的条款来源")
    latency_ms: float = Field(default=0, description="处理耗时 (毫秒)")
    cache_hit: bool = Field(default=False, description="是否命中缓存")


class IngestRequest(BaseModel):
    """文档摄取请求"""

    pdf_paths: list[str] = Field(description="PDF 文件路径列表")
    insurer: str = Field(description="保险公司名称")
    product_name: str = Field(description="产品名称")
    product_code: str = Field(description="产品编码")


# ============================================================
# 端点 1: 同步问答
# ============================================================


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
):
    """
    统一问答端点 — Orchestrator 多 Agent 协作（含长短记忆）

    Header:
      X-User-Id: 用户 ID（Spring 网关认证后注入），用于查询长记忆

    所有请求统一走 Orchestrator，内部自动路由。
    """
    user_id = x_user_id or ""
    logger.info(
        f"收到请求: user_id={user_id[:8] if user_id else '?'} "
        f"query='{request.query[:50]}'"
    )

    # ── 初始化多 Agent 系统 (懒加载) ──
    orch = init_multi_agent()
    if orch is None:
        logger.warning("多 Agent 不可用，降级为纯 RAG")
        try:
            result = rag_system.query(request.query, session_id=request.session_id)
            return ChatResponse(
                answer=result.answer,
                intent=result.intent,
                strategy="rag_fallback",
                sources=result.sources,
                latency_ms=result.latency_ms,
            )
        except Exception as e:
            logger.error(f"RAG 降级失败: {e}")
            return ChatResponse(
                answer="抱歉，系统暂时繁忙，请稍后再试或联系人工客服。",
                strategy="error",
            )

    try:
        # ── 第 1 步: 构建 user_profile (长记忆) ──
        user_profile = {}
        if user_id:
            try:
                mm = get_memory_manager()
                user_profile = mm.get_user_profile(user_id)
            except Exception as e:
                logger.warning(f"获取用户画像失败: {e}，降级为空画像")

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
            logger.warning(f"意图分类失败: {e}，使用默认路由")

        # ── 第 3 步: Orchestrator 处理（注入长记忆 + session_id 用于短记忆）──
        result = orch.process(
            query=request.query,
            intent=intent,
            entities=entities,
            session_id=request.session_id,
            user_profile=user_profile,
        )

        return ChatResponse(
            answer=result["answer"],
            intent=intent,
            strategy=f"orch_{result['route_mode']}",
            sources=result.get("sources", []),
            latency_ms=result["latency_ms"],
        )

    except Exception as e:
        logger.error(f"处理请求失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="服务器内部错误")


# ============================================================
# 端点 2: SSE 流式问答 (推荐)
# ============================================================


@app.get("/chat/stream")
async def chat_stream(query: str, session_id: str = ""):
    """
    SSE 流式问答端点

    使用 Server-Sent Events 逐步推送处理进度和答案:
      event: status   → 处理进度 (正在分析/检索/生成...)
      event: answer   → 答案片段 (逐字或逐句推送)
      event: sources  → 引用的条款来源
      event: done     → 处理完成
      event: error    → 错误信息

    前端使用 EventSource 接收:
      const es = new EventSource('/chat/stream?query=肺炎能赔吗');
      es.addEventListener('answer', (e) => { appendText(e.data); });
      es.addEventListener('done', (e) => { es.close(); });
    """

    async def event_generator():
        try:
            # ── 推送状态: 开始处理 ──
            yield {
                "event": "status",
                "data": json.dumps(
                    {"stage": "analyzing", "message": "正在分析问题..."},
                    ensure_ascii=False,
                ),
            }

            # ── 执行 RAG 查询 ──
            # 注意: RAG 是同步的，用 asyncio.to_thread 放到线程池执行
            # 这样不会阻塞事件循环
            result = await asyncio.to_thread(
                rag_system.query,
                raw_query=query,
                session_id=session_id,
            )

            # ── 推送状态: 检索完成 ──
            yield {
                "event": "status",
                "data": json.dumps(
                    {
                        "stage": "searching",
                        "message": f"检索完成，找到 {len(result.sources)} 条相关条款",
                    },
                    ensure_ascii=False,
                ),
            }

            # ── 推送答案 (逐句推送，模拟打字效果) ──
            sentences = result.answer.split("。")
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    text = sentence.strip() + ("。" if i < len(sentences) - 1 else "")
                    yield {
                        "event": "answer",
                        "data": json.dumps({"text": text}, ensure_ascii=False),
                    }
                    # 每句间隔 50ms，模拟打字效果
                    await asyncio.sleep(0.05)

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
                    },
                    ensure_ascii=False,
                ),
            }

        except Exception as e:
            logger.error(f"流式处理失败: {e}", exc_info=True)
            yield {
                "event": "error",
                "data": json.dumps(
                    {"error": "处理失败，请稍后再试"}, ensure_ascii=False
                ),
            }

    return EventSourceResponse(event_generator())


# ============================================================
# 端点 3: 文档摄取 (管理后台)
# ============================================================


@app.post("/admin/ingest")
async def ingest_documents(request: IngestRequest):
    """
    文档摄取端点 — 新产品 PDF 上线

    管理后台调用此端点，将新产品的条款 PDF 导入知识库。
    处理过程是同步的（可能需要几秒到几十秒），
    生产环境建议改为异步任务队列 (Celery / RQ)。
    """
    logger.info(
        f"文档摄取请求: {request.product_name} ({len(request.pdf_paths)} 个文件)"
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
            "message": f"摄取完成: 成功 {stats['success']} 个, 跳过 {stats['skipped']} 个, 失败 {stats['failed']} 个",
        }

    except Exception as e:
        logger.error(f"文档摄取失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"摄取失败: {e}")


# ============================================================
# 健康检查
# ============================================================


@app.get("/health")
async def health_check():
    """健康检查 — K8s 探针 / 负载均衡健康检查"""
    return {
        "status": "ok",
        "service": "insurance-rag-agent",
        "version": "1.0.0",
    }


# ============================================================
# 启动入口
# ============================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("启动保险智能客服 API...")
    uvicorn.run(
        "gateway.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 开发模式：文件变更自动重启
        log_level="info",
    )
