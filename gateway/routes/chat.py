# -*- coding: utf-8 -*-
"""
POST /chat 端点 — 双通道自动分流

Customer 通道: RAG Pipeline（轻量快速，只看自己的数据，严格合规）
Agent 通道:    Multi-Agent Orchestrator（完整编排，全量数据，宽松合规）
"""

from fastapi import APIRouter, Depends, HTTPException

from base.logger import logger
from gateway.models import ChatRequest, ChatResponse
from gateway.auth import UserContext, get_current_user
from gateway.services import get_rag_system, get_orchestrator

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user: UserContext = Depends(get_current_user),
):
    """
    统一问答端点 — 按角色自动分流

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
        rag_system = get_rag_system()
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
    rag_system = get_rag_system()
    orch = get_orchestrator()

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
        # ── 第 1 步: 构建 user_profile ──
        user_profile = {}
        if user.user_id:
            try:
                from agent.memory import get_memory_manager
                mm = get_memory_manager()
                user_profile = mm.get_user_profile(
                    user_id=user.user_id,
                    role=user.role.value,
                )
            except Exception as e:
                logger.warning(f"[AGENT] 获取用户画像失败: {e}，降级为空画像")

        # ── 第 2 步: 意图分类（复用 RAG 系统的完整三层分类器）──
        intent = ""
        entities = ""
        try:
            intent_result = rag_system.classifier.classify(request.query)
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
            intent=result.get("intent", intent),
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
