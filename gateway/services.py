# -*- coding: utf-8 -*-
"""
服务单例 — Gateway 层共享的服务实例

从 app.py 中抽离，使路由模块可以独立导入所需服务，
同时避免循环导入问题。

使用方式:
    from gateway.services import get_rag_system, get_orchestrator
    rag = get_rag_system()
    orch = get_orchestrator()
"""

import threading

from base.logger import logger

# ── RAG 系统 (单例) ──
_rag_system = None
_rag_lock = threading.Lock()


def get_rag_system():
    """获取 RAG 系统单例"""
    global _rag_system
    if _rag_system is None:
        with _rag_lock:
            if _rag_system is None:
                from rag_qa.core.rag_system import RAGSystem
                _rag_system = RAGSystem()
                logger.info("RAGSystem 初始化完成")
    return _rag_system


# ── 多 Agent Orchestrator (懒加载单例) ──
_orchestrator = None
_agent_lock = threading.Lock()


def get_orchestrator():
    """
    获取多 Agent Orchestrator 单例（懒加载 + 双重检查锁）

    失败时返回 None，调用方会降级到纯 RAG 模式。
    """
    global _orchestrator

    if _orchestrator is not None:
        return _orchestrator

    with _agent_lock:
        if _orchestrator is not None:
            return _orchestrator

        try:
            from base.llm_client import get_llm_client

            llm_client = get_llm_client()
            rag_system = get_rag_system()

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
            from agent.memory import get_memory_manager
            mm = get_memory_manager()
            checkpointer = mm.get_checkpointer()

            # ── 统一服务（智能管道）──
            from rag_qa.core.pipeline.retrieval_interface import RetrievalInterface
            from rag_qa.core.kg.service import KGService

            retrieval_interface = RetrievalInterface(vector_store=vector_store)
            kg_service = KGService()

            # ── 四个子 Agent ──
            from agent.sub_agents import (
                InsuranceAgent, UnderwritingAgent, ClaimAgent, ServiceAgent,
            )
            from agent.orchestrator import Orchestrator

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
            _orchestrator._mysql_session = mysql_session
            _orchestrator._retrieval = retrieval_interface
            _orchestrator._kg_service = kg_service

            logger.info(
                f"多 Agent 系统初始化完成 — "
                f"已注册: {list(agents.keys())} | "
                f"checkpointer: {'RedisSaver' if checkpointer else '无状态'} | "
                f"RAG 降级: {'可用' if rag_system else '不可用'} | "
                f"RetrievalInterface: 已注入 | "
                f"KGService: 已注入"
            )

            return _orchestrator

        except Exception as e:
            logger.warning(
                f"多 Agent 初始化失败 (RAG 模式仍可用): {e}",
                exc_info=True,
            )
            return None
