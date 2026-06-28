# -*- coding: utf-8 -*-
"""
FastAPI 网关 — 双通道 API 层入口

双通道架构:
  - CUSTOMER 通道: 外部客户 → RAG Pipeline（轻量、快速、只看自己数据、严格合规）
  - AGENT 通道:   内部人员 → Multi-Agent Orchestrator（完整编排、全量数据、宽松合规）

端点:
  POST /chat          — 同步问答（按角色自动分流）
  GET  /chat/stream   — SSE 流式问答（按角色自动分流）
  POST /admin/ingest  — 文档摄取（仅 ADMIN）
  GET  /health        — 健康检查

路由实现分布在 gateway/routes/ 目录中。
服务单例见 gateway/services.py。
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from base.logger import logger
from gateway.routes import chat_router, stream_router, ingest_router, health_router

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

# ── 注册路由 ──
app.include_router(chat_router)
app.include_router(stream_router)
app.include_router(ingest_router)
app.include_router(health_router)


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
