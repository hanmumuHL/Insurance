# -*- coding: utf-8 -*-
"""
Gateway 路由模块

每个端点独立文件，通过 APIRouter 注册到主 app。
"""

from gateway.routes.chat import router as chat_router
from gateway.routes.stream import router as stream_router
from gateway.routes.ingest import router as ingest_router
from gateway.routes.health import router as health_router

__all__ = ["chat_router", "stream_router", "ingest_router", "health_router"]
