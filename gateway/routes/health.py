# -*- coding: utf-8 -*-
"""
GET /health 端点 — 健康检查

用于 K8s 探针 / 负载均衡健康检查。
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """健康检查 — K8s 探针 / 负载均衡健康检查"""
    return {
        "status": "ok",
        "service": "insurance-rag-agent-dual-channel",
        "version": "4.0.0",
        "channels": ["customer", "agent"],
    }
