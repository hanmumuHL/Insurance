# -*- coding: utf-8 -*-
"""
应用组合根 (Composition Root)

集中管理所有服务单例的创建和依赖注入。
这是整个应用的唯一"组装点"——所有组件从这里获取依赖，
而非自行导入具体实现。

服务清单:
  - RAGSystem:          RAG 检索增强系统（向量检索 + LLM 生成 + 合规审查）
  - Orchestrator:       多 Agent 调度中心（4 领域 Agent 协作）
  - LLMClient:          LLM 客户端（DeepSeek 主 + Qwen 降级 + 熔断器）
  - BGEM3Encoder:       向量编码器（Dense 1024d + Sparse）
  - RedisClient:        缓存客户端
  - MySQL Session:      数据库连接
  - MemoryManager:      长短期记忆管理
  - RetrievalInterface: 统一检索接口
  - KGService:          知识图谱统一服务

使用方式:
    from bootstrap import get_rag_system, get_orchestrator

    rag = get_rag_system()
    orch = get_orchestrator()
"""

# ── 从 gateway/services.py 重导出（所有服务单例）──
from gateway.services import get_rag_system, get_orchestrator

__all__ = ["get_rag_system", "get_orchestrator"]
