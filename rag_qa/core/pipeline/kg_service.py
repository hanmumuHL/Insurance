# -*- coding: utf-8 -*-
"""
⚠️ 已迁移 — 请使用新路径

KGService 已从 rag_qa.core.pipeline.kg_service 迁移到 rag_qa.core.kg.service。

旧路径仍可用但建议更新:
    from rag_qa.core.kg import KGService  # 推荐
    from rag_qa.core.kg.service import KGService  # 也可
"""
from rag_qa.core.kg.service import KGService  # noqa: F401 — 向后兼容
