# -*- coding: utf-8 -*-
"""
保险领域知识图谱 (KG) 子系统

组件:
  - InsuranceKG / get_kg():        图存储 (Neo4j + NetworkX 降级)
  - KGEntityLinker / get_entity_linker(): 实体链接
  - KGReasoner:                     多跳推理引擎
  - KGService:                      统一服务入口（推荐使用）
  - NodeType / EdgeType:            节点与关系类型常量
  - DISEASE_TO_CATEGORY / INSURER_ALIAS: 疾病分类与保司别名

推荐入口:
    from rag_qa.core.kg import KGService
    kg = KGService()
    entities = kg.link_entities("肺炎住院能赔吗")
    reasoning = kg.get_reasoning("肺炎住院能赔吗", "理赔咨询")
"""

from rag_qa.core.kg_store import (
    InsuranceKG, get_kg,
    NodeType, EdgeType,
    DISEASE_TO_CATEGORY, DISEASE_CATEGORY_MAP,
    INSURER_ALIAS,
)
from rag_qa.core.kg_entity_linker import KGEntityLinker, get_entity_linker
from rag_qa.core.kg_reasoner import KGReasoner, ReasonPath, ReasonStep

# KGService 已从 pipeline/ 迁出到此
from rag_qa.core.kg.service import KGService

__all__ = [
    # 存储
    "InsuranceKG", "get_kg",
    "NodeType", "EdgeType",
    "DISEASE_TO_CATEGORY", "DISEASE_CATEGORY_MAP",
    "INSURER_ALIAS",
    # 实体链接
    "KGEntityLinker", "get_entity_linker",
    # 推理
    "KGReasoner", "ReasonPath", "ReasonStep",
    # 统一服务
    "KGService",
]
