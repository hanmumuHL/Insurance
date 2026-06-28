# -*- coding: utf-8 -*-
"""
KG 统一服务 — 统一知识图谱的所有访问入口

解决重复问题:
  当前 KG 调用散落在 6 处:
    - rag_system.py          _kg_reasoning_enhance()  → KGReasoner.reason_from_query()
    - orchestrator.py        _get_kg_context()        → KGReasoner.reason_from_query()
    - query_classifier.py    _extract_entities()      → KGEntityLinker.link()
    - strategy_selector.py   _count_product_mentions() → KGEntityLinker.link()
    - kg_reasoner.py         parse_query()            → KGEntityLinker.link() (自引用)
    - kg_entity_linker.py    link()                   → 实体链接主入口

  KGService 提供两个统一入口:
    - link_entities(query)    → 实体链接（委托 KGEntityLinker）
    - get_reasoning(query, intent) → KG 推理上下文（委托 KGReasoner）

使用方式:
    from rag_qa.core.kg import KGService

    kg = KGService()
    entities = kg.link_entities("肺炎住院能赔吗")
    context = kg.get_reasoning("肺炎住院能赔吗", "理赔咨询")
"""

from base.logger import logger


class KGService:
    """
    KG 统一服务 — 知识图谱访问的唯一入口

    封装实体链接和推理两种能力，统一错误处理和降级策略。
    所有需要 KG 的组件都通过此服务访问，不再直接 import KGReasoner/KGEntityLinker。
    """

    # 需要 KG 推理的复杂意图（与 orchestrator._get_kg_context 保持一致）
    COMPLEX_INTENTS = {"理赔咨询", "产品对比", "核保咨询", "条款解读"}

    def __init__(self):
        """延迟初始化，首次调用时创建底层实例"""
        self._linker = None
        self._reasoner = None

    # ================================================================
    # 实体链接
    # ================================================================

    def link_entities(self, query: str) -> dict:
        """
        从 query 中抽取并链接保险实体

        委托 KGEntityLinker.link()，失败时返回空 dict。

        Args:
            query: 用户查询文本

        Returns:
            dict: {
                "insurer": str or None,
                "product": str or None,
                "products": list[str],
                "disease": str or None,
                "disease_category": str or None,
                "event": str or None,
                "dimensions": list[str],
                "all_entities": list[dict],
            }
        """
        try:
            if self._linker is None:
                from rag_qa.core.kg_entity_linker import get_entity_linker
                self._linker = get_entity_linker()
            return self._linker.link(query)
        except Exception as e:
            logger.debug(f"KGService.link_entities 失败: {e}")
            return {}

    # ================================================================
    # KG 推理
    # ================================================================

    def get_reasoning(self, query: str, intent: str = "") -> str:
        """
        为查询获取 KG 推理上下文

        仅对复杂意图（理赔/对比/核保/条款解读）启用 KG 推理，
        其他意图直接返回空字符串（避免无效 Neo4j 查询）。

        委托 KGReasoner.reason_from_query()，失败时返回 ""。

        Args:
            query: 用户查询文本
            intent: 意图类型

        Returns:
            str: KG 推理上下文字符串（可直接注入 LLM prompt），
                 无推理结果时返回 ""
        """
        # 非复杂意图跳过 KG 推理
        if intent and intent not in self.COMPLEX_INTENTS:
            return ""

        try:
            if self._reasoner is None:
                from rag_qa.core.kg_reasoner import KGReasoner
                self._reasoner = KGReasoner()

            paths = self._reasoner.reason_from_query(query)
            if not paths:
                return ""

            # 构建推理摘要（与 rag_system._kg_reasoning_enhance 格式一致）
            parts = []
            for i, path in enumerate(paths[:3]):
                parts.append(f"推理路径{i+1}: {path.explain()}")
                for entity in path.entities_found[:3]:
                    etype = entity.get("type", "")
                    ename = (
                        entity.get("product_name")
                        or entity.get("disease")
                        or entity.get("name", "")
                    )
                    if etype == "Product":
                        parts.append(f"  关联产品: {ename}")

            return "\n".join(parts) if parts else ""

        except Exception as e:
            logger.debug(f"KGService.get_reasoning 跳过: {e}")
            return ""

    # ================================================================
    # 便捷方法: 同时获取实体和推理
    # ================================================================

    def enrich(self, query: str, intent: str = "") -> dict:
        """
        一次性获取实体链接 + KG 推理

        Args:
            query: 用户查询
            intent: 意图类型

        Returns:
            dict: {"entities": {...}, "reasoning": "..."}
        """
        return {
            "entities": self.link_entities(query),
            "reasoning": self.get_reasoning(query, intent),
        }
