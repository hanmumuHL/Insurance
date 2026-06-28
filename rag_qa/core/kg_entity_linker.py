# -*- coding: utf-8 -*-
"""
KG 实体链接器 — 将用户 query 中的实体链接到知识图谱 (Neo4j 后端)

与 NetworkX 版本的区别:
  - 不再直接遍历 kg.graph.nodes()，改为通过 KG 查询 API
  - 索引构建改为 Neo4j 按标签查询 / NetworkX 降级
  - 接口完全兼容，外部调用方无需修改

使用方式:
    from rag_qa.core.kg_entity_linker import KGEntityLinker

    linker = KGEntityLinker(kg)
    entities = linker.link("平安e生保肺炎住院能赔吗")
"""

import re
import threading
from typing import Optional

from base.logger import logger
from rag_qa.core.kg_store import (
    InsuranceKG, NodeType, EdgeType,
    DISEASE_TO_CATEGORY, INSURER_ALIAS,
)


# ============================================================
# 单例
# ============================================================

_entity_linker: Optional["KGEntityLinker"] = None
_linker_lock = threading.Lock()


def get_entity_linker() -> "KGEntityLinker":
    """获取 KGEntityLinker 单例（线程安全）"""
    global _entity_linker
    if _entity_linker is None:
        with _linker_lock:
            if _entity_linker is None:
                _entity_linker = KGEntityLinker()
    return _entity_linker


class KGEntityLinker:
    """基于知识图谱的实体链接器 (Neo4j/NetworkX 双后端)"""

    def __init__(self, kg: InsuranceKG = None):
        from rag_qa.core.kg_store import get_kg
        self.kg = kg or get_kg()
        self._indices_built = False
        self._insurer_index = {}
        self._product_index = {}
        self._disease_index = {}
        self._event_index = {}

    def _ensure_indices(self):
        """构建实体名称→节点ID的快速索引"""
        if self._indices_built:
            return

        self.kg.connect()

        if self.kg._use_neo4j:
            self._build_indices_neo4j()
        else:
            self._build_indices_fallback()

        self._indices_built = True
        logger.info(
            f"[EntityLinker] 索引: insurers={len(self._insurer_index)}, "
            f"products={len(self._product_index)}, diseases={len(self._disease_index)}, "
            f"events={len(self._event_index)}"
        )

    def _build_indices_neo4j(self):
        """从 Neo4j 按标签批量加载索引"""
        try:
            with self.kg._driver.session() as s:
                # 保司
                for record in s.run("MATCH (n:Insurer) RETURN n.name as name, n.aliases as aliases"):
                    name = record["name"]
                    self._insurer_index[name] = name
                    aliases = record.get("aliases") or []
                    if isinstance(aliases, list):
                        for alias in aliases:
                            self._insurer_index[alias] = name
                # 产品
                for record in s.run("MATCH (n:Product) RETURN n.name as name, n.code as code"):
                    name = record["name"]
                    self._product_index[name] = name
                    code = record.get("code")
                    if code:
                        self._product_index[code] = name
                # 疾病
                for record in s.run("MATCH (n:Disease) RETURN n.name as name"):
                    name = record["name"]
                    self._disease_index[name] = name
                # 医疗事件
                for record in s.run("MATCH (n:MedicalEvent) RETURN n.name as name"):
                    name = record["name"]
                    self._event_index[name] = f"event:{name}"
        except Exception as e:
            logger.warning(f"[EntityLinker] Neo4j 索引构建失败: {e}")

    def _build_indices_fallback(self):
        """从 NetworkX 降级图构建索引"""
        g = self.kg._fallback_graph
        if g is None:
            return
        for node, attrs in g.nodes(data=True):
            node_type = attrs.get("type", "")
            if node_type == NodeType.INSURER:
                name = attrs.get("name", node)
                self._insurer_index[name] = node
                for alias in attrs.get("aliases", []):
                    self._insurer_index[alias] = node
            elif node_type == NodeType.PRODUCT:
                name = attrs.get("name", node)
                code = attrs.get("code", "")
                self._product_index[name] = node
                if code:
                    self._product_index[code] = node
            elif node_type == NodeType.DISEASE:
                name = attrs.get("name", node)
                self._disease_index[name] = node
            elif node_type == NodeType.MEDICAL_EVENT:
                name = attrs.get("name", node)
                self._event_index[name] = node

    # ================================================================
    # 主入口
    # ================================================================

    def link(self, query: str) -> dict:
        """从 query 中抽取并链接所有实体"""
        self._ensure_indices()

        result = {
            "insurer": None, "insurer_raw": None,
            "product": None, "products": [],
            "disease": None, "disease_category": None,
            "event": None, "dimensions": [],
            "all_entities": [],
        }

        insurer_result = self._link_insurer(query)
        if insurer_result:
            result["insurer"] = insurer_result["canonical"]
            result["insurer_raw"] = insurer_result["mention"]
            result["all_entities"].append(insurer_result)

        products = self._link_products(query)
        if products:
            result["products"] = [p["canonical"] for p in products]
            result["product"] = result["products"][0]
            result["all_entities"].extend(products)

        disease_result = self._link_disease(query)
        if disease_result:
            result["disease"] = disease_result["canonical"]
            result["disease_category"] = disease_result.get("category")
            result["all_entities"].append(disease_result)

        event_result = self._link_medical_event(query)
        if event_result:
            result["event"] = event_result["canonical"]
            result["all_entities"].append(event_result)

        result["dimensions"] = self._extract_dimensions(query)
        return result

    # ================================================================
    # 各类型实体链接
    # ================================================================

    def _link_insurer(self, query: str) -> Optional[dict]:
        best_match, best_len = None, 0
        for name, node_id in self._insurer_index.items():
            if name in query and len(name) > best_len:
                best_match, best_len = name, len(name)
        if best_match:
            canonical = INSURER_ALIAS.get(best_match, best_match)
            return {"type": NodeType.INSURER, "mention": best_match,
                    "canonical": canonical, "node_id": self._insurer_index.get(best_match, canonical)}
        return None

    def _link_products(self, query: str) -> list[dict]:
        found = []
        for name, node_id in self._product_index.items():
            if name in query:
                found.append({"type": NodeType.PRODUCT, "mention": name,
                              "canonical": name, "node_id": node_id})
        if not found:
            product_patterns = [
                r"([一-龥a-zA-Z]{2,10}(?:保|生|e生|尊享|守护|医疗|重疾))",
            ]
            seen = set()
            for pat in product_patterns:
                for m in re.finditer(pat, query):
                    name = m.group(1)
                    if name not in seen and name not in self._insurer_index:
                        seen.add(name)
                        found.append({"type": NodeType.PRODUCT, "mention": name,
                                      "canonical": name, "node_id": name})
        return found

    def _link_disease(self, query: str) -> Optional[dict]:
        best_match, best_len = None, 0
        for disease, node_id in self._disease_index.items():
            if disease in query and len(disease) > best_len:
                best_match, best_len = disease, len(disease)
        if not best_match:
            for disease in DISEASE_TO_CATEGORY:
                if disease in query and len(disease) > best_len:
                    best_match, best_len = disease, len(disease)
        if best_match:
            category = DISEASE_TO_CATEGORY.get(best_match, "其他")
            return {"type": NodeType.DISEASE, "mention": best_match,
                    "canonical": best_match, "category": category,
                    "node_id": best_match}
        return None

    def _link_medical_event(self, query: str) -> Optional[dict]:
        event_keywords = [
            "住院", "手术", "门诊", "急诊", "ICU", "重症监护",
            "放化疗", "透析", "器官移植",
        ]
        best_match, best_len = None, 0
        for event in event_keywords:
            if event in query and len(event) > best_len:
                best_match, best_len = event, len(event)
        if best_match:
            return {"type": NodeType.MEDICAL_EVENT, "mention": best_match,
                    "canonical": best_match, "node_id": f"event:{best_match}"}
        return None

    @staticmethod
    def _extract_dimensions(query: str) -> list[str]:
        dim_patterns = [
            (r"(等待期|犹豫期|宽限期)", "等待期"),
            (r"(免赔额|起付线|自付比例|自费)", "免赔额"),
            (r"(保费|费率|多少钱|一年多少|每月多少|价格)", "保费"),
            (r"(保障范围|保险责任|保什么|覆盖)", "保障范围"),
            (r"(续保|续费|保证续保|自动续费)", "续保"),
            (r"(健康告知|核保|告知义务|如实告知|既往症)", "健康告知"),
            (r"(理赔|赔付|报销|能赔|赔多少|理赔流程|理赔材料)", "理赔"),
            (r"(退保|退保险|现金价值|犹豫期退保)", "退保"),
        ]
        dimensions, seen = [], set()
        for pattern, dim_name in dim_patterns:
            if re.search(pattern, query) and dim_name not in seen:
                dimensions.append(dim_name)
                seen.add(dim_name)
        return dimensions

    def resolve_ambiguity(self, mentions: list[str], context: dict = None) -> list[dict]:
        """消歧: 多个候选实体中选出最匹配的"""
        candidates = []
        for mention in mentions:
            entity = self.kg.query_entity(mention)
            if entity:
                score = 0.0
                # 使用 query_relations 获取度数替代 NetworkX degree
                relations = self.kg.query_relations(entity["name"], direction="both")
                degree = len(relations)
                score += min(degree / 10.0, 1.0) * 0.3
                if context and context.get("insurer"):
                    for rel in relations:
                        if rel["type"] == NodeType.INSURER and rel["name"] == context["insurer"]:
                            score += 0.5
                            break
                candidates.append({**entity, "score": round(score, 3)})
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates

    def __repr__(self) -> str:
        return f"KGEntityLinker(kg={self.kg})"
