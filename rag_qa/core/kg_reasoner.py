# -*- coding: utf-8 -*-
"""
KG 多跳推理引擎 — 支持保险领域的复杂多步推理查询 (Neo4j/NetworkX双后端)

与 NetworkX 版本的区别:
  - 推理方法改为通过 KG 查询 API 或 Cypher 直查 Neo4j
  - NetworkX 降级路径保留
  - 外部接口完全兼容

使用方式:
    from rag_qa.core.kg_reasoner import KGReasoner
    reasoner = KGReasoner(kg)
    paths = reasoner.reason_from_query("覆盖心脏疾病且等待期不超过30天的产品")
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from base.logger import logger
from rag_qa.core.kg_store import (
    InsuranceKG, NodeType, EdgeType,
    DISEASE_TO_CATEGORY,
)


# ============================================================
# 推理路径数据结构
# ============================================================

@dataclass
class ReasonStep:
    node: str
    node_type: str
    relation: str
    evidence: str = ""


@dataclass
class ReasonPath:
    steps: list[ReasonStep] = field(default_factory=list)
    confidence: float = 1.0
    entities_found: list[dict] = field(default_factory=list)

    def explain(self) -> str:
        if not self.steps:
            return "无推理路径"
        parts = [self.steps[0].node]
        for step in self.steps[1:]:
            parts.append(f" --[{step.relation}]--> {step.node}")
        return " ".join(parts)

    def get_final_entities(self) -> list[str]:
        return [e["name"] for e in self.entities_found]


# ============================================================
# 推理引擎
# ============================================================

class KGReasoner:
    """知识图谱多跳推理引擎"""

    def __init__(self, kg: InsuranceKG = None):
        from rag_qa.core.kg_store import get_kg
        self.kg = kg or get_kg()
        self.kg.connect()

    @property
    def _neo4j(self):
        """是否使用 Neo4j 后端"""
        return self.kg._use_neo4j and self.kg._driver

    # ================================================================
    # 推理: 疾病 → 产品
    # ================================================================

    def reason_disease_to_products(self, disease: str) -> list[ReasonPath]:
        """推理: 哪些产品覆盖某种疾病"""
        if self._neo4j:
            return self._neo4j_disease_to_products(disease)
        else:
            return self._fallback_disease_to_products(disease)

    def _neo4j_disease_to_products(self, disease: str) -> list[ReasonPath]:
        query = """
            MATCH (d:Disease {name: $disease})<-[r:covers]-(c:Clause)<-[:has_clause]-(p:Product)
            RETURN p.name as product, c.clause_type as clause_type
        """
        paths = []
        try:
            with self.kg._driver.session() as s:
                records = list(s.run(query, disease=disease))
                for rec in records:
                    paths.append(ReasonPath(
                        steps=[
                            ReasonStep(node=f"disease:{disease}", node_type=NodeType.DISEASE,
                                       relation="START", evidence=f"查询疾病: {disease}"),
                            ReasonStep(node=f"Clause({rec['clause_type']})", node_type=NodeType.CLAUSE,
                                       relation=EdgeType.COVERS, evidence=f"{rec['clause_type']}覆盖该疾病"),
                            ReasonStep(node=rec["product"], node_type=NodeType.PRODUCT,
                                       relation=EdgeType.HAS_CLAUSE, evidence=f"产品包含该条款"),
                        ],
                        entities_found=[{"name": rec["product"], "type": NodeType.PRODUCT,
                                         "product_name": rec["product"], "clause_type": rec["clause_type"]}],
                    ))
        except Exception as e:
            logger.warning(f"[Reasoner] Neo4j 疾病推理失败: {e}")

        logger.info(f"[Reasoner] disease→products: '{disease}' → {len(paths)} 个产品")
        return paths

    def _fallback_disease_to_products(self, disease: str) -> list[ReasonPath]:
        g = self.kg._fallback_graph
        if g is None:
            return []
        disease_key = f"disease:{disease}"
        if disease_key not in g:
            category = DISEASE_TO_CATEGORY.get(disease)
            if category:
                return self._fallback_category_to_products(category, disease)
            return []

        paths = []
        disease_attrs = dict(g.nodes[disease_key])
        for clause_key in g.predecessors(disease_key):
            edge_data = g.get_edge_data(clause_key, disease_key)
            if not edge_data or edge_data.get("type") != EdgeType.COVERS:
                continue
            clause_attrs = dict(g.nodes[clause_key])
            for product_key in g.predecessors(clause_key):
                pe = g.get_edge_data(product_key, clause_key)
                if not pe or pe.get("type") != EdgeType.HAS_CLAUSE:
                    continue
                product_attrs = dict(g.nodes[product_key])
                paths.append(ReasonPath(
                    steps=[
                        ReasonStep(node=disease_key, node_type=NodeType.DISEASE, relation="START"),
                        ReasonStep(node=clause_key, node_type=NodeType.CLAUSE,
                                   relation=EdgeType.COVERS, evidence=clause_attrs.get("clause_type", "")),
                        ReasonStep(node=product_key, node_type=NodeType.PRODUCT,
                                   relation=EdgeType.HAS_CLAUSE),
                    ],
                    entities_found=[{"name": product_key, "type": NodeType.PRODUCT,
                                     "product_name": product_attrs.get("name", product_key)}],
                ))
        return paths

    def _fallback_category_to_products(self, category: str, original_disease: str = "") -> list[ReasonPath]:
        g = self.kg._fallback_graph
        if g is None:
            return []
        cat_key = f"category:{category}"
        if cat_key not in g:
            return []
        paths = []
        for disease_key in g.predecessors(cat_key):
            if dict(g.nodes[disease_key]).get("type") != NodeType.DISEASE:
                continue
            for clause_key in g.predecessors(disease_key):
                if not g.get_edge_data(clause_key, disease_key) or \
                   g.get_edge_data(clause_key, disease_key).get("type") != EdgeType.COVERS:
                    continue
                for product_key in g.predecessors(clause_key):
                    product_attrs = dict(g.nodes[product_key])
                    paths.append(ReasonPath(
                        steps=[
                            ReasonStep(node=f"disease:{original_disease}" if original_disease else disease_key,
                                       node_type=NodeType.DISEASE, relation="START"),
                            ReasonStep(node=cat_key, node_type=NodeType.DISEASE_CATEGORY,
                                       relation=EdgeType.BELONGS_TO),
                            ReasonStep(node=product_key, node_type=NodeType.PRODUCT,
                                       relation=EdgeType.HAS_CLAUSE),
                        ],
                        entities_found=[{"name": product_key, "type": NodeType.PRODUCT,
                                         "product_name": product_attrs.get("name", product_key)}],
                    ))
        return paths

    # ================================================================
    # 推理: 产品 → 疾病
    # ================================================================

    def reason_product_to_diseases(self, product_name: str) -> list[ReasonPath]:
        if self._neo4j:
            return self._neo4j_product_to_diseases(product_name)
        else:
            return self._fallback_product_to_diseases(product_name)

    def _neo4j_product_to_diseases(self, product_name: str) -> list[ReasonPath]:
        query = """
            MATCH (p:Product {name: $name})-[:has_clause]->(c:Clause)-[r]->(d:Disease)
            OPTIONAL MATCH (d)-[:belongs_to]->(cat:DiseaseCategory)
            RETURN d.name as disease, type(r) as relation, c.clause_type as clause_type,
                   cat.name as category
        """
        paths = []
        try:
            with self.kg._driver.session() as s:
                for rec in s.run(query, name=product_name):
                    paths.append(ReasonPath(
                        steps=[
                            ReasonStep(node=product_name, node_type=NodeType.PRODUCT, relation="START"),
                            ReasonStep(node=f"Clause({rec['clause_type']})", node_type=NodeType.CLAUSE,
                                       relation=EdgeType.HAS_CLAUSE),
                            ReasonStep(node=rec["disease"], node_type=NodeType.DISEASE,
                                       relation=rec["relation"],
                                       evidence=f"分类: {rec['category']}" if rec["category"] else ""),
                        ],
                        entities_found=[{"name": rec["disease"], "type": NodeType.DISEASE,
                                         "disease": rec["disease"], "category": rec["category"],
                                         "relation": rec["relation"]}],
                    ))
        except Exception as e:
            logger.warning(f"[Reasoner] Neo4j 产品推理失败: {e}")
        return paths

    def _fallback_product_to_diseases(self, product_name: str) -> list[ReasonPath]:
        g = self.kg._fallback_graph
        if g is None:
            return []
        product_key = None
        for node, attrs in g.nodes(data=True):
            if attrs.get("type") == NodeType.PRODUCT and (attrs.get("name") == product_name or node == product_name):
                product_key = node
                break
        if not product_key:
            return []
        paths = []
        for clause_key in g.successors(product_key):
            if not g.get_edge_data(product_key, clause_key) or \
               g.get_edge_data(product_key, clause_key).get("type") != EdgeType.HAS_CLAUSE:
                continue
            for disease_key in g.successors(clause_key):
                de = g.get_edge_data(clause_key, disease_key)
                if not de:
                    continue
                disease_attrs = dict(g.nodes[disease_key])
                category = ""
                for cat_key in g.successors(disease_key):
                    ce = g.get_edge_data(disease_key, cat_key)
                    if ce and ce.get("type") == EdgeType.BELONGS_TO:
                        category = dict(g.nodes[cat_key]).get("name", "")
                        break
                paths.append(ReasonPath(
                    steps=[
                        ReasonStep(node=product_key, node_type=NodeType.PRODUCT, relation="START"),
                        ReasonStep(node=clause_key, node_type=NodeType.CLAUSE, relation=EdgeType.HAS_CLAUSE),
                        ReasonStep(node=disease_key, node_type=NodeType.DISEASE, relation=de.get("type", "")),
                    ],
                    entities_found=[{"name": disease_key, "type": NodeType.DISEASE,
                                     "disease": disease_attrs.get("name", ""), "category": category}],
                ))
        return paths

    # ================================================================
    # 推理: 产品 → 等待期/免赔额
    # ================================================================

    def reason_product_to_waiting_period(self, product_name: str) -> list[ReasonPath]:
        return self._reason_product_attr(product_name, "WaitingPeriod", "has_waiting_period", "days")

    def reason_product_to_deductible(self, product_name: str) -> list[ReasonPath]:
        return self._reason_product_attr(product_name, "Deductible", "has_deductible", "amount")

    def _reason_product_attr(self, product_name: str, label: str,
                              rel_type: str, attr_key: str) -> list[ReasonPath]:
        if self._neo4j:
            query = f"""
                MATCH (p:Product {{name: $name}})-[:{rel_type}]->(a:{label})
                RETURN a.{attr_key} as value
            """
            paths = []
            try:
                with self.kg._driver.session() as s:
                    for rec in s.run(query, name=product_name):
                        paths.append(ReasonPath(
                            steps=[
                                ReasonStep(node=product_name, node_type=NodeType.PRODUCT, relation="START"),
                                ReasonStep(node=f"{label}({attr_key}={rec['value']})",
                                           node_type=label, relation=rel_type),
                            ],
                            entities_found=[{"name": f"{product_name}:{label}", "type": label,
                                             attr_key: rec["value"]}],
                        ))
            except Exception as e:
                logger.warning(f"[Reasoner] Neo4j 属性推理失败: {e}")
            return paths
        else:
            g = self.kg._fallback_graph
            if g is None:
                return []
            product_key = None
            for node, attrs in g.nodes(data=True):
                if attrs.get("type") == NodeType.PRODUCT and (attrs.get("name") == product_name or node == product_name):
                    product_key = node
                    break
            if not product_key:
                return []
            paths = []
            for target in g.successors(product_key):
                edge = g.get_edge_data(product_key, target)
                if not edge or edge.get("type") != rel_type:
                    continue
                tattrs = dict(g.nodes[target])
                paths.append(ReasonPath(
                    steps=[
                        ReasonStep(node=product_key, node_type=NodeType.PRODUCT, relation="START"),
                        ReasonStep(node=target, node_type=label, relation=rel_type,
                                   evidence=f"{attr_key}={tattrs.get(attr_key, '?')}"),
                    ],
                    entities_found=[{"name": target, "type": label, attr_key: tattrs.get(attr_key)}],
                ))
            return paths

    # ================================================================
    # 推理: 复杂多约束查询
    # ================================================================

    def reason_complex(
        self, disease: str = "", max_waiting_days: int = None,
        max_deductible: int = None, insurer: str = "", product_type: str = "",
    ) -> list[ReasonPath]:
        """多约束组合查询"""
        if self._neo4j:
            return self._neo4j_complex(disease, max_waiting_days, max_deductible, insurer)
        else:
            return self._fallback_complex(disease, max_waiting_days, max_deductible, insurer)

    def _neo4j_complex(
        self, disease: str, max_waiting_days: int, max_deductible: int, insurer: str
    ) -> list[ReasonPath]:
        """用 Cypher 组合查询实现多约束过滤"""
        # 构建动态 Cypher
        where_clauses = []
        if disease:
            where_clauses.append(
                "(d:Disease {name: $disease})<-[:covers]-(:Clause)<-[:has_clause]-(p:Product)"
            )
        else:
            where_clauses.append("(p:Product)")

        query_parts = ["MATCH " + where_clauses[0]]

        if insurer:
            query_parts.append("MATCH (i:Insurer {name: $insurer})-[:issues]->(p)")

        if max_waiting_days is not None:
            query_parts.append(
                "OPTIONAL MATCH (p)-[:has_waiting_period]->(wp:WaitingPeriod)"
            )

        query_parts.append("RETURN DISTINCT p.name as product, p.code as code")

        if max_waiting_days is not None:
            query_parts[-1] = (
                "WHERE wp.days <= $max_days OR wp IS NULL "
                "RETURN DISTINCT p.name as product, p.code as code"
            )

        query = "\n".join(query_parts)
        params = {"disease": disease, "insurer": insurer, "max_days": max_waiting_days}
        params = {k: v for k, v in params.items() if v is not None}

        paths = []
        try:
            with self.kg._driver.session() as s:
                for rec in s.run(query, **params):
                    paths.append(ReasonPath(
                        steps=[ReasonStep(node=rec["product"], node_type=NodeType.PRODUCT,
                                          relation="FINAL",
                                          evidence=f"产品: {rec['product']}" +
                                          (f" ({rec['code']})" if rec.get("code") else ""))],
                        confidence=0.9,
                        entities_found=[{"name": rec["product"], "type": NodeType.PRODUCT,
                                         "product_name": rec["product"],
                                         "product_code": rec.get("code", "")}],
                    ))
        except Exception as e:
            logger.warning(f"[Reasoner] Neo4j 复杂查询失败: {e}")

        logger.info(f"[Reasoner] 复杂查询: {len(paths)} 个产品满足约束")
        return paths

    def _fallback_complex(
        self, disease: str, max_waiting_days: int, max_deductible: int, insurer: str
    ) -> list[ReasonPath]:
        g = self.kg._fallback_graph
        if g is None:
            return []

        if disease:
            candidate_paths = self.reason_disease_to_products(disease)
            if not candidate_paths:
                category = DISEASE_TO_CATEGORY.get(disease, "")
                if category:
                    candidate_paths = self._fallback_category_to_products(category, disease)
            if not candidate_paths:
                return []
            candidate_products = list(set(
                e["name"] for p in candidate_paths for e in p.entities_found
            ))
        else:
            candidate_products = [
                node for node, attrs in g.nodes(data=True)
                if attrs.get("type") == NodeType.PRODUCT
            ]

        results = []
        for product_key in candidate_products:
            product_attrs = dict(g.nodes[product_key])
            product_name = product_attrs.get("name", product_key)

            if max_waiting_days is not None:
                wp_paths = self.reason_product_to_waiting_period(product_key)
                if wp_paths:
                    valid = any(
                        e.get("days") and e["days"] <= max_waiting_days
                        for wp in wp_paths for e in wp.entities_found
                    )
                    if not valid:
                        continue

            if insurer:
                found = any(
                    dict(g.nodes[pred]).get("type") == NodeType.INSURER and
                    (insurer in dict(g.nodes[pred]).get("name", pred) or
                     dict(g.nodes[pred]).get("name", pred) in insurer)
                    for pred in g.predecessors(product_key)
                )
                if not found:
                    continue

            results.append(ReasonPath(
                steps=[ReasonStep(node=product_key, node_type=NodeType.PRODUCT,
                                  relation="FINAL", evidence=f"产品: {product_name}")],
                confidence=0.9,
                entities_found=[{"name": product_key, "type": NodeType.PRODUCT,
                                 "product_name": product_name}],
            ))

        return results

    # ================================================================
    # 自然语言查询
    # ================================================================

    def parse_query(self, query: str) -> dict:
        """从自然语言解析约束条件"""
        from rag_qa.core.kg_entity_linker import get_entity_linker
        linker = get_entity_linker()
        entities = linker.link(query)

        constraints = {
            "disease": entities.get("disease"),
            "max_waiting_days": None,
            "max_deductible": None,
            "insurer": entities.get("insurer"),
            "product_type": None,
        }

        m = re.search(r"等待期[不超≤少于]{1,2}\s*(\d+)\s*[天日]", query)
        if m:
            constraints["max_waiting_days"] = int(m.group(1))

        m = re.search(r"免赔额[不超≤少于]{1,2}\s*(\d+)\s*[万元]", query)
        if m:
            val = int(m.group(1))
            if "万" in m.group(0):
                val *= 10000
            constraints["max_deductible"] = val

        for ptype in ["百万医疗", "重疾险", "意外险", "定期寿险", "医疗险"]:
            if ptype in query:
                constraints["product_type"] = ptype
                break

        return constraints

    def reason_from_query(self, query: str) -> list[ReasonPath]:
        """从自然语言 query 直接推理"""
        constraints = self.parse_query(query)
        if not any([
            constraints["disease"], constraints["max_waiting_days"],
            constraints["max_deductible"], constraints["insurer"],
        ]):
            return []
        return self.reason_complex(**{k: v for k, v in constraints.items()
                                       if k != "product_type"})

    def __repr__(self) -> str:
        return f"KGReasoner(kg={self.kg})"
