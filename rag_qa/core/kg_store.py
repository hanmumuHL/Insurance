# -*- coding: utf-8 -*-
"""
保险领域知识图谱 — Neo4j 图数据库存储

设计背景:
  保险领域是天然的图结构。经过 Phase 1 NetworkX 验证后，迁移到 Neo4j 做生产级存储。
  Neo4j 提供: Cypher 图查询语言、ACID 事务、持久化存储、可视化浏览器。

KG Schema (核心本体):
  Insurer --[issues]--> Product --[has_clause]--> Clause
                                      |
           +-------------+------------+------------+
           |             |            |            |
      Coverage      Exclusion    WaitingPeriod  Deductible
           |                            |
      DiseaseCategory               Duration
      (呼吸系统/心血管/...)         (30天/60天/90天)

技术栈:
  主后端: Neo4j (bolt://localhost:7687)
  降级方案: NetworkX 内存图 (Neo4j 不可用时自动切换)

使用方式:
    from rag_qa.core.kg_store import InsuranceKG

    kg = InsuranceKG()
    kg.build_from_knowledge_base()   # 导入内置疾病分类 + 保司
    kg.build_from_db(session)        # 从 MySQL 导入产品关系

    # 查询 (与原来完全一致)
    entity = kg.query_entity("肺炎")
    relations = kg.query_relations("平安健康", "issues")
    neighbors = kg.get_linked_entities("disease:肺炎", hops=2)
"""

import re
import os
from collections import defaultdict
from typing import Optional

from base.logger import logger
from config.settings import settings


# ============================================================
# 节点与关系类型常量
# ============================================================

class NodeType:
    INSURER = "Insurer"
    PRODUCT = "Product"
    CLAUSE = "Clause"
    DISEASE = "Disease"
    DISEASE_CATEGORY = "DiseaseCategory"
    MEDICAL_EVENT = "MedicalEvent"
    WAITING_PERIOD = "WaitingPeriod"
    DEDUCTIBLE = "Deductible"


class EdgeType:
    ISSUES = "issues"
    HAS_CLAUSE = "has_clause"
    COVERS = "covers"
    EXCLUDES = "excludes"
    BELONGS_TO = "belongs_to"
    HAS_WAITING_PERIOD = "has_waiting_period"
    HAS_DEDUCTIBLE = "has_deductible"
    REQUIRES_EVENT = "requires_event"
    HAS_PREMIUM_RANGE = "has_premium_range"
    RELATED_TO = "related_to"


# ============================================================
# 疾病分类知识库
# ============================================================

DISEASE_CATEGORY_MAP = {
    "呼吸系统": ["肺炎", "支气管炎", "哮喘", "慢阻肺", "肺结核", "肺癌", "肺气肿",
                "上呼吸道感染", "流感", "新冠", "咳嗽", "感冒", "发烧", "咽炎"],
    "心血管": ["高血压", "冠心病", "心肌梗死", "心绞痛", "心力衰竭", "心律失常",
              "心脏瓣膜病", "动脉硬化", "高血脂", "中风", "脑梗", "脑出血"],
    "消化系统": ["胃炎", "胃溃疡", "阑尾炎", "胆囊炎", "胆结石", "胰腺炎",
                "肝炎", "肝硬化", "结肠炎", "肠梗阻", "痔疮", "胃癌", "肝癌"],
    "骨骼关节": ["骨折", "关节炎", "骨质疏松", "颈椎病", "腰椎间盘突出",
                "半月板损伤", "韧带撕裂", "痛风", "意外骨折", "扭伤"],
    "泌尿生殖": ["肾炎", "肾结石", "尿路感染", "前列腺增生", "子宫肌瘤",
                "卵巢囊肿", "宫颈炎", "乳腺增生", "乳腺癌"],
    "内分泌": ["糖尿病", "甲亢", "甲减", "甲状腺结节", "肥胖症"],
    "神经系统": ["癫痫", "帕金森", "阿尔茨海默", "偏头痛", "抑郁症", "焦虑症"],
    "眼科": ["白内障", "青光眼", "近视手术", "视网膜脱落"],
    "皮肤": ["湿疹", "荨麻疹", "银屑病", "带状疱疹"],
    "意外伤害": ["意外摔伤", "交通事故", "烧伤", "烫伤", "溺水", "中毒", "触电"],
    "传染": ["肺炎", "肝炎", "结核", "流感", "新冠", "水痘", "麻疹"],
    "肿瘤": ["肺癌", "肝癌", "胃癌", "乳腺癌", "甲状腺癌", "结肠癌", "白血病",
            "淋巴瘤", "良性肿瘤", "恶性肿瘤"],
    "妊娠分娩": ["顺产", "剖腹产", "流产", "宫外孕", "孕期检查"],
    "先天遗传": ["先天性心脏病", "唇腭裂", "血友病", "色盲"],
}

DISEASE_TO_CATEGORY = {}
for cat, diseases in DISEASE_CATEGORY_MAP.items():
    for d in diseases:
        DISEASE_TO_CATEGORY[d] = cat

INSURER_ALIAS = {
    "平安": "平安健康", "众安": "众安保险", "太平洋": "太平洋保险",
    "人保": "中国人保", "太平": "太平人寿", "阳光": "阳光保险",
    "泰康": "泰康在线", "新华": "新华保险", "友邦": "友邦保险",
    "国寿": "中国人寿",
}


# ============================================================
# 知识图谱主体 — Neo4j 后端
# ============================================================

class InsuranceKG:
    """保险领域知识图谱 — Neo4j 主后端 + NetworkX 降级"""

    def __init__(self):
        self._driver = None
        self._fallback_graph = None  # NetworkX fallback
        self._use_neo4j = False

    # ================================================================
    # 连接管理
    # ================================================================

    def connect(self) -> bool:
        """连接 Neo4j，失败则降级到 NetworkX"""
        if self._use_neo4j:
            return True

        try:
            from neo4j import GraphDatabase
            cfg = settings.neo4j
            self._driver = GraphDatabase.driver(
                cfg.uri, auth=(cfg.user, cfg.password)
            )
            self._driver.verify_connectivity()
            self._use_neo4j = True
            logger.info(f"[KG] Neo4j 已连接: {cfg.uri}")
            return True
        except Exception as e:
            logger.warning(f"[KG] Neo4j 连接失败 ({e})，降级为 NetworkX")
            self._use_neo4j = False
            self._init_fallback()
            return False

    def _init_fallback(self):
        """初始化 NetworkX 降级图"""
        if self._fallback_graph is not None:
            return
        import networkx as nx
        self._fallback_graph = nx.DiGraph()

    def close(self):
        """关闭 Neo4j 连接"""
        if self._driver:
            self._driver.close()
            self._driver = None
            self._use_neo4j = False

    # ================================================================
    # 构建方法
    # ================================================================

    def build_from_knowledge_base(self) -> int:
        """
        导入内置知识库到 Neo4j

        包括: 疾病分类、疾病节点、保司节点
        """
        self.connect()

        if self._use_neo4j:
            return self._neo4j_import_knowledge_base()
        else:
            return self._fallback_import_knowledge_base()

    def build_from_db(self, session) -> int:
        """从 MySQL 导入产品-保司关系"""
        self.connect()

        if self._use_neo4j:
            return self._neo4j_import_from_db(session)
        else:
            return self._fallback_import_from_db(session)

    def build_from_clauses(self, chunks: list) -> int:
        """从条款文档 chunks 抽取实体和关系"""
        self.connect()

        if self._use_neo4j:
            return self._neo4j_import_from_clauses(chunks)
        else:
            return self._fallback_import_from_clauses(chunks)

    # ================================================================
    # Neo4j 导入实现
    # ================================================================

    def _neo4j_import_knowledge_base(self) -> int:
        """Neo4j: 导入疾病分类 + 保司"""
        count = 0

        def do_import(tx):
            nonlocal count
            # ── 疾病分类 + 疾病节点 ──
            for category, diseases in DISEASE_CATEGORY_MAP.items():
                tx.run(
                    "MERGE (c:DiseaseCategory {name: $name})",
                    name=category,
                )
                count += 1

                for disease in diseases:
                    tx.run(
                        "MERGE (d:Disease {name: $name}) "
                        "SET d.category = $category",
                        name=disease, category=category,
                    )
                    count += 1
                    tx.run(
                        "MATCH (d:Disease {name: $disease}) "
                        "MATCH (c:DiseaseCategory {name: $category}) "
                        "MERGE (d)-[:belongs_to]->(c)",
                        disease=disease, category=category,
                    )
                    count += 1

            # ── 保司节点 ──
            insurers = [
                ("平安健康", ["平安", "平安人寿", "平安产险"]),
                ("众安保险", ["众安"]),
                ("太平洋保险", ["太平洋", "太平洋人寿"]),
                ("中国人保", ["人保", "人保寿险", "人保健康"]),
                ("太平人寿", ["太平"]),
                ("阳光保险", ["阳光", "阳光人寿"]),
                ("泰康在线", ["泰康", "泰康人寿"]),
                ("新华保险", ["新华"]),
                ("友邦保险", ["友邦"]),
                ("中国人寿", ["国寿"]),
            ]
            for name, aliases in insurers:
                tx.run(
                    "MERGE (i:Insurer {name: $name}) "
                    "SET i.aliases = $aliases",
                    name=name, aliases=aliases,
                )
                count += 1

        with self._driver.session() as s:
            s.execute_write(do_import)

        logger.info(f"[KG] Neo4j 知识库导入完成: {count} 个操作")
        return count

    def _neo4j_import_from_db(self, session) -> int:
        """Neo4j: 从 MySQL 导入产品-保司关系"""
        count = 0

        try:
            rows = session.execute(
                "SELECT DISTINCT insurer, product_name, product_code "
                "FROM policy_cache WHERE is_valid = TRUE"
            ).fetchall()
        except Exception as e:
            logger.warning(f"[KG] MySQL 查询失败: {e}")
            return 0

        def do_import(tx, rows_):
            nonlocal count
            for row in rows_:
                insurer_raw = row[0] or ""
                product_name = row[1] or ""
                product_code = row[2] or ""
                if not insurer_raw or not product_name:
                    continue

                insurer_name = INSURER_ALIAS.get(insurer_raw, insurer_raw)
                product_key = product_code or product_name

                tx.run("MERGE (i:Insurer {name: $name})", name=insurer_name)
                count += 1
                tx.run(
                    "MERGE (p:Product {name: $name}) "
                    "SET p.code = $code",
                    name=product_key, code=product_code,
                )
                count += 1
                tx.run(
                    "MATCH (i:Insurer {name: $insurer}) "
                    "MATCH (p:Product {name: $product}) "
                    "MERGE (i)-[:issues]->(p)",
                    insurer=insurer_name, product=product_key,
                )
                count += 1

        with self._driver.session() as s:
            s.execute_write(do_import, rows)

        logger.info(f"[KG] Neo4j DB 导入完成: {count} 个操作, {len(rows)} 条记录")
        return count

    def _neo4j_import_from_clauses(self, chunks: list) -> int:
        """Neo4j: 从条款文档抽取实体"""
        count = 0

        def do_import(tx, chunk_list):
            nonlocal count
            for chunk in chunk_list:
                text = chunk.parent_text or chunk.text
                product_name = chunk.product_name or ""
                clause_type = chunk.clause_type or ""
                if not text or not product_name:
                    continue

                product_key = chunk.product_code or product_name
                clause_key = f"{product_key}:{clause_type}"

                tx.run("MERGE (p:Product {name: $name})", name=product_key)
                count += 1

                tx.run(
                    "MERGE (c:Clause {name: $name}) "
                    "SET c.clause_type = $clause_type, c.product = $product",
                    name=clause_key, clause_type=clause_type, product=product_key,
                )
                count += 1

                tx.run(
                    "MATCH (p:Product {name: $product}) "
                    "MATCH (c:Clause {name: $clause}) "
                    "MERGE (p)-[:has_clause]->(c)",
                    product=product_key, clause=clause_key,
                )
                count += 1

                # 抽取疾病
                for disease in self._extract_diseases(text):
                    tx.run(
                        "MERGE (d:Disease {name: $name}) "
                        "SET d.category = coalesce(d.category, $category)",
                        name=disease,
                        category=DISEASE_TO_CATEGORY.get(disease, "其他"),
                    )
                    count += 1

                    is_exclusion = clause_type in ("责任免除", "免责条款", "除外责任")
                    edge_type = "excludes" if is_exclusion else "covers"
                    tx.run(
                        "MATCH (c:Clause {name: $clause}) "
                        "MATCH (d:Disease {name: $disease}) "
                        "MERGE (c)-[r:" + edge_type + "]->(d)",
                        clause=clause_key, disease=disease,
                    )
                    count += 1

                # 抽取等待期
                for days in self._extract_waiting_periods(text):
                    tx.run(
                        "MERGE (wp:WaitingPeriod {days: $days, product: $product})",
                        days=days, product=product_key,
                    )
                    count += 1
                    tx.run(
                        "MATCH (p:Product {name: $product}) "
                        "MATCH (wp:WaitingPeriod {product: $product, days: $days}) "
                        "MERGE (p)-[:has_waiting_period]->(wp)",
                        product=product_key, days=days,
                    )
                    count += 1

                # 抽取免赔额
                for amount in self._extract_deductibles(text):
                    tx.run(
                        "MERGE (dd:Deductible {amount: $amount, product: $product})",
                        amount=amount, product=product_key,
                    )
                    count += 1
                    tx.run(
                        "MATCH (p:Product {name: $product}) "
                        "MATCH (dd:Deductible {product: $product, amount: $amount}) "
                        "MERGE (p)-[:has_deductible]->(dd)",
                        product=product_key, amount=amount,
                    )
                    count += 1

                # 抽取医疗事件
                for event in self._extract_medical_events(text):
                    tx.run("MERGE (e:MedicalEvent {name: $name})", name=event)
                    count += 1
                    tx.run(
                        "MATCH (c:Clause {name: $clause}) "
                        "MATCH (e:MedicalEvent {name: $event}) "
                        "MERGE (c)-[:requires_event]->(e)",
                        clause=clause_key, event=event,
                    )
                    count += 1

        with self._driver.session() as s:
            s.execute_write(do_import, chunks)

        logger.info(f"[KG] Neo4j 条款导入完成: {count} 个操作")
        return count

    # ================================================================
    # 查询方法 (Neo4j 实现)
    # ================================================================

    def query_entity(self, name: str, node_type: str = None) -> Optional[dict]:
        """按名称查询实体（支持模糊匹配）"""
        self.connect()

        if self._use_neo4j:
            return self._neo4j_query_entity(name, node_type)
        else:
            return self._fallback_query_entity(name, node_type)

    def _neo4j_query_entity(self, name: str, node_type: str = None) -> Optional[dict]:
        query = "MATCH (n) WHERE n.name = $name"
        if node_type:
            query += f" AND n:{node_type}"
        query += " RETURN n, labels(n) as labels LIMIT 1"

        try:
            with self._driver.session() as s:
                result = s.run(query, name=name).single()
                if result:
                    node = result["n"]
                    labels = result["labels"]
                    return {
                        "name": name,
                        "type": labels[0] if labels else "",
                        "attrs": dict(node),
                    }
        except Exception as e:
            logger.warning(f"[KG] 查询实体失败: {e}")

        # 模糊匹配
        try:
            fuzzy_query = "MATCH (n) WHERE n.name CONTAINS $name"
            if node_type:
                fuzzy_query += f" AND n:{node_type}"
            fuzzy_query += " RETURN n, labels(n) as labels LIMIT 1"

            with self._driver.session() as s:
                result = s.run(fuzzy_query, name=name).single()
                if result:
                    node = result["n"]
                    labels = result["labels"]
                    return {
                        "name": node.get("name", ""),
                        "type": labels[0] if labels else "",
                        "attrs": dict(node),
                    }
        except Exception:
            pass

        return None

    def query_relations(
        self, source: str, relation_type: str = None, direction: str = "out"
    ) -> list[dict]:
        """查询关联实体"""
        self.connect()

        if self._use_neo4j:
            return self._neo4j_query_relations(source, relation_type, direction)
        else:
            return self._fallback_query_relations(source, relation_type, direction)

    def _neo4j_query_relations(
        self, source: str, relation_type: str = None, direction: str = "out"
    ) -> list[dict]:
        results = []

        if direction in ("out", "both"):
            query = "MATCH (n {name: $name})-[r]->(m) RETURN m, labels(m) as labels, type(r) as rel"
            results = self._run_relation_query(query, source, relation_type, "out")

        if direction in ("in", "both"):
            query = "MATCH (n {name: $name})<-[r]-(m) RETURN m, labels(m) as labels, type(r) as rel"
            results += self._run_relation_query(query, source, relation_type, "in")

        return results

    def _run_relation_query(
        self, base_query: str, source: str, relation_type: str, direction: str
    ) -> list[dict]:
        """执行关系查询并返回结果列表"""
        results = []
        try:
            with self._driver.session() as s:
                for record in s.run(base_query, name=source):
                    rel = record["rel"]
                    if relation_type and rel != relation_type:
                        continue
                    node = record["m"]
                    labels = record["labels"]
                    results.append({
                        "name": node.get("name", ""),
                        "type": labels[0] if labels else "",
                        "relation": rel,
                        "direction": direction,
                    })
        except Exception as e:
            logger.warning(f"[KG] 关系查询失败: {e}")
        return results

    def get_linked_entities(
        self, name: str, hops: int = 2, node_type: str = None
    ) -> list[dict]:
        """获取 N 跳关联实体"""
        self.connect()
        if self._use_neo4j:
            return self._neo4j_get_linked(name, hops, node_type)
        else:
            return self._fallback_get_linked(name, hops, node_type)

    def _neo4j_get_linked(
        self, name: str, hops: int = 2, node_type: str = None
    ) -> list[dict]:
        # Cypher 可变长度路径: (n)-[*1..hops]-(m)
        type_filter = f":{node_type}" if node_type else ""
        query = (
            f"MATCH (n {{name: $name}})-[r*1..{hops}]-(m{type_filter}) "
            "WHERE n <> m "
            "RETURN DISTINCT m, labels(m) as labels, "
            "size(r) as distance, [x in r | type(x)] as relations"
        )
        results = []
        try:
            with self._driver.session() as s:
                for record in s.run(query, name=name):
                    node = record["m"]
                    labels = record["labels"]
                    results.append({
                        "name": node.get("name", ""),
                        "type": labels[0] if labels else "",
                        "distance": record["distance"],
                        "relations": record["relations"],
                        "attrs": dict(node),
                    })
        except Exception as e:
            logger.warning(f"[KG] N跳查询失败: {e}")
        return results

    def find_path(
        self, source: str, target: str, max_length: int = 4
    ) -> Optional[list[dict]]:
        """查找最短路径"""
        self.connect()
        if self._use_neo4j:
            return self._neo4j_find_path(source, target, max_length)
        else:
            return self._fallback_find_path(source, target, max_length)

    def _neo4j_find_path(
        self, source: str, target: str, max_length: int = 4
    ) -> Optional[list[dict]]:
        query = (
            f"MATCH p=shortestPath((a {{name: $source}})-[*..{max_length}]-(b {{name: $target}})) "
            "RETURN p"
        )
        try:
            with self._driver.session() as s:
                result = s.run(query, source=source, target=target).single()
                if result:
                    path = []
                    for rel in result["p"].relationships:
                        end_node = rel.end_node
                        path.append({
                            "node": end_node.get("name", ""),
                            "relation": rel.type,
                        })
                    return path
        except Exception as e:
            logger.warning(f"[KG] 路径查询失败: {e}")
        return None

    # ================================================================
    # 统计信息
    # ================================================================

    @property
    def node_count(self) -> int:
        self.connect()
        if self._use_neo4j:
            try:
                with self._driver.session() as s:
                    return s.run("MATCH (n) RETURN count(n) as cnt").single()["cnt"]
            except Exception:
                return 0
        return (self._fallback_graph.number_of_nodes()
                if self._fallback_graph else 0)

    @property
    def edge_count(self) -> int:
        self.connect()
        if self._use_neo4j:
            try:
                with self._driver.session() as s:
                    return s.run("MATCH ()-[r]->() RETURN count(r) as cnt").single()["cnt"]
            except Exception:
                return 0
        return (self._fallback_graph.number_of_edges()
                if self._fallback_graph else 0)

    def get_stats(self) -> dict:
        """获取统计信息"""
        self.connect()
        stats = {"total_nodes": self.node_count, "total_edges": self.edge_count}

        if self._use_neo4j:
            try:
                with self._driver.session() as s:
                    node_types = {}
                    for record in s.run(
                        "MATCH (n) RETURN distinct labels(n) as label, count(n) as cnt"
                    ):
                        for label in record["label"]:
                            node_types[label] = record["cnt"]
                    stats["node_types"] = node_types

                    edge_types = {}
                    for record in s.run(
                        "MATCH ()-[r]->() RETURN type(r) as t, count(r) as cnt"
                    ):
                        edge_types[record["t"]] = record["cnt"]
                    stats["edge_types"] = edge_types
            except Exception:
                pass
        else:
            stats["node_types"] = {}
            stats["edge_types"] = {}

        return stats

    # ================================================================
    # Neo4j 专用: 清空 / 索引
    # ================================================================

    def clear_all(self):
        """清空 Neo4j 中所有节点和关系"""
        self.connect()
        if self._use_neo4j:
            with self._driver.session() as s:
                s.run("MATCH (n) DETACH DELETE n")
            logger.info("[KG] Neo4j 已清空")
        else:
            self._fallback_graph.clear()

    def create_indexes(self):
        """创建 Neo4j 索引（加速查询）"""
        self.connect()
        if not self._use_neo4j:
            return

        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (n:Disease) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Product) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Insurer) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:DiseaseCategory) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Clause) ON (n.name)",
        ]
        with self._driver.session() as s:
            for idx in indexes:
                try:
                    s.run(idx)
                except Exception as e:
                    logger.warning(f"[KG] 索引创建跳过: {e}")
        logger.info("[KG] Neo4j 索引已就绪")

    # ================================================================
    # 实体抽取辅助方法 (静态方法，与原来一致)
    # ================================================================

    @staticmethod
    def _extract_diseases(text: str) -> list[str]:
        found = set()
        for disease in DISEASE_TO_CATEGORY:
            if disease in text:
                found.add(disease)
        return list(found)

    @staticmethod
    def _extract_waiting_periods(text: str) -> list[int]:
        periods = set()
        for pat in [r"等待期[为是]?\s*(\d+)\s*天", r"等待期[为是]?\s*(\d+)\s*日"]:
            for m in re.finditer(pat, text):
                periods.add(int(m.group(1)))
        return sorted(periods)

    @staticmethod
    def _extract_deductibles(text: str) -> list[int]:
        amounts = set()
        for pat in [
            r"免赔额[为是]?\s*(\d+)\s*万\s*元?",
            r"免赔额[为是]?\s*(\d+)\s*元",
            r"免赔[为是]?\s*(\d+)\s*万",
            r"年免赔额\s*(\d+)\s*万",
        ]:
            for m in re.finditer(pat, text):
                val = int(m.group(1))
                if "万" in m.group(0):
                    val *= 10000
                amounts.add(val)
        return sorted(amounts)

    @staticmethod
    def _extract_medical_events(text: str) -> list[str]:
        events = []
        for event in ["住院", "手术", "门诊", "急诊", "ICU", "重症监护",
                       "放化疗", "透析", "器官移植", "康复治疗", "中医治疗"]:
            if event in text:
                events.append(event)
        return events

    # ================================================================
    # NetworkX 降级实现 (接口一致)
    # ================================================================

    def _fallback_import_knowledge_base(self) -> int:
        self._init_fallback()
        g = self._fallback_graph
        count = 0
        for cat, diseases in DISEASE_CATEGORY_MAP.items():
            cat_key = f"category:{cat}"
            g.add_node(cat_key, type=NodeType.DISEASE_CATEGORY, name=cat)
            count += 1
            for d in diseases:
                d_key = f"disease:{d}"
                g.add_node(d_key, type=NodeType.DISEASE, name=d, category=cat)
                count += 1
                g.add_edge(d_key, cat_key, type=EdgeType.BELONGS_TO)
                count += 1
        for name, aliases in [
            ("平安健康", ["平安"]), ("众安保险", ["众安"]), ("太平洋保险", ["太平洋"]),
            ("中国人保", ["人保"]), ("太平人寿", ["太平"]), ("阳光保险", ["阳光"]),
            ("泰康在线", ["泰康"]), ("新华保险", ["新华"]), ("友邦保险", ["友邦"]),
            ("中国人寿", ["国寿"]),
        ]:
            g.add_node(name, type=NodeType.INSURER, aliases=aliases)
            count += 1
        logger.info(f"[KG] Fallback 知识库: {count} 个操作")
        return count

    def _fallback_import_from_db(self, session) -> int:
        self._init_fallback()
        g = self._fallback_graph
        count = 0
        try:
            rows = session.execute(
                "SELECT DISTINCT insurer, product_name, product_code "
                "FROM policy_cache WHERE is_valid = TRUE"
            ).fetchall()
        except Exception as e:
            logger.warning(f"[KG] MySQL 查询失败: {e}")
            return 0

        for row in rows:
            insurer_raw, product_name, product_code = row[0] or "", row[1] or "", row[2] or ""
            if not insurer_raw or not product_name:
                continue
            insurer_name = INSURER_ALIAS.get(insurer_raw, insurer_raw)
            product_key = product_code or product_name
            if not g.has_node(insurer_name):
                g.add_node(insurer_name, type=NodeType.INSURER); count += 1
            if not g.has_node(product_key):
                g.add_node(product_key, type=NodeType.PRODUCT, name=product_name, code=product_code); count += 1
            if not g.has_edge(insurer_name, product_key):
                g.add_edge(insurer_name, product_key, type=EdgeType.ISSUES); count += 1
        logger.info(f"[KG] Fallback DB: {count} 个操作")
        return count

    def _fallback_import_from_clauses(self, chunks: list) -> int:
        self._init_fallback()
        g = self._fallback_graph
        count = 0
        for chunk in chunks:
            text = chunk.parent_text or chunk.text
            prod_name = chunk.product_name or ""
            clause_type = chunk.clause_type or ""
            if not text or not prod_name:
                continue
            product_key = chunk.product_code or prod_name
            clause_key = f"{product_key}:{clause_type}"
            if not g.has_node(product_key):
                g.add_node(product_key, type=NodeType.PRODUCT, name=prod_name); count += 1
            if not g.has_node(clause_key):
                g.add_node(clause_key, type=NodeType.CLAUSE, clause_type=clause_type, product=product_key); count += 1
            if not g.has_edge(product_key, clause_key):
                g.add_edge(product_key, clause_key, type=EdgeType.HAS_CLAUSE); count += 1
            for disease in self._extract_diseases(text):
                d_key = f"disease:{disease}"
                if not g.has_node(d_key):
                    g.add_node(d_key, type=NodeType.DISEASE, name=disease,
                              category=DISEASE_TO_CATEGORY.get(disease, "其他")); count += 1
                is_excl = clause_type in ("责任免除", "免责条款", "除外责任")
                edge = EdgeType.EXCLUDES if is_excl else EdgeType.COVERS
                if not g.has_edge(clause_key, d_key):
                    g.add_edge(clause_key, d_key, type=edge); count += 1
        logger.info(f"[KG] Fallback 条款: {count} 个操作")
        return count

    def _fallback_query_entity(self, name: str, node_type: str = None) -> Optional[dict]:
        g = self._fallback_graph
        if name in g:
            attrs = dict(g.nodes[name])
            if node_type is None or attrs.get("type") == node_type:
                return {"name": name, "type": attrs.get("type", ""), "attrs": attrs}
        for node, attrs in g.nodes(data=True):
            node_name = attrs.get("name", node)
            if node_name == name or name in str(node_name):
                if node_type is None or attrs.get("type") == node_type:
                    return {"name": node, "type": attrs.get("type", ""), "attrs": dict(attrs)}
        return None

    def _fallback_query_relations(self, source: str, relation_type: str = None,
                                   direction: str = "out") -> list[dict]:
        g = self._fallback_graph
        if source not in g:
            return []
        results = []
        if direction in ("out", "both"):
            for _, target, attrs in g.out_edges(source, data=True):
                etype = attrs.get("type", "")
                if relation_type and etype != relation_type:
                    continue
                tattrs = dict(g.nodes[target])
                results.append({"name": target, "type": tattrs.get("type", ""),
                                "relation": etype, "direction": "out"})
        if direction in ("in", "both"):
            for pred, _, attrs in g.in_edges(source, data=True):
                etype = attrs.get("type", "")
                if relation_type and etype != relation_type:
                    continue
                pattrs = dict(g.nodes[pred])
                results.append({"name": pred, "type": pattrs.get("type", ""),
                                "relation": etype, "direction": "in"})
        return results

    def _fallback_get_linked(self, name: str, hops: int = 2,
                              node_type: str = None) -> list[dict]:
        g = self._fallback_graph
        from collections import deque
        if name not in g:
            entity = self._fallback_query_entity(name)
            if entity:
                name = entity["name"]
            else:
                return []
        results, visited, queue = [], {name}, deque([(name, 0, [name])])
        while queue:
            current, dist, path = queue.popleft()
            if dist > hops:
                continue
            if dist > 0:
                cattrs = dict(g.nodes[current])
                if node_type is None or cattrs.get("type") == node_type:
                    results.append({"name": current, "type": cattrs.get("type", ""),
                                    "distance": dist, "path": path, "attrs": cattrs})
            if dist < hops:
                for neighbor in list(g.neighbors(current)) + list(g.predecessors(current)):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1, path + [neighbor]))
        return results

    def _fallback_find_path(self, source: str, target: str,
                             max_length: int = 4) -> Optional[list[dict]]:
        g = self._fallback_graph
        import networkx as nx
        try:
            path = nx.shortest_path(g, source, target)
            if len(path) - 1 > max_length:
                return None
            result = []
            for i in range(len(path) - 1):
                edge_data = g.get_edge_data(path[i], path[i + 1])
                result.append({"node": path[i + 1],
                               "relation": edge_data.get("type", "") if edge_data else ""})
            return result
        except nx.NetworkXNoPath:
            return None

    def __repr__(self) -> str:
        backend = "Neo4j" if self._use_neo4j else "NetworkX(fallback)"
        return (f"InsuranceKG(nodes={self.node_count}, edges={self.edge_count}, "
                f"backend={backend})")

    def __del__(self):
        self.close()


# ============================================================
# 单例工厂
# ============================================================

_kg_instance: Optional[InsuranceKG] = None


def get_kg() -> InsuranceKG:
    """获取知识图谱单例"""
    global _kg_instance
    if _kg_instance is None:
        _kg_instance = InsuranceKG()
    return _kg_instance
