# -*- coding: utf-8 -*-
"""
Milvus 向量存储 — 保险条款的向量化存储与检索

核心职责:
  1. 创建/管理 Milvus Collection (包含 12 个标量字段 + 向量字段)
  2. 执行混合检索 (Dense 语义 + Sparse 关键词 + 标量过滤)
  3. RRF (Reciprocal Rank Fusion) 融合两路检索结果

Milvus Schema 设计:
  每个 chunk 不只是向量，还带着丰富的业务元数据:
    - insurer / product_name / product_code → 支持按保司/产品过滤
    - doc_type / clause_type → 支持按文档类型/条款类型过滤
    - chunk_type / parent_id → 支持 Parent-Child 检索策略
    - is_valid / version → 支持版本管理（旧条款标记失效）
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

from base.logger import logger
from config.settings import settings


# ============================================================
# 检索结果数据结构
# ============================================================


@dataclass
class ChunkResult:
    """
    Milvus 检索返回的单个 chunk

    Attributes:
        chunk_id: 唯一标识，格式: {doc_id}_child_0001 或 {doc_id}_parent_0001
        text: chunk 文本内容
        score: 检索相似度分数 (0-1)
        insurer: 保险公司名称
        product_name: 产品名称
        product_code: 产品编码
        doc_type: 文档类型 (条款/投保须知/费率表/理赔指南)
        clause_type: 条款章节 (保险责任/责任免除/释义 等)
        chunk_type: "child" 或 "parent"
        parent_id: 父块 ID（child 块才有，parent 块为 None）
        parent_text: 父块文本（检索后回查补全）
    """

    chunk_id: str
    text: str
    score: float = 0.0
    insurer: str = ""
    product_name: str = ""
    product_code: str = ""
    doc_type: str = ""
    clause_type: str = ""
    chunk_type: str = "child"
    parent_id: Optional[str] = None
    parent_text: Optional[str] = None


# ============================================================
# Milvus Collection Schema 定义
# ============================================================

# Collection 中的字段定义
# 这些字段在创建 Collection 时注册，检索时作为过滤条件
COLLECTION_FIELDS = [
    # ── 主键 ──
    {"name": "id", "type": "VARCHAR", "max_length": 100, "is_primary": True},
    # ── 文本字段 ──
    {"name": "text", "type": "VARCHAR", "max_length": 65535},
    # ── 向量字段 ──
    # Dense: BGE-M3 输出的 1024 维语义向量（捕捉语义相似度）
    {"name": "dense_vector", "type": "FLOAT_VECTOR", "dim": 1024},
    # Sparse: BGE-M3 输出的稀疏词汇向量（捕捉关键词精确匹配）
    {"name": "sparse_vector", "type": "SPARSE_FLOAT_VECTOR"},
    # ── 业务标量字段（用于过滤）──
    {"name": "insurer", "type": "VARCHAR", "max_length": 100},  # 保司名
    {"name": "product_name", "type": "VARCHAR", "max_length": 200},  # 产品名
    {"name": "product_code", "type": "VARCHAR", "max_length": 50},  # 产品编码
    {"name": "doc_type", "type": "VARCHAR", "max_length": 50},  # 文档类型
    {"name": "clause_type", "type": "VARCHAR", "max_length": 100},  # 条款章节
    {"name": "chunk_type", "type": "VARCHAR", "max_length": 10},  # child/parent
    {"name": "parent_id", "type": "VARCHAR", "max_length": 100},  # 父块 ID
    {"name": "is_valid", "type": "BOOL"},  # 是否有效
    {"name": "version", "type": "VARCHAR", "max_length": 20},  # 版本号
]


# ============================================================
# Milvus 向量存储类
# ============================================================


class VectorStore:
    """
    Milvus 向量存储

    负责:
      1. Collection 生命周期管理（创建、加载、释放）
      2. 数据写入（批量插入 chunks）
      3. 混合检索（Dense + Sparse + 标量过滤 + RRF 融合）
      4. Parent-Child 回查（子块检索后补全父块上下文）
    """

    def __init__(self):
        """
        初始化: 连接 Milvus 并加载 Collection

        注意: 这里不直接 import pymilvus，而是延迟导入。
        因为在开发/测试环境中可能没有 Milvus，直接 import 会报错。
        """
        self.collection = None
        self._connected = False

    def connect(self):
        """
        连接 Milvus 服务器并加载 Collection 到内存

        Milvus 的 Collection 需要先 load() 到内存才能检索。
        load() 会把索引加载到内存，检索速度从秒级降到毫秒级。
        """
        try:
            from pymilvus import connections, Collection

            cfg = settings.milvus

            # 建立连接（幂等操作，重复调用不会报错）
            connections.connect(
                alias="default",
                host=cfg.host,
                port=cfg.port,
            )

            # 加载 Collection
            self.collection = Collection(cfg.collection)
            self.collection.load()  # 加载到内存，加速检索
            self._connected = True

            logger.info(f"Milvus 连接成功: {cfg.host}:{cfg.port}/{cfg.collection}")

        except Exception as e:
            logger.error(f"Milvus 连接失败: {e}")
            raise

    def ensure_collection(self):
        """
        确保 Collection 存在（不存在则创建）

        创建时会:
          1. 定义 Schema（字段类型、维度）
          2. 创建 Dense 索引: IVF_FLAT（倒排文件，适合百万级数据）
          3. 创建 Sparse 索引: SPARSE_INVERTED_INDEX（稀疏倒排，用于 BM25）
        """
        if not self._connected:
            self.connect()

        from pymilvus import (
            Collection,
            FieldSchema,
            CollectionSchema,
            DataType,
            utility,
        )

        collection_name = settings.milvus.collection

        # 如果 Collection 已存在，直接使用
        if utility.has_collection(collection_name):
            self.collection = Collection(collection_name)
            self.collection.load()
            logger.info(f"Collection '{collection_name}' 已存在，直接加载")
            return

        # ── 创建 Schema ──
        fields = []
        for f in COLLECTION_FIELDS:
            kwargs = {"name": f["name"], "dtype": getattr(DataType, f["type"])}
            if "max_length" in f:
                kwargs["max_length"] = f["max_length"]
            if "dim" in f:
                kwargs["dim"] = f["dim"]
            if f.get("is_primary"):
                kwargs["is_primary"] = True
            fields.append(FieldSchema(**kwargs))

        schema = CollectionSchema(
            fields=fields,
            description="保险条款向量存储",
            enable_dynamic_field=False,
        )

        self.collection = Collection(
            name=collection_name,
            schema=schema,
        )

        # ── 创建索引 ──

        # Dense 向量索引: IVF_FLAT
        # IVF_FLAT = 倒排文件 + 精确距离计算
        # nlist=128: 将向量空间分成 128 个簇，检索时只搜索最近的几个簇
        # 适合百万级数据，召回率高
        self.collection.create_index(
            field_name="dense_vector",
            index_params={
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",  # 余弦相似度（BGE-M3 输出已归一化）
                "params": {"nlist": 128},
            },
        )

        # Sparse 向量索引: SPARSE_INVERTED_INDEX
        # 类似 BM25 的倒排索引，用于关键词精确匹配
        self.collection.create_index(
            field_name="sparse_vector",
            index_params={
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "IP",  # 内积（稀疏向量用内积）
            },
        )

        # 加载到内存
        self.collection.load()
        logger.info(f"Collection '{collection_name}' 创建完成，索引已建立")

    # ============================================================
    # 数据写入
    # ============================================================

    def insert(self, chunks: list[dict]):
        """
        批量插入 chunks 到 Milvus

        Args:
            chunks: chunk 列表，每个 chunk 是一个 dict，包含:
                - chunk_id: 唯一 ID
                - text: 文本内容
                - dense_vector: 1024 维 Dense 向量 (numpy array 或 list)
                - sparse_vector: 稀疏向量 (dict 格式 {token_id: weight})
                - insurer, product_name, product_code, doc_type, clause_type
                - chunk_type: "child" 或 "parent"
                - parent_id: 父块 ID（child 块有，parent 块为空字符串）
                - is_valid: True
                - version: "1.0"

        注意:
          插入后需要调用 flush() 才能在检索中看到。
          但不要每次 insert 都 flush，批量插入完再统一 flush。
        """
        if not self._connected:
            self.connect()

        # 将 dict 列表转为 Milvus 需要的列式格式
        entities = {
            "id": [c["chunk_id"] for c in chunks],
            "text": [c["text"] for c in chunks],
            "dense_vector": [
                c["dense_vector"].tolist()
                if isinstance(c["dense_vector"], np.ndarray)
                else c["dense_vector"]
                for c in chunks
            ],
            "sparse_vector": [c.get("sparse_vector", {}) for c in chunks],
            "insurer": [c.get("insurer", "") for c in chunks],
            "product_name": [c.get("product_name", "") for c in chunks],
            "product_code": [c.get("product_code", "") for c in chunks],
            "doc_type": [c.get("doc_type", "") for c in chunks],
            "clause_type": [c.get("clause_type", "") for c in chunks],
            "chunk_type": [c.get("chunk_type", "child") for c in chunks],
            "parent_id": [c.get("parent_id", "") for c in chunks],
            "is_valid": [c.get("is_valid", True) for c in chunks],
            "version": [c.get("version", "1.0") for c in chunks],
        }

        result = self.collection.insert(entities)
        logger.info(
            f"Milvus 插入 {len(chunks)} 条 chunks, insert_count={result.insert_count}"
        )

    def flush(self):
        """
        刷新数据 — 使新插入的数据可检索

        Milvus 是 WAL (Write-Ahead Log) 架构，
        insert 后数据在 WAL 中，flush 后才写入 Segment 并可检索。
        生产环境中不要频繁 flush，建议批量插入完再统一 flush。
        """
        if self.collection:
            self.collection.flush()
            logger.info("Milvus flush 完成")

    # ============================================================
    # 混合检索
    # ============================================================

    def search(
        self,
        dense_vector: np.ndarray,
        sparse_vector: dict = None,
        filters: dict = None,
        top_k: int = 30,
        include_parent: bool = True,
    ) -> list[ChunkResult]:
        """
        混合检索 — Dense 语义 + Sparse 关键词 + 标量过滤 + RRF 融合

        Args:
            dense_vector: BGE-M3 编码的 Dense 向量 (1024维)
            sparse_vector: BGE-M3 编码的 Sparse 向量 (可选)
            filters: 标量过滤条件，如 {"insurer": "平安", "product_name": "e生保"}
            top_k: 返回 Top-K 结果
            include_parent: 是否回查父块文本（small2big 策略）

        Returns:
            list[ChunkResult]: 按 RRF 分数排序的结果列表

        检索流程:
          1. 构造 Milvus 过滤表达式（标量字段过滤）
          2. Dense 检索: ANN 搜索，余弦相似度
          3. Sparse 检索: BM25 式倒排索引，内积
          4. RRF 融合: 两路结果按排名融合
          5. (可选) 回查父块: 子块 → parent_id → 补全上下文
        """
        if not self._connected:
            self.connect()

        # ── 步骤 1: 构造过滤表达式 ──
        filter_expr = self._build_filter_expr(filters)

        # ── 步骤 2: Dense 语义检索 ──
        # nprobe=16: 搜索 IVF_FLAT 中最近的 16 个簇
        # 值越大召回率越高但速度越慢，16 是性价比最优值
        dense_results = self.collection.search(
            data=[dense_vector.tolist()],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=top_k,
            expr=filter_expr,
            output_fields=[
                "text",
                "insurer",
                "product_name",
                "product_code",
                "doc_type",
                "clause_type",
                "chunk_type",
                "parent_id",
            ],
        )

        # ── 步骤 3: Sparse 关键词检索 (可选) ──
        sparse_results = None
        if sparse_vector:
            sparse_results = self.collection.search(
                data=[sparse_vector],
                anns_field="sparse_vector",
                param={"metric_type": "IP"},
                limit=top_k,
                expr=filter_expr,
                output_fields=[
                    "text",
                    "insurer",
                    "product_name",
                    "product_code",
                    "doc_type",
                    "clause_type",
                    "chunk_type",
                    "parent_id",
                ],
            )

        # ── 步骤 4: RRF 融合 ──
        merged = self._rrf_merge(dense_results, sparse_results, k=60)

        # 截取 Top-K
        merged = merged[:top_k]

        # ── 步骤 5: 回查父块 (small2big) ──
        if include_parent:
            merged = self._enrich_with_parent(merged)

        logger.info(f"Milvus 检索完成: {len(merged)} 条结果, filter={filter_expr}")
        return merged

    # ============================================================
    # 内部方法
    # ============================================================

    # ============================================================
    # 辅助方法
    # ============================================================

    @staticmethod
    def _escape_filter_value(value: str) -> str:
        """转义 Milvus 过滤表达式中的特殊字符（双引号、反斜杠）"""
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def _build_filter_expr(self, filters: dict) -> str:
        """
        将 dict 过滤条件转换为 Milvus 表达式字符串

        输入: {"insurer": "平安", "product_name": "e生保"}
        输出: 'insurer == "平安" and product_name like "e生保%"'

        Milvus 表达式语法:
          ==  精确匹配
          like  前缀匹配 (只支持后缀通配符 %)
          in    集合匹配: clause_type in ["保险责任", "释义"]
          and/or/not  逻辑组合

        默认过滤:
          is_valid == True  → 只检索有效条款（旧版本条款标记为 False）
          chunk_type == "child"  → 先检索子块，再回查父块
        """
        conditions = [
            "is_valid == true",  # 只检索有效条款
            'chunk_type == "child"',  # 先检索子块（更精确）
        ]

        if filters:
            for key, value in filters.items():
                if key == "product_name":
                    # 产品名用前缀匹配（用户可能只输入部分产品名）
                    escaped_val = self._escape_filter_value(value)
                    conditions.append(f'{key} like "{escaped_val}%"')
                elif key == "clause_type" and isinstance(value, list):
                    # 条款类型支持列表: ["保险责任", "释义"]
                    quoted = ", ".join(
                        f'"{self._escape_filter_value(v)}"' for v in value
                    )
                    conditions.append(f"{key} in [{quoted}]")
                else:
                    escaped_val = self._escape_filter_value(str(value))
                    conditions.append(f'{key} == "{escaped_val}"')

        return " and ".join(conditions)

    def _rrf_merge(
        self,
        dense_results,
        sparse_results,
        k: int = 60,
    ) -> list[ChunkResult]:
        """
        RRF (Reciprocal Rank Fusion) 融合两路检索结果

        RRF 公式: score(d) = Σ 1 / (k + rank_i(d))
          - k=60 是标准值，防止排名靠前的结果权重过大
          - 如果一个文档同时出现在 Dense 第1名和 Sparse 第3名:
            score = 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323

        为什么用 RRF 而不是加权求和:
          Dense 和 Sparse 的分数量纲不同（余弦 vs 内积），
          直接加权需要调参。RRF 只看排名不看分数，天然归一化。

        Args:
            dense_results: Dense 检索结果 (Milvus SearchResult)
            sparse_results: Sparse 检索结果 (可选)
            k: RRF 常数，默认 60

        Returns:
            按 RRF 分数降序排列的 ChunkResult 列表
        """
        # 用 dict 累加每个 chunk 的 RRF 分数
        scores = {}  # {chunk_id: rrf_score}
        chunk_data = {}  # {chunk_id: ChunkResult}

        # ── 处理 Dense 结果 ──
        if dense_results and len(dense_results) > 0:
            for rank, hit in enumerate(dense_results[0]):
                chunk_id = hit.id
                rrf_score = 1.0 / (k + rank + 1)  # rank 从 0 开始
                scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score
                chunk_data[chunk_id] = ChunkResult(
                    chunk_id=chunk_id,
                    text=hit.entity.get("text", ""),
                    score=hit.score,
                    insurer=hit.entity.get("insurer", ""),
                    product_name=hit.entity.get("product_name", ""),
                    product_code=hit.entity.get("product_code", ""),
                    doc_type=hit.entity.get("doc_type", ""),
                    clause_type=hit.entity.get("clause_type", ""),
                    chunk_type=hit.entity.get("chunk_type", "child"),
                    parent_id=hit.entity.get("parent_id", ""),
                )

        # ── 处理 Sparse 结果 ──
        if sparse_results and len(sparse_results) > 0:
            for rank, hit in enumerate(sparse_results[0]):
                chunk_id = hit.id
                rrf_score = 1.0 / (k + rank + 1)
                scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score
                # 如果 Dense 没有这个 chunk，从 Sparse 结果创建
                if chunk_id not in chunk_data:
                    chunk_data[chunk_id] = ChunkResult(
                        chunk_id=chunk_id,
                        text=hit.entity.get("text", ""),
                        score=hit.score,
                        insurer=hit.entity.get("insurer", ""),
                        product_name=hit.entity.get("product_name", ""),
                        product_code=hit.entity.get("product_code", ""),
                        doc_type=hit.entity.get("doc_type", ""),
                        clause_type=hit.entity.get("clause_type", ""),
                        chunk_type=hit.entity.get("chunk_type", "child"),
                        parent_id=hit.entity.get("parent_id", ""),
                    )

        # ── 按 RRF 分数降序排列 ──
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        results = []
        for cid in sorted_ids:
            chunk = chunk_data[cid]
            chunk.score = scores[cid]  # 用 RRF 分数覆盖原始分数
            results.append(chunk)

        return results

    def _enrich_with_parent(self, results: list[ChunkResult]) -> list[ChunkResult]:
        """
        Parent-Child 回查: 子块检索后补全父块上下文

        为什么需要:
          子块 (~512字) 检索精度高但上下文不完整
          父块 (~2000字) 上下文完整但检索精度低
          策略: 用子块检索（精度），返回时补上父块（上下文）

        流程:
          1. 收集所有子块的 parent_id
          2. 批量查询父块文本
          3. 回填到子块的 parent_text 字段
        """
        # 收集需要回查的 parent_id（去重）
        parent_ids = list(
            set(r.parent_id for r in results if r.parent_id and r.chunk_type == "child")
        )

        if not parent_ids:
            return results

        # 批量查询父块
        # 用 Milvus query (不是 search) — 通过主键精确查询
        try:
            parent_expr = "id in [" + ", ".join(f'"{pid}"' for pid in parent_ids) + "]"
            parent_hits = self.collection.query(
                expr=parent_expr,
                output_fields=["text"],
            )
            # 建立 parent_id → text 的映射
            parent_texts = {hit["id"]: hit["text"] for hit in parent_hits}
        except Exception as e:
            logger.warning(f"父块回查失败: {e}")
            return results

        # 回填 parent_text
        for chunk in results:
            if chunk.parent_id and chunk.parent_id in parent_texts:
                chunk.parent_text = parent_texts[chunk.parent_id]

        return results

    def delete_by_product(self, insurer: str, product_code: str):
        """
        按保司+产品删除 chunks（产品下架时使用）

        不是物理删除，而是标记 is_valid=False。
        这里为了简单直接删除，生产环境建议标记。
        """
        if not self._connected:
            self.connect()

        expr = f'insurer == "{insurer}" and product_code == "{product_code}"'
        self.collection.delete(expr)
        self.collection.flush()
        logger.info(f"已删除: {insurer}/{product_code} 的所有 chunks")

    def invalidate_by_product(self, insurer: str, product_code: str, version: str):
        """
        产品更新时: 旧版本标记失效，新版本保持有效

        用法: 新版本 chunks 插入后，调用此方法把旧版本标记为 is_valid=False。
        这样检索时 filter='is_valid == true' 自动过滤掉旧版本。
        """
        if not self._connected:
            self.connect()

        # 查找旧版本的 chunk IDs
        expr = (
            f'insurer == "{insurer}" '
            f'and product_code == "{product_code}" '
            f'and version != "{version}"'
        )
        old_chunks = self.collection.query(expr, output_fields=["id"])

        if old_chunks:
            old_ids = [c["id"] for c in old_chunks]
            # Milvus 不支持单字段 UPDATE，采用「删旧+重插」策略：
            # 先查询旧 chunk 的完整实体，改 is_valid=False 后重新插入
            old_entities = self.collection.query(
                expr, output_fields=["*"],
            )
            if old_entities:
                # 标记失效
                for e in old_entities:
                    e["is_valid"] = False
                # 删除旧实体，然后重新插入失效版本
                self.collection.delete(expr)
                self.collection.flush()
                self.collection.insert(old_entities)
                self.collection.flush()
                logger.info(
                    f"标记 {len(old_ids)} 条旧版本 chunks 失效: {insurer}/{product_code}"
                )
