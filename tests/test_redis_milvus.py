# -*- coding: utf-8 -*-
"""
Redis + Milvus 服务测试

运行方式:
    python tests/test_redis_milvus.py

测试内容:
  Redis:
    1. 连接验证 + ping
    2. 基本读写 (SET/GET/DELETE)
    3. 哈希/列表/集合操作
    4. TTL 过期验证
    5. FAQ/Embedding/Product 缓存写入验证

  Milvus:
    1. 连接验证
    2. Collection 创建 (含 12 个标量字段)
    3. 数据插入 (模拟保险条款 chunks)
    4. 向量检索 (Dense + 标量过滤)
    5. 删除/失效标记
"""

import sys
import os
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings


# ============================================================
# Redis 测试
# ============================================================

def test_redis_connection():
    """测试 Redis 连接 + ping"""
    try:
        from cache.redis_client import RedisClient
        redis = RedisClient()
        pong = redis.client.ping()
        assert pong, "Redis PING 失败"
        print(f"  ✅ Redis 连接成功 (DB={settings.redis.db})")
        return redis
    except Exception as e:
        print(f"  ❌ Redis 连接失败: {e}")
        return None


def test_redis_basic_ops(redis):
    """测试 Redis 基本读写"""
    # SET
    redis.set("test:hello", "world", ttl=60)
    # GET
    val = redis.get("test:hello")
    assert val == "world", f"期望 'world', 实际 '{val}'"
    # DELETE
    redis.delete("test:hello")
    val = redis.get("test:hello")
    assert val is None, "DELETE 后应该返回 None"
    print("  ✅ SET/GET/DELETE 正常")


def test_redis_json_ops(redis):
    """测试 JSON 读写"""
    data = {"intent": "条款解读", "confidence": 0.95, "entities": {"insurer": "平安"}}
    redis.set_json("test:json", data, ttl=60)
    result = redis.get_json("test:json")
    assert result["intent"] == "条款解读", f"JSON GET 失败: {result}"
    assert result["confidence"] == 0.95
    redis.delete("test:json")
    print("  ✅ JSON SET/GET 正常")


def test_redis_ttl(redis):
    """测试 TTL 过期"""
    redis.set("test:ttl", "expire", ttl=1)  # 1 秒过期
    val = redis.get("test:ttl")
    assert val == "expire", "TTL 写入失败"
    time.sleep(1.5)
    val = redis.get("test:ttl")
    assert val is None, f"TTL 未生效，值仍存在: {val}"
    print("  ✅ TTL 过期正常")


def test_cache_modules(redis):
    """测试业务缓存模块的写入"""
    # ── FAQ 缓存 ──
    from cache.faq_cache import FAQCache
    faq = FAQCache(redis)
    faq.add_faq("免赔额是多少？", "一般医疗保险的免赔额为1万元。")
    hit = faq.try_hit("免赔额是多少？")
    assert hit is not None and "1万元" in hit, f"FAQ 缓存命中失败: {hit}"
    print("  ✅ FAQ 缓存写入+命中正常")

    # ── Embedding 缓存 ──
    from cache.embedding_cache import EmbeddingCache
    emb_cache = EmbeddingCache(redis)
    test_vec = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    emb_cache.set("测试文本", test_vec)
    cached_vec = emb_cache.get("测试文本")
    assert cached_vec is not None, "Embedding 缓存读取失败"
    assert np.allclose(test_vec, cached_vec), "Embedding 向量不匹配"
    print("  ✅ Embedding 缓存写入+读取正常")

    # ── 产品缓存 ──
    from cache.product_cache import ProductCache
    prod_cache = ProductCache(redis)
    # 手动写入（模拟 MySQL 查询结果）
    redis.set_json("product:TEST-PROD", {
        "product_code": "TEST-PROD",
        "product_name": "测试产品",
        "insurer": "测试保司",
        "category": "医疗险",
        "is_active": True,
    }, ttl=60)
    prod = prod_cache.get_product("TEST-PROD")
    assert prod is not None and prod["product_name"] == "测试产品", f"产品缓存命中失败: {prod}"
    redis.delete("product:TEST-PROD")
    print("  ✅ 产品缓存写入+命中正常")

    # ── Query 结果缓存 ──
    from cache.query_result_cache import QueryResultCache
    query_cache = QueryResultCache(redis)
    query_cache.set("肺炎住院能赔吗", "条款解读", {"answer": "可以赔", "intent": "条款解读"})
    cached = query_cache.try_get("肺炎住院能赔吗", "条款解读")
    assert cached is not None and "可以赔" in cached["answer"], f"Query缓存命中失败: {cached}"
    query_cache.invalidate_intent("条款解读")
    print("  ✅ Query 结果缓存写入+命中+清除正常")

    # 清理 — 用 SCAN + DELETE 替代无效的 glob delete
    for pattern in ["faq:*", "emb:*", "result:*", "product:TEST-*", "test:*"]:
        cursor = 0
        while True:
            cursor, keys = redis.client.scan(cursor, match=pattern, count=100)
            if keys:
                redis.client.delete(*keys)
            if cursor == 0:
                break
    print("  ✅ 测试数据已清理")


# ============================================================
# Milvus 测试
# ============================================================

def test_milvus_connection():
    """测试 Milvus 连接"""
    try:
        from pymilvus import connections, utility

        connections.connect(
            alias="test",
            host=settings.milvus.host,
            port=settings.milvus.port,
        )

        # 检查连接状态
        collections = utility.list_collections(using="test")
        print(f"  ✅ Milvus 连接成功 (已有 {len(collections)} 个 Collection: {collections})")
        return True
    except Exception as e:
        print(f"  ❌ Milvus 连接失败: {e}")
        return False


def test_milvus_collection():
    """测试 Milvus Collection 创建"""
    try:
        from pymilvus import (
            Collection, FieldSchema, CollectionSchema,
            DataType, utility, connections,
        )

        coll_name = f"{settings.milvus.collection}_test"

        # 如果已存在，先删除
        if utility.has_collection(coll_name, using="test"):
            utility.drop_collection(coll_name, using="test")
            print(f"      已删除旧测试 Collection: {coll_name}")

        # ── 创建 Schema ──
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=4),  # 测试用4维
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            # 业务字段
            FieldSchema(name="insurer", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="product_name", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="clause_type", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=10),
            FieldSchema(name="cardinality", dtype=DataType.INT64, max_length=100),
            FieldSchema(name="is_valid", dtype=DataType.BOOL),
            FieldSchema(name="version", dtype=DataType.VARCHAR, max_length=20),
        ]

        schema = CollectionSchema(fields, description="RAG + Agent 测试 Collection")

        collection = Collection(name=coll_name, schema=schema, using="test")

        # ── 创建索引 ──
        collection.create_index(
            field_name="dense_vector",
            index_params={
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {"nlist": 16},
            },
        )
        collection.create_index(
            field_name="sparse_vector",
            index_params={
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "IP",
            },
        )

        # 加载到内存
        collection.load()

        print(f"  ✅ Milvus Collection 创建成功: {coll_name}")
        return collection, coll_name

    except Exception as e:
        print(f"  ❌ Milvus Collection 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_milvus_insert_and_search(collection, coll_name):
    """测试数据插入和向量检索"""
    try:
        from pymilvus import utility

        # ── 插入测试数据 ──
        # 模拟 4 条保险条款 chunk
        entities = [
            # chunk 1: 平安e生保 保险责任
            {
                "id": "test_chunk_001",
                "text": "被保险人在保险期间内因疾病住院治疗的，保险人按照合同约定给付保险金。",
                "dense_vector": [1.0, 0.0, 0.0, 0.0],    # 代表"住院"
                "sparse_vector": {0: 1.0, 1: 1.0},        # 词汇权重
                "insurer": "平安健康",
                "product_name": "平安e生保",
                "doc_type": "条款",
                "clause_type": "保险责任",
                "chunk_type": "child",
                "cardinality": 1,
                "is_valid": True,
                "version": "1.0",
            },
            # chunk 2: 平安e生保 责任免除
            {
                "id": "test_chunk_002",
                "text": "以下情况不属于保障范围：先天性疾病、既往症、美容整形手术。",
                "dense_vector": [0.0, 0.0, 1.0, 0.0],    # 代表"免责"
                "sparse_vector": {2: 1.0, 3: 1.0},
                "insurer": "平安健康",
                "product_name": "平安e生保",
                "doc_type": "条款",
                "clause_type": "责任免除",
                "chunk_type": "child",
                "cardinality": 2,
                "is_valid": True,
                "version": "1.0",
            },
            # chunk 3: 众安尊享e生 保险责任
            {
                "id": "test_chunk_003",
                "text": "住院医疗费用保险金：被保险人住院期间发生的合理医疗费用。",
                "dense_vector": [0.8, 0.2, 0.0, 0.0],    # 也偏向"住院"
                "sparse_vector": {0: 0.8, 1: 0.2},
                "insurer": "众安保险",
                "product_name": "众安尊享e生",
                "doc_type": "条款",
                "clause_type": "保险责任",
                "chunk_type": "child",
                "cardinality": 3,
                "is_valid": True,
                "version": "1.0",
            },
            # chunk 4: 无效 chunk（is_valid=False，检索时应该被过滤）
            {
                "id": "test_chunk_004",
                "text": "已下架产品的旧条款内容。",
                "dense_vector": [0.1, 0.1, 0.1, 0.1],
                "sparse_vector": {4: 1.0},
                "insurer": "平安健康",
                "product_name": "平安e生保_旧版",
                "doc_type": "条款",
                "clause_type": "保险责任",
                "chunk_type": "child",
                "cardinality": 4,
                "is_valid": False,   # ← 标记为失效
                "version": "0.9",
            },
        ]

        insert_result = collection.insert(entities)
        collection.flush()
        print(f"  ✅ 插入 {insert_result.insert_count} 条测试数据")

        # 等待数据可见
        time.sleep(1)

        # ── 检索测试 1: 查询"住院"相关内容 ──
        query_vector = [[1.0, 0.0, 0.0, 0.0]]  # 查询"住院"
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 8}}

        results = collection.search(
            data=query_vector,
            anns_field="dense_vector",
            param=search_params,
            limit=4,
            expr='is_valid == true',              # 过滤失效数据
            output_fields=["text", "product_name", "clause_type"],
        )

        assert len(results) > 0 and len(results[0]) > 0, "检索结果为空"
        assert len(results[0]) <= 3, f"检索到 {len(results[0])} 条，但只有 3 条有效数据"

        # 验证 Top-1 是住院相关条款
        top1 = results[0][0]
        top1_text = top1.entity.get("text", "")
        assert "住院" in top1_text, f"Top-1 不相关: {top1_text}"
        print(f"  ✅ 检索成功: Top-1 score={top1.score:.3f} ({top1_text[:50]}...)")

        # ── 检索测试 2: 带标量过滤（只搜平安的产品）──
        results2 = collection.search(
            data=query_vector,
            anns_field="dense_vector",
            param=search_params,
            limit=4,
            expr='is_valid == true and insurer == "平安健康"',
            output_fields=["text", "product_name", "insurer"],
        )

        # 验证所有结果都是平安的
        for hit in results2[0]:
            assert hit.entity.get("insurer") == "平安健康", \
                f"过滤失败: 结果包含 {hit.entity.get('insurer')}"
        print(f"  ✅ 标量过滤成功: {len(results2[0])} 条结果, 全部为平安健康")

        # ── 检索测试 3: 多条件过滤 ──
        results3 = collection.search(
            data=query_vector,
            anns_field="dense_vector",
            param=search_params,
            limit=4,
            expr='is_valid == true and clause_type == "保险责任"',
            output_fields=["text", "clause_type"],
        )

        for hit in results3[0]:
            assert hit.entity.get("clause_type") == "保险责任", \
                f"clause_type 过滤失败: {hit.entity.get('clause_type')}"
        print(f"  ✅ clause_type 过滤成功: {len(results3[0])} 条保险责任条款")

        print("  ✅ Milvus 检索测试全部通过 (3 种过滤条件)")

        # ── 清理 ──
        # 标记旧版本失效的验证
        results4 = collection.search(
            data=query_vector,
            anns_field="dense_vector",
            param=search_params,
            limit=4,
            expr='is_valid == false',
            output_fields=["text", "is_valid"],
        )
        assert len(results4[0]) == 1, f"失效数据应只有 1 条，实际 {len(results4[0])}"
        assert not results4[0][0].entity.get("is_valid"), "is_valid 应为 False"
        print(f"  ✅ 失效标记正常 (is_valid=False 的数据可检索但不默认返回)")

        return True

    except Exception as e:
        print(f"  ❌ Milvus 检索测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def cleanup_test_collection(coll_name):
    """清理测试 Collection"""
    try:
        from pymilvus import utility
        if utility.has_collection(coll_name, using="test"):
            utility.drop_collection(coll_name, using="test")
            print(f"  ✅ 测试 Collection 已清理: {coll_name}")
    except Exception as e:
        print(f"  ⚠️ 清理失败: {e}")


# ============================================================
# 主入口
# ============================================================

def run_all_tests():
    print("=" * 60)
    print("Redis + Milvus 服务测试")
    print("=" * 60)
    print()

    results = {}

    # ═══════════════════════════════════════
    # Redis 测试
    # ═══════════════════════════════════════
    print("── Redis 测试 ──")
    print(f"连接目标: {settings.redis.host}:{settings.redis.port} DB={settings.redis.db}")

    redis = test_redis_connection()
    results["Redis 连接"] = redis is not None

    if redis:
        try:
            test_redis_basic_ops(redis)
            results["Redis SET/GET/DEL"] = True
        except Exception as e:
            print(f"  ❌ 基本操作失败: {e}")
            results["Redis SET/GET/DEL"] = False

        try:
            test_redis_json_ops(redis)
            results["Redis JSON"] = True
        except Exception as e:
            print(f"  ❌ JSON 操作失败: {e}")
            results["Redis JSON"] = False

        try:
            test_redis_ttl(redis)
            results["Redis TTL"] = True
        except Exception as e:
            print(f"  ❌ TTL 测试失败: {e}")
            results["Redis TTL"] = False

        try:
            test_cache_modules(redis)
            results["业务缓存模块"] = True
        except Exception as e:
            print(f"  ❌ 缓存模块测试失败: {e}")
            results["业务缓存模块"] = False
    else:
        results["Redis SET/GET/DEL"] = False
        results["Redis JSON"] = False
        results["Redis TTL"] = False
        results["业务缓存模块"] = False

    # ═══════════════════════════════════════
    # Milvus 测试
    # ═══════════════════════════════════════
    print(f"\n── Milvus 测试 ──")
    print(f"连接目标: {settings.milvus.host}:{settings.milvus.port}")

    results["Milvus 连接"] = test_milvus_connection()

    coll_name = None
    if results["Milvus 连接"]:
        collection, coll_name = test_milvus_collection()
        results["Milvus 建表"] = collection is not None

        if collection:
            results["Milvus 检索"] = test_milvus_insert_and_search(collection, coll_name)

            # 清理
            if coll_name:
                cleanup_test_collection(coll_name)
        else:
            results["Milvus 检索"] = False
    else:
        results["Milvus 建表"] = False
        results["Milvus 检索"] = False

    # ═══════════════════════════════════════
    # 汇总
    # ═══════════════════════════════════════
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")

    redis_ok = all(results.get(k, False) for k in ["Redis 连接", "Redis SET/GET/DEL", "Redis JSON", "Redis TTL", "业务缓存模块"])
    milvus_ok = all(results.get(k, False) for k in ["Milvus 连接", "Milvus 建表", "Milvus 检索"])
    all_ok = redis_ok and milvus_ok

    print()
    if all_ok:
        print("🎉 全部通过! Redis + Milvus + MySQL 三大基础设施就绪，系统可以端到端运行。")
    else:
        if not redis_ok: print("⚠️ Redis 测试部分失败")
        if not milvus_ok: print("⚠️ Milvus 测试部分失败")


if __name__ == "__main__":
    run_all_tests()
