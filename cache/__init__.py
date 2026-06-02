"""
三级缓存层
  L1: Redis 热缓存 (<1ms)
  L2: Milvus 内置向量缓存
  L3: MySQL 持久层
"""
from cache.redis_client import RedisClient
from cache.faq_cache import FAQCache
from cache.embedding_cache import EmbeddingCache
from cache.product_cache import ProductCache
from cache.query_result_cache import QueryResultCache
from cache.cache_guard import CacheGuard

__all__ = [
    "RedisClient",
    "FAQCache",
    "EmbeddingCache",
    "ProductCache",
    "QueryResultCache",
    "CacheGuard",
]
