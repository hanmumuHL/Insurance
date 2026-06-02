"""
Query 结果缓存 — 相同问题 + 相同意图 → 秒回
  适合高峰期重复咨询（新产品上线后大量用户问相同问题）
"""
import hashlib
import re
import json
from cache.redis_client import RedisClient
from base.logger import logger


class QueryResultCache:
    def __init__(self, redis: RedisClient, ttl: int = 600):
        self.redis = redis
        self.ttl = ttl  # 默认 10 分钟

    def _normalize(self, text: str) -> str:
        text = re.sub(r"[，。！？、；：""''（）\s\n]", "", text)
        return text.lower().strip()

    def _key(self, query: str, intent: str) -> str:
        normalized = self._normalize(query)
        md5 = hashlib.md5(normalized.encode()).hexdigest()
        return f"result:{intent}:{md5}"

    def try_get(self, query: str, intent: str) -> dict | None:
        key = self._key(query, intent)
        cached = self.redis.get_json(key)
        if cached is not None:
            logger.info(f"Query 结果缓存命中: {query[:40]} intent={intent}")
            return cached
        return None

    def set(self, query: str, intent: str, result: dict):
        key = self._key(query, intent)
        self.redis.set_json(key, result, ttl=self.ttl)

    def invalidate_intent(self, intent: str):
        """清空某意图的所有缓存（如文档更新后）"""
        pattern = f"result:{intent}:*"
        try:
            cursor = 0
            while True:
                cursor, keys = self.redis.client.scan(cursor, match=pattern, count=100)
                if keys:
                    self.redis.delete(*keys)
                if cursor == 0:
                    break
        except Exception as e:
            logger.warning(f"Query 缓存清理失败: {e}")
