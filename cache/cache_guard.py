"""
缓存防护 — 穿透 / 击穿 / 雪崩
"""
import json
import time
import random
from cache.redis_client import RedisClient
from base.logger import logger


class CacheGuard:
    def __init__(self, redis: RedisClient):
        self.redis = redis

    # ── 防穿透：空值也缓存 ──
    def get_with_null_cache(self, key: str, db_func, null_ttl: int = 60, data_ttl: int = 3600):
        """
        db_func: 无参函数，返回数据或 None
        """
        cached = self.redis.get(key)
        if cached is not None:
            return None if cached == "__NULL__" else json.loads(cached)

        result = db_func()
        if result is None:
            self.redis.setex(key, null_ttl, "__NULL__")
        else:
            self.redis.set_json(key, result, ttl=data_ttl)
        return result

    # ── 防击穿：互斥锁 ──
    def get_with_lock(self, key: str, db_func, lock_ttl: int = 10, data_ttl: int = 3600):
        """
        同一时刻只有一个请求去查 DB，其他请求等待后重试读缓存。

        等待策略: 总等待时间 = lock_ttl * 0.8，分 10 次等间隔轮询。
        若 lock_ttl=10s，每 800ms 检查一次，最多等 8 秒。
        """
        cached = self.redis.get_json(key)
        if cached is not None:
            return cached

        lock_key = f"lock:{key}"
        acquired = self.redis.set_nx(lock_key, "1", ttl=lock_ttl)

        if acquired:
            try:
                result = db_func()
                self.redis.set_json(key, result, ttl=data_ttl)
                return result
            finally:
                self.redis.delete(lock_key)
        else:
            # 基于锁 TTL 的动态等待（而非固定 300ms）
            total_wait = lock_ttl * 0.8
            interval = total_wait / 10.0
            for attempt in range(10):
                time.sleep(interval)
                cached = self.redis.get_json(key)
                if cached is not None:
                    return cached
            return None

    # ── 防雪崩：随机 TTL ──
    def set_random_ttl(self, key: str, value, base_ttl: int = 3600):
        """基础 TTL ±20% 随机抖动"""
        ttl = int(base_ttl * (1 + random.uniform(-0.2, 0.2)))
        self.redis.set_json(key, value, ttl=ttl)
