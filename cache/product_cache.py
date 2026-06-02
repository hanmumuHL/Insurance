"""
产品/保司信息缓存 — Cache-Aside 模式
  读多写少，适合 Redis 缓存
"""
import json
import random
from cache.redis_client import RedisClient
from base.logger import logger


class ProductCache:
    def __init__(self, redis: RedisClient, mysql_session=None, ttl: int = 3600):
        self.redis = redis
        self.mysql = mysql_session
        self.base_ttl = ttl

    def _random_ttl(self) -> int:
        """随机 TTL ±20%，防雪崩"""
        return int(self.base_ttl * (1 + random.uniform(-0.2, 0.2)))

    def get_product(self, product_code: str) -> dict | None:
        """查询产品信息，优先 Redis"""
        key = f"product:{product_code}"

        # ── 查 Redis ──
        cached = self.redis.get_json(key)
        if cached is not None:
            if cached == "null":
                return None  # 防穿透的空值缓存
            return cached

        # ── 查 MySQL ──
        if self.mysql is None:
            return None

        try:
            row = self.mysql.execute(
                """SELECT product_code, product_name, insurer,
                          category, is_active, updated_at
                   FROM products
                   WHERE product_code = %s""",
                (product_code,),
            )
            result = row.fetchone()
        except Exception as e:
            logger.warning(f"ProductCache MySQL 查询失败: {e}")
            return None

        if not result:
            # 防穿透：空值也缓存，短 TTL
            self.redis.set_json(key, "null", ttl=60)
            return None

        data = dict(zip(
            ["product_code", "product_name", "insurer",
             "category", "is_active", "updated_at"],
            result,
        ))
        self.redis.set_json(key, data, ttl=self._random_ttl())
        return data

    def invalidate(self, product_code: str):
        """产品信息变更时删除缓存"""
        self.redis.delete(f"product:{product_code}")

    def get_insurer_products(self, insurer: str) -> list[dict]:
        """查某保司下所有产品"""
        key = f"insurer_products:{insurer}"

        cached = self.redis.get_json(key)
        if cached is not None:
            return cached

        if self.mysql is None:
            return []

        try:
            rows = self.mysql.execute(
                """SELECT product_code, product_name, category
                   FROM products
                   WHERE insurer = %s AND is_active = 1
                   ORDER BY product_name""",
                (insurer,),
            )
            products = [
                dict(zip(["product_code", "product_name", "category"], r))
                for r in rows.fetchall()
            ]
        except Exception as e:
            logger.warning(f"ProductCache 查询保司产品失败: {e}")
            return []

        self.redis.set_json(key, products, ttl=self._random_ttl())
        return products
