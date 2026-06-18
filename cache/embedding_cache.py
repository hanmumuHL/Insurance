"""
Embedding 缓存 — 缓存 query 的向量编码结果
  高频 query 反复编码浪费 GPU，缓存后直接复用
"""
import hashlib
import numpy as np
from cache.redis_client import RedisClient
from base.logger import logger


class EmbeddingCache:
    def __init__(self, redis: RedisClient, ttl: int = 86400):
        self.redis = redis
        self.ttl = ttl  # 默认 24 小时

    def _key(self, text: str) -> str:
        md5 = hashlib.md5(text.strip().encode()).hexdigest()
        return f"emb:{md5}"

    def get(self, text: str) -> np.ndarray | None:
        """尝试从缓存获取向量"""
        key = self._key(text)
        cached = self.redis.client.get(key)  # 二进制模式
        if cached:
            logger.debug(f"Embedding 缓存命中: {text[:30]}")
            return np.frombuffer(cached.encode("latin-1"), dtype=np.float32)
        return None

    def set(self, text: str, vector: np.ndarray):
        """写入缓存"""
        key = self._key(text)
        raw = vector.astype(np.float32).tobytes()
        try:
            self.redis.client.setex(key, self.ttl, raw.decode("latin-1"))
        except Exception as e:
            logger.warning(f"Embedding 缓存写入失败: {e}")

    def get_or_encode(self, text: str, encoder) -> np.ndarray:
        """先查缓存，未命中才调用编码器"""
        cached = self.get(text)
        if cached is not None:
            return cached

        # 兼容不同 encoder 的参数名：BGEM3Encoder 用 normalize，sentence-transformers 用 normalize_embeddings
        try:
            vector = encoder.encode(text, normalize=True)
        except TypeError:
            vector = encoder.encode(text, normalize_embeddings=True)
        if isinstance(vector, list):
            vector = np.array(vector)

        self.set(text, vector)
        return vector
