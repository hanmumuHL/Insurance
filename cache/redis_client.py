"""
Redis 客户端封装 — 底层连接 + 基本读写
"""
import json
import redis
from base.logger import logger
from config.settings import settings


class RedisClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        cfg = settings.redis
        try:
            self.client = redis.StrictRedis(
                host=cfg.host,
                port=cfg.port,
                password=cfg.password or None,
                db=cfg.db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            self.client.ping()
            logger.info(f"Redis 连接成功 {cfg.host}:{cfg.port}")
        except redis.RedisError as e:
            logger.error(f"Redis 连接失败: {e}")
            raise

    # ── 基础操作 ──

    def get(self, key: str):
        try:
            return self.client.get(key)
        except redis.RedisError as e:
            logger.error(f"Redis GET 失败 {key}: {e}")
            return None

    def set(self, key: str, value: str, ttl: int = 0):
        try:
            if ttl > 0:
                self.client.setex(key, ttl, value)
            else:
                self.client.set(key, value)
        except redis.RedisError as e:
            logger.error(f"Redis SET 失败 {key}: {e}")

    def setex(self, key: str, ttl: int, value: str):
        self.set(key, value, ttl)

    def delete(self, *keys):
        try:
            self.client.delete(*keys)
        except redis.RedisError as e:
            logger.error(f"Redis DELETE 失败: {e}")

    def exists(self, key: str) -> bool:
        try:
            return bool(self.client.exists(key))
        except redis.RedisError:
            return False

    def get_json(self, key: str):
        data = self.get(key)
        if data is not None:
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return None
        return None

    def set_json(self, key: str, value, ttl: int = 0):
        self.set(key, json.dumps(value, ensure_ascii=False, default=str), ttl)


def get_redis_client():
    return RedisClient()
