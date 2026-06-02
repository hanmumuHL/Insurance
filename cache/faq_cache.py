"""
FAQ 精确匹配缓存
  - 高频标准问题 → 直接返回，跳过全部 RAG 流程
  - 性价比最高的缓存层：一次命中省 350ms + 几千 token
"""
import hashlib
import re
from cache.redis_client import RedisClient
from base.logger import logger


class FAQCache:
    def __init__(self, redis: RedisClient, mysql_session=None):
        self.redis = redis
        self.mysql = mysql_session
        self.ttl = 3600  # 1 小时

    def _normalize(self, text: str) -> str:
        """标准化 query：去标点、去空格、转小写"""
        text = re.sub(r"[，。！？、；：""''（）\s\n]", "", text)
        return text.lower().strip()

    def _key(self, normalized: str) -> str:
        md5 = hashlib.md5(normalized.encode()).hexdigest()
        return f"faq:{md5}"

    def try_hit(self, query: str) -> str | None:
        """
        尝试 FAQ精确命中
        返回 None = 未命中，需走完整流程
        """
        normalized = self._normalize(query)
        if not normalized:
            return None

        key = self._key(normalized)

        # ── L1: Redis ──
        cached = self.redis.get(key)
        if cached is not None:
            # 热度统计 (sorted set)
            try:
                self.redis.client.zincrby("faq:hot", 1, key)
            except Exception:
                pass
            logger.info(f"FAQ 缓存命中: {query[:50]}")
            return cached

        # ── L2: MySQL FAQ 表 ──
        if self.mysql is not None:
            try:
                row = self.mysql.execute(
                    """SELECT answer FROM faq_questions
                       WHERE normalized_question = %s
                       ORDER BY frequency DESC
                       LIMIT 1""",
                    (normalized,),
                ).fetchone()
                if row:
                    answer = row[0]
                    self.redis.setex(key, self.ttl, answer)
                    logger.info(f"FAQ MySQL 命中，已缓存到 Redis: {query[:50]}")
                    return answer
            except Exception as e:
                logger.warning(f"FAQ MySQL 查询失败: {e}")

        return None

    def add_faq(self, question: str, answer: str):
        """手动添加 FAQ 到缓存"""
        normalized = self._normalize(question)
        key = self._key(normalized)
        self.redis.setex(key, self.ttl, answer)
        logger.info(f"FAQ 已添加: {question[:50]}")
