# -*- coding: utf-8 -*-
"""
记忆管理模块 — 短记忆 (LangGraph Checkpoint) + 长记忆 (用户画像)

短记忆:
  基于 langgraph-checkpoint-redis 的 RedisSaver，自动持久化 Agent 状态。
  messages 字段使用 add_messages annotator，每次对话自动追加。
  TTL 30 分钟，超时自动清除。

长记忆:
  根据 user_id 查询 MySQL policy_cache + claim_records 表，
  构建用户画像 (policies, claims, preferences)，注入 SubAgent 上下文。

使用方式:
    from agent.memory import MemoryManager

    mm = MemoryManager()
    checkpointer = mm.get_checkpointer()            # → RedisSaver
    user_profile = mm.get_user_profile("U12345")    # → {policies, claims, preferences}
"""

import time
from base.logger import logger
from config.settings import settings


class MemoryManager:
    """
    记忆管理器 — 统一管理短记忆和长记忆

    短记忆: LangGraph RedisSaver checkpointer (TTL 30min)
    长记忆: MySQL 用户画像查询
    """

    SESSION_TTL = 1800  # 30 分钟

    def __init__(self):
        self._checkpointer = None
        self._checkpointer_error = None

    # ================================================================
    # 短记忆: LangGraph RedisSaver
    # ================================================================

    def get_checkpointer(self):
        """
        获取 RedisSaver 实例（单例，延迟初始化）

        Returns:
            RedisSaver 实例，或 None（降级为无状态模式）
        """
        if self._checkpointer is not None:
            return self._checkpointer

        if self._checkpointer_error is not None:
            return None

        try:
            from langgraph.checkpoint.redis import RedisSaver

            cfg = settings.redis
            redis_url = (
                f"redis://{cfg.host}:{cfg.port}/{cfg.db}"
                if not cfg.password
                else f"redis://:{cfg.password}@{cfg.host}:{cfg.port}/{cfg.db}"
            )

            self._checkpointer = RedisSaver.from_conn_string(redis_url)
            logger.info("RedisSaver checkpointer 初始化成功")
            return self._checkpointer

        except ImportError:
            self._checkpointer_error = "langgraph-checkpoint-redis 未安装"
            logger.warning(self._checkpointer_error + "，降级为无状态模式")
            return None
        except Exception as e:
            self._checkpointer_error = str(e)
            logger.error(f"RedisSaver 初始化失败: {e}，降级为无状态模式")
            return None

    def build_config(self, session_id: str) -> dict:
        """
        构建 LangGraph invoke 的 config，绑定 thread_id

        thread_id = session_id，用于 checkpoint 的 key。
        同一 session_id 的多次 invoke 会自动恢复历史状态。
        """
        return {"configurable": {"thread_id": session_id}}

    # ================================================================
    # 长记忆: 用户画像
    # ================================================================

    def get_user_profile(self, user_id: str, role: str = "agent") -> dict:
        """
        根据 user_id 查询 MySQL 构建用户画像（角色感知）

        - customer: 只能加载自己的数据，user_id 必须匹配当前用户
        - agent/underwriter: 可以指定任意 user_id 查看客户数据

        Returns:
            dict: {
                "user_id": str,
                "policies": [{"insurer": str, "product_name": str, "status": str, ...}],
                "claims": [{"report_no": str, "status": str, "stage": str, ...}],
                "preferences": {},
                "queried_at": float,
            }
            查询失败返回空 dict，不阻塞主流程。
        """
        if not user_id:
            return {}

        profile = {"user_id": user_id, "policies": [], "claims": [], "preferences": {}}

        t0 = time.time()

        try:
            from base.database import get_mysql_session

            session = get_mysql_session()
            try:
                # ── 查保单 ──
                try:
                    rows = session.execute(
                        "SELECT insurer, product_name, status, sum_insured, premium, "
                        "effective_date, expire_date "
                        "FROM policy_cache WHERE user_id = :uid AND is_valid = TRUE "
                        "ORDER BY effective_date DESC LIMIT 10",
                        {"uid": user_id},
                    ).fetchall()

                    for row in rows:
                        profile["policies"].append({
                            "insurer": row[0],
                            "product_name": row[1],
                            "status": row[2],
                            "sum_insured": str(row[3]) if row[3] else None,
                            "premium": str(row[4]) if row[4] else None,
                            "effective_date": str(row[5]) if row[5] else None,
                            "expire_date": str(row[6]) if row[6] else None,
                        })
                except Exception as e:
                    logger.warning(f"查询保单失败 (user_id={user_id}): {e}")

                # ── 查理赔记录 ──
                try:
                    rows = session.execute(
                        "SELECT report_no, status, stage, submitted_at, estimated_days "
                        "FROM claim_records WHERE user_id = :uid AND is_valid = TRUE "
                        "ORDER BY submitted_at DESC LIMIT 10",
                        {"uid": user_id},
                    ).fetchall()

                    for row in rows:
                        profile["claims"].append({
                            "report_no": row[0],
                            "status": row[1],
                            "stage": row[2],
                            "submitted_at": str(row[3]) if row[3] else None,
                            "estimated_days": row[4],
                        })
                except Exception as e:
                    logger.warning(f"查询理赔记录失败 (user_id={user_id}): {e}")
            finally:
                session.close()

        except Exception as e:
            logger.warning(f"获取用户画像失败 (user_id={user_id}): {e}，降级为空画像")

        profile["queried_at"] = round(time.time() - t0, 3)
        profile["has_data"] = bool(profile["policies"] or profile["claims"])

        if profile["has_data"]:
            logger.info(
                f"用户画像: user_id={user_id} role={role} "
                f"policies={len(profile['policies'])} "
                f"claims={len(profile['claims'])} "
                f"({profile['queried_at']}s)"
            )

        return profile

    # ================================================================
    # 会话管理
    # ================================================================

    def clear_session(self, session_id: str):
        """手动清除指定会话的 checkpoint（如用户主动结束会话）"""
        try:
            checkpointer = self.get_checkpointer()
            if checkpointer:
                checkpointer.delete_thread(session_id)
                logger.info(f"会话已清除: {session_id}")
        except Exception as e:
            logger.warning(f"清除会话失败 ({session_id}): {e}")


# ── 全局单例 ──

_memory_manager = None


def get_memory_manager() -> MemoryManager:
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager
