# -*- coding: utf-8 -*-
"""
Agent 智能体系统 — 纯多 Agent Orchestrator + 领域 Agent + 记忆管理

核心设计:
  所有请求统一走 Orchestrator 多 Agent 调度中心:
    Orchestrator: 意图路由 → 任务拆解 → 多 Agent 调度 → LLM 聚合 → 合规终审
    领域 Agent: Plan → Exec → Check → Synthesize (独立的 LangGraph 状态图)
    MemoryManager: 短记忆(RedisSaver Checkpoint) + 长记忆(MySQL 用户画像)

记忆架构:
  短记忆: LangGraph RedisSaver checkpointer, thread_id=session_id, TTL 30min
  长记忆: 根据 X-User-Id 查 MySQL policy_cache + claim_records → user_profile
"""

from agent.memory import MemoryManager, get_memory_manager

__all__ = [
    "MemoryManager",
    "get_memory_manager",
]
