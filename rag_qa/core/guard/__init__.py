# -*- coding: utf-8 -*-
"""
安全守卫 (Guard) 子系统

保险是强监管行业，所有 LLM 输出必须经过合规检查。
三个守卫分别在管道的不同阶段介入:

  1. DomainBoundaryGuard     — 领域边界守卫（Pipeline 入口）
      判断 query 是否在保险业务范围内，拦截无关/越界请求。

  2. RetrievalQualityGuard   — 检索质量守卫（Retrieve 之后、Generate 之前）
      检查检索结果质量，低质量结果触发降级策略。

  3. ComplianceGuard         — 合规守卫（Generate 之后，返回用户之前）
      5 规则检查: 医疗建议 | 监管敏感词 | 贬低 | 金额引用 | 角色感知严格度
"""

from rag_qa.core.compliance_guard import ComplianceGuard, ComplianceResult
from rag_qa.core.domain_guard import DomainBoundaryGuard
from rag_qa.core.retrieval_quality_guard import RetrievalQualityGuard

__all__ = [
    "DomainBoundaryGuard",
    "RetrievalQualityGuard",
    "ComplianceGuard",
    "ComplianceResult",
]
