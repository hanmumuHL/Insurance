# -*- coding: utf-8 -*-
"""
管道 Stage 集合 — 每个 Stage 是可独立测试的处理单元
"""

from rag_qa.core.pipeline.stages.faq_stage import FAQStage
from rag_qa.core.pipeline.stages.domain_guard_stage import DomainGuardStage
from rag_qa.core.pipeline.stages.classify_stage import ClassifyStage
from rag_qa.core.pipeline.stages.cache_stage import CacheStage
from rag_qa.core.pipeline.stages.chitchat_stage import ChitchatStage
from rag_qa.core.pipeline.stages.strategy_stage import StrategyStage
from rag_qa.core.pipeline.stages.kg_stage import KGStage
from rag_qa.core.pipeline.stages.retrieval_stage import RetrievalStage
from rag_qa.core.pipeline.stages.quality_stage import QualityStage
from rag_qa.core.pipeline.stages.generate_stage import GenerateStage
from rag_qa.core.pipeline.stages.compliance_stage import ComplianceStage
from rag_qa.core.pipeline.stages.cache_write_stage import CacheWriteStage

__all__ = [
    "FAQStage",
    "DomainGuardStage",
    "ClassifyStage",
    "CacheStage",
    "ChitchatStage",
    "StrategyStage",
    "KGStage",
    "RetrievalStage",
    "QualityStage",
    "GenerateStage",
    "ComplianceStage",
    "CacheWriteStage",
]
