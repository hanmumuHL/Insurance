# -*- coding: utf-8 -*-
"""
智能管道 (Smart Pipeline) — Stage 抽象与编排

按需增强的管道架构:
  QueryAnalyzer 分析复杂度
  ├─ complexity=0: FAQ/闲聊 → 直接返回（<10ms）
  ├─ complexity=1: 简单咨询 → Classify → Cache → Retrieve → Generate → Compliance
  └─ complexity=2: 复杂业务 → Classify → Cache → KG → Retrieve → Generate → Compliance

核心组件:
  - PipelineContext: 管道状态容器，Stage 间共享数据
  - StageResult: Stage 执行结果
  - Stage (ABC): 可组合的处理阶段
  - SmartPipeline: 主编排器，按复杂度路由
  - QueryAnalyzer: 复杂度判定
  - RetrievalInterface: 统一检索（消除 Agent Tools 重复）
  - KGService: 统一 KG 访问（消除 6 处散落调用）
"""

from rag_qa.core.pipeline.context import PipelineContext, StageResult
from rag_qa.core.pipeline.stage import Stage
from rag_qa.core.pipeline.retrieval_interface import RetrievalInterface
from rag_qa.core.kg.service import KGService  # 已迁移到 kg/ 子包
from rag_qa.core.pipeline.query_analyzer import QueryAnalyzer, AnalyzerResult
from rag_qa.core.pipeline.smart_pipeline import SmartPipeline

__all__ = [
    "PipelineContext",
    "StageResult",
    "Stage",
    "RetrievalInterface",
    "KGService",
    "QueryAnalyzer",
    "AnalyzerResult",
    "SmartPipeline",
]
