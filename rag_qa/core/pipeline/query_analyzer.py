# -*- coding: utf-8 -*-
"""
查询分析器 — 判定查询复杂度和意图，决定走哪条管道路径

智能管道的核心决策点。一次 query 进来，QueryAnalyzer 分析后输出:
  - complexity=0: FAQ/闲聊/投诉 → 直接返回，不走检索
  - complexity=1: 简单咨询 → 轻量管道（分类→检索→生成→合规）
  - complexity=2: 复杂业务 → 完整管道（分类→KG→检索→生成→合规）

复杂度判定规则:
  1. FAQ 精确命中                        → complexity=0
  2. 意图为闲聊/投诉/被拒绝               → complexity=0
  3. 短查询(≤4字)且无保险关键词           → complexity=0
  4. 意图为理赔咨询/产品对比/核保咨询      → complexity=2
  5. entities 含 ≥2 个产品                → complexity=2
  6. 默认                                 → complexity=1

使用方式:
    from rag_qa.core.pipeline.query_analyzer import QueryAnalyzer
    analyzer = QueryAnalyzer()
    result = analyzer.analyze(query, session_id, user_role)
    # result.complexity, result.intent, result.entities, result.faq_answer
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from base.logger import logger


@dataclass
class AnalyzerResult:
    """
    查询分析结果

    Attributes:
        complexity: 复杂度 0/1/2
        intent: 意图类型
        entities: 提取的实体
        faq_answer: FAQ 命中时的答案（非 None 表示应直接返回）
        domain_blocked: 是否被领域守卫拦截
        fallback_response: 被拦截时的兜底话术
    """

    complexity: int = 1
    intent: str = ""
    entities: dict = field(default_factory=dict)
    faq_answer: Optional[str] = None
    domain_blocked: bool = False
    fallback_response: str = ""


class QueryAnalyzer:
    """
    查询分析器 — 判定复杂度和意图

    委托现有组件（FAQCache、DomainBoundaryGuard、QueryClassifier），
    不做重复实现。仅负责"分析"——不产生副作用（不写缓存、不调 LLM 生成）。
    """

    # 需要走完整管道的复杂意图
    COMPLEX_INTENTS = {"理赔咨询", "产品对比", "核保咨询"}

    # 应短路返回的简单意图
    TRIVIAL_INTENTS = {"闲聊寒暄", "投诉建议"}

    # 保险核心词（用于短查询放行判断）
    # 这些词即使 query 很短也说明是保险相关，不应降级
    INSURANCE_CORE_WORDS = {
        "投保", "续保", "退保", "理赔", "核保", "保费", "保额",
        "条款", "保障", "免赔", "等待期", "犹豫期", "赔付",
        "医疗险", "重疾险", "意外险", "寿险", "车险",
        "住院", "门诊", "手术", "确诊",
        "平安", "众安", "太平洋", "人保", "e生保",
    }

    def __init__(self, faq_cache=None, domain_guard=None, classifier=None):
        """
        Args:
            faq_cache: FAQCache 实例（可选，默认创建）
            domain_guard: DomainBoundaryGuard 实例（可选，默认创建）
            classifier: QueryClassifier 实例（可选，默认创建）
        """
        self._faq_cache = faq_cache
        self._domain_guard = domain_guard
        self._classifier = classifier

    # ============================================================
    # 主入口
    # ============================================================

    def analyze(self, query: str, session_id: str = "", user_role: str = "agent") -> AnalyzerResult:
        """
        分析 query，返回复杂度判定和意图信息

        Args:
            query: 用户查询
            session_id: 会话 ID
            user_role: 用户角色

        Returns:
            AnalyzerResult: 复杂度、意图、实体、可能的 FAQ 答案
        """
        # ── Step 1: FAQ 缓存检查 ──
        faq_answer = self._check_faq(query)
        if faq_answer:
            return AnalyzerResult(complexity=0, intent="FAQ", faq_answer=faq_answer)

        # ── Step 2: 领域边界检查 ──
        guard_result = self._check_domain(query, user_role)
        if not guard_result.passed:
            return AnalyzerResult(
                complexity=0,
                intent="out_of_domain",
                domain_blocked=True,
                fallback_response=guard_result.fallback_response,
            )

        # ── Step 3: 意图分类 ──
        intent_result = self._classify(query)
        intent = intent_result.intent
        entities = intent_result.entities

        # ── Step 4: 复杂度判定 ──

        # 4a. 意图被拒绝 → complexity=0
        if intent_result.reject:
            return AnalyzerResult(
                complexity=0,
                intent=intent,
                entities=entities,
                fallback_response=intent_result.fallback_response,
            )

        # 4b. 闲聊/投诉 → complexity=0
        if intent in self.TRIVIAL_INTENTS:
            return AnalyzerResult(complexity=0, intent=intent, entities=entities)

        # 4c. 复杂意图 → complexity=2
        if intent in self.COMPLEX_INTENTS:
            return AnalyzerResult(complexity=2, intent=intent, entities=entities)

        # 4d. 多产品对比 → complexity=2
        products = entities.get("products", [])
        if len(products) >= 2:
            return AnalyzerResult(complexity=2, intent=intent, entities=entities)

        # 4e. 短查询无保险关键词 → complexity=0（降级为 RAG 或拒答）
        if self._is_trivial_short_query(query):
            return AnalyzerResult(complexity=0, intent=intent, entities=entities)

        # 4f. 默认 → complexity=1（简单咨询）
        return AnalyzerResult(complexity=1, intent=intent, entities=entities)

    # ============================================================
    # 内部方法
    # ============================================================

    def _check_faq(self, query: str) -> Optional[str]:
        """检查 FAQ 缓存，命中则返回答案"""
        try:
            if self._faq_cache is None:
                from cache.faq_cache import FAQCache
                from cache.redis_client import RedisClient
                self._faq_cache = FAQCache(RedisClient())
            return self._faq_cache.try_hit(query)
        except Exception as e:
            logger.debug(f"QueryAnalyzer: FAQ 检查跳过 ({e})")
            return None

    def _check_domain(self, query: str, user_role: str):
        """领域边界检查，返回 GuardResult"""
        try:
            if self._domain_guard is None:
                from rag_qa.core.domain_guard import DomainBoundaryGuard
                self._domain_guard = DomainBoundaryGuard()
            return self._domain_guard.check(query, user_role=user_role)
        except Exception as e:
            logger.debug(f"QueryAnalyzer: 领域守卫跳过 ({e})")
            # 降级：放行
            from rag_qa.core.domain_guard import GuardResult
            return GuardResult(passed=True)

    def _classify(self, query: str):
        """意图分类，返回 IntentResult"""
        if self._classifier is None:
            from rag_qa.core.query_classifier import QueryClassifier
            from rag_qa.core.bert_intent_classifier import get_bert_classifier
            from base.llm_client import get_llm_client
            bert_model = get_bert_classifier()
            llm_client = get_llm_client()
            self._classifier = QueryClassifier(
                bert_model=bert_model, llm_client=llm_client
            )
        return self._classifier.classify(query)

    def _is_trivial_short_query(self, query: str) -> bool:
        """
        判断是否为应降级的短查询

        规则: 长度 ≤ 4 字，且不包含任何保险核心词。
        这样 "理赔" "退保" 等短词不会被误降级。
        """
        clean = query.strip()
        if len(clean) > 4:
            return False

        # 包含保险核心词 → 不是 trivial
        for word in self.INSURANCE_CORE_WORDS:
            if word in clean:
                return False

        # 纯标点或单字 → trivial
        if len(clean) <= 1:
            return True

        # 短查询无保险关键词 → trivial
        return True
