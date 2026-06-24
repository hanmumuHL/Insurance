# -*- coding: utf-8 -*-
"""
检索策略选择器 — 根据意图 + query 特征选择最优检索策略

设计背景:
  不同意图需要不同的检索方式。比如"产品对比"不能直接把 query 编码成
  一个向量去搜（会得到两家条款混杂的结果），需要拆成两路并行检索。
  "保费试算"根本不需要语义检索，走结构化条件过滤就行。

  所以需要一个"策略选择器"，在检索之前决定用哪种检索方式。

6 种策略:
  1. direct     — 直接检索: query → 编码 → Milvus (最常见)
  2. hyde       — HyDE 增强: LLM 先生成假设文档 → 编码假设文档 → Milvus
  3. sub_query  — 子查询拆分: LLM 拆成 N 个子 query → N 路并行 → RRF 融合
  4. compare    — 对比检索: 拆多路并行 → 各自 Rerank → LLM 交叉对比
  5. conditional — 条件检索: 结构化条件 → Milvus 标量过滤 (不走语义)
  6. fallback   — 回溯检索: Top-K 不足 → 放宽过滤 → 二次检索 → 融合去重
"""

import re
from dataclasses import dataclass, field
from enum import Enum

from base.logger import logger


# ============================================================
# 策略类型枚举
# ============================================================

class StrategyType(Enum):
    """6 种检索策略"""
    DIRECT = "direct"             # 直接检索
    HYDE = "hyde"                 # HyDE 假设文档增强
    SUB_QUERY = "sub_query"       # 子查询拆分
    COMPARE = "compare"           # 对比检索
    CONDITIONAL = "conditional"   # 条件检索（结构化）
    FALLBACK = "fallback"         # 回溯检索（召回不足时放宽）


# ============================================================
# 策略计划 — 告诉下游执行器该怎么检索
# ============================================================

@dataclass
class StrategyPlan:
    """
    检索策略计划

    这个对象由 StrategySelector 生成，传递给 Milvus 检索器。
    检索器根据 plan 中的字段决定执行什么操作。

    Attributes:
        strategy: 策略类型枚举
        sub_queries: 子查询列表（拆分 / 对比时使用）
        filters: Milvus 标量过滤条件 dict
        use_hyde: 是否启用 HyDE（先让 LLM 生成假设文档再编码）
        fallback_enabled: 召回不足时是否启用回溯（放宽过滤重试）
        reason: 选择该策略的原因（日志 + 调试用）
    """
    strategy: StrategyType
    sub_queries: list[str] = field(default_factory=list)
    filters: dict = field(default_factory=dict)
    use_hyde: bool = False
    fallback_enabled: bool = True
    reason: str = ""


# ============================================================
# 策略选择器主体
# ============================================================

class StrategySelector:
    """
    检索策略选择器

    选择逻辑 (优先级从高到低):
      1. 意图 = 产品对比  → 对比检索 (compare)
      2. 意图 = 保费试算  → 条件检索 (conditional)
      3. query 含多个产品名 → 子查询拆分 (sub_query)
      4. query 过于模糊    → HyDE 增强 (hyde)
      5. 以上都不满足      → 直接检索 (direct)
    """

    def select(self, intent: str, query: str, entities: dict) -> StrategyPlan:
        """
        根据意图和 query 特征选择检索策略

        Args:
            intent: 意图分类结果，如 "条款解读"、"产品对比"
            query: 用户原始 query（已脱敏）
            entities: 从 query 中提取的实体，如 {"insurer": "平安", "product_name": "e生保"}

        Returns:
            StrategyPlan: 检索策略计划，包含策略类型、过滤条件、子查询等
        """

        # ── 优先级 1: 意图驱动 ──
        # 产品对比必须走对比检索，不能直接编码（会导致两家条款混杂）
        if intent == "产品对比":
            return self._plan_compare(query, entities)

        # 保费试算是结构化查询，不需要语义检索
        if intent == "保费试算":
            return self._plan_conditional(query, entities)

        # ── 优先级 2: query 特征驱动 ──
        # 检查 query 中提到了几个产品，>=2 个说明需要拆分
        product_count = self._count_product_mentions(query)
        if product_count >= 2:
            return self._plan_sub_query(query, entities)

        # 检查 query 是否过于模糊（字少 + 无实体）
        if self._is_vague_query(query, entities):
            return self._plan_hyde(query, entities)

        # ── 优先级 3: 默认直接检索 ──
        return self._plan_direct(query, entities)

    # ============================================================
    # 各策略的计划生成方法
    # ============================================================

    def _plan_direct(self, query: str, entities: dict) -> StrategyPlan:
        """
        直接检索计划

        最简单的策略: query 编码成向量 → Milvus ANN 检索
        适用于: 意图明确、query 清晰的标准查询

        流程:
          query "平安e生保免赔额多少"
          → BGE-M3 编码为 1024 维向量
          → Milvus 混合检索 (Dense + Sparse + filter)
          → Top-30 子块
        """
        filters = self._build_filters(entities)
        logger.info(f"策略: 直接检索 | filters={filters}")
        return StrategyPlan(
            strategy=StrategyType.DIRECT,
            filters=filters,
            reason="意图明确，query 清晰，直接检索",
        )

    def _plan_hyde(self, query: str, entities: dict) -> StrategyPlan:
        """
        HyDE (Hypothetical Document Embedding) 检索计划

        为什么需要 HyDE:
          当 query 很短很模糊时（如"保险怎么买"），
          query 向量和条款文档向量在语义空间里距离较远，
          因为 query 是口语化的短句，文档是书面化的长段落。

        HyDE 原理:
          先让 LLM 根据 query 生成一个"假设的理想回答"，
          这个假设回答在语义空间上和真实条款文档很接近，
          然后用假设回答的向量去检索，效果远好于直接用 query 向量。

        示例:
          query: "保险怎么买"
          HyDE 生成: "购买保险时需考虑保障范围、保费、免赔额、
                      等待期等因素，建议先明确自身需求再对比产品..."
          → 编码这段话 → Milvus 检索 → 效果好得多

        代价: 多一次 LLM API 调用 (~200ms)
        """
        filters = self._build_filters(entities)
        logger.info(f"策略: HyDE 检索 | query 模糊，需假设文档增强")
        return StrategyPlan(
            strategy=StrategyType.HYDE,
            filters=filters,
            use_hyde=True,
            reason="query 模糊（<8字且无实体），使用 HyDE 增强",
        )

    def _plan_sub_query(self, query: str, entities: dict) -> StrategyPlan:
        """
        子查询拆分计划

        为什么需要拆分:
          复杂 query 编码成一个向量会丢失细节。
          例如: "平安e生保的等待期和众安尊享e生的免赔额分别是多少？"
          编码后向量 = "等待期" + "免赔额" + "平安" + "众安" 的混合，
          Milvus 返回的结果可能两个产品的条款混在一起。

        拆分后:
          子查询1: "平安e生保 等待期"  → 带 filter insurer=平安
          子查询2: "众安尊享e生 免赔额" → 带 filter insurer=众安
          → 各自检索 → RRF (Reciprocal Rank Fusion) 融合结果
        """
        filters = self._build_filters(entities)

        # 从 entities 中提取产品名和维度，生成子查询
        sub_queries = []
        products = entities.get("products", [])
        dimensions = entities.get("dimensions", []) or self._extract_dimensions(query)

        if products and dimensions:
            for product in products:
                for dim in dimensions:
                    sub_queries.append(f"{product} {dim}")
        elif products:
            for product in products:
                sub_queries.append(f"{product} 条款")
        else:
            # 无法拆分时回退为原查询
            sub_queries = [query]

        logger.info(f"策略: 子查询拆分 | 生成 {len(sub_queries)} 个子查询: {sub_queries}")
        return StrategyPlan(
            strategy=StrategyType.SUB_QUERY,
            sub_queries=sub_queries,
            filters=filters,
            reason="query 含多个产品/条件，拆分子查询并行检索",
        )

    def _plan_compare(self, query: str, entities: dict) -> StrategyPlan:
        """
        对比检索计划

        对比检索 vs 子查询拆分的区别:
          子查询拆分: 各子查询检索不同维度的信息（等待期、免赔额）
          对比检索: 各子查询检索相同维度但不同产品，用于交叉对比

        对比检索流程:
          "平安e生保和众安尊享e生哪个好"
          → 拆成 2 路:
              路A: filter=product_name like "e生保"  → 检索 → Rerank → 结果A
              路B: filter=product_name like "尊享e生" → 检索 → Rerank → 结果B
          → LLM 对比 结果A vs 结果B → 生成对比表格
        """
        filters = self._build_filters(entities)

        # 从 entities 中提取产品名生成对比子查询
        sub_queries = []
        products = entities.get("products", [])
        if products:
            for product in products:
                sub_queries.append(f"{product} 保障范围 保费 免赔额")
        else:
            # 无法拆分时回退为原查询
            sub_queries = [query]

        logger.info(f"策略: 对比检索 | 生成 {len(sub_queries)} 路对比子查询")
        return StrategyPlan(
            strategy=StrategyType.COMPARE,
            sub_queries=sub_queries,
            filters=filters,
            reason="产品对比意图，拆分多路并行检索后交叉对比",
        )

    def _plan_conditional(self, query: str, entities: dict) -> StrategyPlan:
        """
        条件检索计划

        为什么不需要语义检索:
          "30岁买平安e生保要多少钱" 这个问题的答案在费率表里，
          费率表是结构化数据（年龄 → 保费），存在 MySQL 而不是 Milvus。
          直接 SQL 查询: SELECT premium FROM rates
                        WHERE product='平安e生保' AND age=30
          比语义检索精确得多。
        """
        filters = self._build_filters(entities)
        logger.info(f"策略: 条件检索 | 结构化查询")
        return StrategyPlan(
            strategy=StrategyType.CONDITIONAL,
            filters=filters,
            reason="结构化条件查询（保费试算），走标量过滤 + MySQL",
        )

    # ============================================================
    # 辅助方法
    # ============================================================

    def _build_filters(self, entities: dict) -> dict:
        """
        根据提取的实体构造 Milvus 过滤条件

        entities 示例: {"insurer": "平安", "product_name": "e生保"}
        返回: {"insurer": "平安", "product_name": "e生保"}

        这些 filter 会在 Milvus 检索时拼成表达式:
          filter='insurer=="平安" and product_name like "e生保%"'
        """
        filters = {}
        if entities.get("insurer"):
            filters["insurer"] = entities["insurer"]
        if entities.get("product_name"):
            filters["product_name"] = entities["product_name"]
        return filters

    def _count_product_mentions(self, query: str) -> int:
        """
        统计 query 中提到了几个产品

        优先使用 KG 实体链接器（更精准），回退到正则匹配。
        >=2 个说明用户在做对比或同时问多个产品，需要拆分
        """
        # ── 尝试 KG 实体链接 ──
        try:
            from rag_qa.core.kg_entity_linker import KGEntityLinker
            linker = KGEntityLinker()
            entities = linker.link(query)
            products = entities.get("products", [])
            if products:
                return len(products)
        except Exception:
            pass

        # ── 回退: 正则匹配 ──
        patterns = [
            r"[\u4e00-\u9fa5]{2,6}(?:保|e生|尊享|守护|健康)",
        ]
        products = set()
        for pat in patterns:
            matches = re.findall(pat, query)
            products.update(matches)
        return len(products)

