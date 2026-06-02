# -*- coding: utf-8 -*-
"""
检索质量守卫 — 检索结果太差时拒绝回答，防止 LLM 幻觉

设计背景:
  有些 query 虽然通过了领域守卫和意图分类，但实际检索到的条款
  和用户问题相关性很低。比如用户瞎编了一个不存在的产品名，
  或者问题太泛导致检索到的条款都是弱相关。

  如果不检查检索质量就直接丢给 LLM 生成答案，LLM 会"编造"
  看起来合理但实际错误的答案（幻觉）。这在保险场景是不可接受的。

  所以需要在检索之后、生成之前加一道质量检查。

检查维度:
  1. Top-1 相似度 — 最相关的条款是否足够相关
  2. Top-5 平均相似度 — 整体检索质量
  3. 高相似度 chunk 数量 — 是否有足够多的相关条款
  4. chunk 长度 — 条款内容是否完整（太短说明截断了）
"""

import numpy as np
from dataclasses import dataclass
from base.logger import logger


@dataclass
class QualityResult:
    """
    检索质量检查结果

    Attributes:
        passed: 是否通过质量检查
        reason: 未通过时的原因描述
        fallback_response: 未通过时返回给用户的兜底话术
        metrics: 各项指标的实际值（用于日志和监控）
    """
    passed: bool
    reason: str = ""
    fallback_response: str = ""
    metrics: dict = None


class RetrievalQualityGuard:
    """
    检索质量守卫

    在 Milvus 检索 + Reranker 精排之后调用。
    检查检索结果的质量指标，不达标则拒绝生成答案。
    """

    # ── 质量阈值 ──
    # 这些阈值需要根据实际数据调优，初始值基于经验设定

    MIN_TOP1_SCORE = 0.55         # Top-1 的 cosine similarity 最低值
                                  # 低于此值说明最相关的条款也不够相关
    MIN_AVG_TOP5_SCORE = 0.40     # Top-5 的平均 similarity 最低值
                                  # 低于此值说明整体检索质量差
    MIN_GOOD_CHUNK_COUNT = 2      # similarity > 0.5 的 chunk 最少数量
                                  # 少于 2 个说明没有足够多的相关条款来支撑回答
    MIN_CHUNK_LENGTH = 50         # 单个 chunk 最短字符数
                                  # 太短说明条款被截断或解析异常

    # ── 各意图的差异化阈值 ──
    # 涉及金额/赔付的意图要求更高，因为错误答案后果更严重
    INTENT_THRESHOLDS = {
        "条款解读": {"top1": 0.55, "avg5": 0.40},
        "理赔咨询": {"top1": 0.60, "avg5": 0.45},  # 理赔更严格
        "产品对比": {"top1": 0.50, "avg5": 0.35},  # 对比可以稍宽松
        "退保咨询": {"top1": 0.55, "avg5": 0.40},
    }

    # ── 兜底话术 ──
    FALLBACK_NO_RESULTS = "抱歉，未找到相关信息。请确认您的问题是否与保险相关，或联系人工客服获取帮助。"
    FALLBACK_LOW_QUALITY = (
        "抱歉，您的问题与知识库中的保险条款匹配度较低。\n"
        "建议您：\n"
        "1. 换一种方式描述问题\n"
        "2. 提供具体的产品名称\n"
        "3. 联系人工客服获取帮助"
    )
    FALLBACK_INSUFFICIENT = (
        "抱歉，知识库中与您问题高度相关的条款较少。\n"
        "建议您重新描述问题，或转接人工客服获取帮助。"
    )

    def check(self, results: list, intent: str = "") -> QualityResult:
        """
        检查检索结果质量

        Args:
            results: ChunkResult 列表（已经过 Reranker 精排）
            intent: 意图类型（不同意图有不同阈值）

        Returns:
            QualityResult: 检查结果，passed=True 表示可以生成答案
        """

        # ── 检查 0: 结果为空 ──
        if not results:
            logger.warning("检索质量检查: 结果为空")
            return QualityResult(
                passed=False,
                reason="检索结果为空",
                fallback_response=self.FALLBACK_NO_RESULTS,
                metrics={"count": 0},
            )

        # 提取分数列表
        scores = [r.score for r in results]
        top1_score = scores[0]

        # ── 获取当前意图的阈值（有差异化阈值就用差异化的）──
        thresholds = self.INTENT_THRESHOLDS.get(intent, {
            "top1": self.MIN_TOP1_SCORE,
            "avg5": self.MIN_AVG_TOP5_SCORE,
        })
        min_top1 = thresholds["top1"]
        min_avg5 = thresholds["avg5"]

        # 统计指标
        metrics = {
            "count": len(results),
            "top1_score": round(top1_score, 4),
            "avg_top5_score": round(float(np.mean(scores[:5])), 4) if len(scores) >= 5 else round(float(np.mean(scores)), 4),
            "good_chunk_count": sum(1 for s in scores if s > 0.5),
            "intent": intent,
        }

        # ── 检查 1: Top-1 相似度太低 ──
        # 最相关的条款都不够相关 → 后续条款更不相关，直接拒绝
        if top1_score < min_top1:
            logger.warning(
                f"检索质量检查失败: top1={top1_score:.3f} < {min_top1} "
                f"intent={intent}"
            )
            return QualityResult(
                passed=False,
                reason=f"Top-1 相似度不足: {top1_score:.3f} < {min_top1}",
                fallback_response=self.FALLBACK_LOW_QUALITY,
                metrics=metrics,
            )

        # ── 检查 2: Top-5 平均相似度太低 ──
        # 即使 Top-1 还行，但整体质量差也不行（可能是碰巧一个相关）
        avg_top5 = metrics["avg_top5_score"]
        if avg_top5 < min_avg5:
            logger.warning(
                f"检索质量检查失败: avg_top5={avg_top5:.3f} < {min_avg5}"
            )
            return QualityResult(
                passed=False,
                reason=f"Top-5 平均相似度不足: {avg_top5:.3f} < {min_avg5}",
                fallback_response=self.FALLBACK_LOW_QUALITY,
                metrics=metrics,
            )

        # ── 检查 3: 高相似度 chunk 太少 ──
        # 只有 1 个相关条款不足以支撑一个完整回答
        good_count = metrics["good_chunk_count"]
        if good_count < self.MIN_GOOD_CHUNK_COUNT:
            logger.warning(
                f"检索质量检查失败: good_chunks={good_count} < {self.MIN_GOOD_CHUNK_COUNT}"
            )
            return QualityResult(
                passed=False,
                reason=f"高相似度 chunk 不足: {good_count} < {self.MIN_GOOD_CHUNK_COUNT}",
                fallback_response=self.FALLBACK_INSUFFICIENT,
                metrics=metrics,
            )

        # ── 全部通过 ──
        logger.info(
            f"检索质量检查通过: top1={top1_score:.3f}, "
            f"avg5={avg_top5:.3f}, good={good_count}"
        )
        return QualityResult(
            passed=True,
            metrics=metrics,
        )
