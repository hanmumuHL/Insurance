# -*- coding: utf-8 -*-
"""
bge-reranker-large 精排器 — 对 Milvus 粗排结果进行精细化重排序

为什么需要 Reranker:
  Milvus 的 Dense 检索和 Sparse 检索都是"粗排"，
  它们在百万级向量中快速筛选出最相似的 Top-30~50。
  但粗排的相似度计算比较简单（余弦/内积），精度有限。

  Reranker 是"精排"——拿 query 和每个候选 chunk 做深度交互编码：
    - 把 query 和 chunk 拼在一起喂给 Cross-Encoder
    - 让模型逐 token 计算 query 和 chunk 的匹配度
    - 精度远高于 Dense 向量检索的余弦相似度

  代价: Reranker 计算量大，只能对 Top-30 做精排，不能全量做。

工作流程:
  Milvus 粗排 Top-30 → Reranker 精排 → Top-5 → LLM 生成答案

性能指标:
  - GPU 推理: ~10ms 重排 30 条
  - CPU 推理: ~50ms 重排 30 条
  - 精度提升: Reranker 重排后的 Top-1 准确率比 Milvus 粗排高 10-15%

模型选择:
  bge-reranker-large (BAAI) — 中文最优
  参数量: ~560M，显存占用 ~2GB
"""

import numpy as np
from typing import Optional

from base.logger import logger
from config.settings import settings


class Reranker:
    """
    精排器 — 对候选 chunks 按与 query 的真实相关度重排序

    用法:
        reranker = Reranker()
        reranker.load()

        # 对 Top-30 做精排，取 Top-5
        reranked = reranker.rerank(
            query="肺炎住院能赔吗",
            chunks=top30_chunks,  # ChunkResult 列表
            top_k=5,
        )
    """

    def __init__(self, model_path: str = None, use_fp16: bool = False):
        """
        Args:
            model_path: 模型路径，默认从 settings 读取
            use_fp16: 半精度 (减少显存，仅 GPU)
        """
        self.model_path = model_path or settings.models.reranker
        self.use_fp16 = use_fp16
        self._model = None
        self._loaded = False

    def load(self):
        """
        加载 bge-reranker-large 模型

        首次下载 ~2GB，后续从缓存加载。
        优先使用 FlagEmbedding 官方封装，降级到 HuggingFace。
        """
        if self._loaded:
            return

        logger.info(f"加载 Reranker 模型: {self.model_path} ...")

        try:
            # ── 方法 1: FlagEmbedding 官方 (推荐) ──
            from FlagEmbedding import FlagReranker
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"

            self._model = FlagReranker(
                self.model_path,
                use_fp16=self.use_fp16 and device == "cuda",
                device=device,
            )
            self._loaded = True
            self._is_flag = True

            logger.info(f"Reranker 加载完成 (FlagEmbedding, device={device})")

        except ImportError:
            # ── 方法 2: HuggingFace Cross-Encoder (备选) ──
            logger.warning("FlagEmbedding 未安装，降级为 HuggingFace CrossEncoder")
            self._load_fallback()

    def _load_fallback(self):
        """降级方案: HuggingFace CrossEncoder"""
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_path)
            self._loaded = True
            self._is_flag = False
            logger.info("Reranker 加载完成 (CrossEncoder 降级)")
        except ImportError:
            raise ImportError(
                "请安装 FlagEmbedding 或 sentence-transformers:\n"
                "  pip install FlagEmbedding>=1.2.10\n"
                "  或 pip install sentence-transformers>=2.7.0"
            )

    # ============================================================
    # 精排接口
    # ============================================================

    def rerank(
        self,
        query: str,
        chunks: list,
        top_k: int = 5,
    ) -> list:
        """
        对候选 chunks 精排，返回 Top-K

        输入 chunks 可以是:
          - ChunkResult 对象列表 (有 .text 和 .score 属性)
          - 普通 dict 列表 (有 "text" 和 "score" key)
          - 纯字符串列表

        精排后的 chunk 会更新 score 字段为 Reranker 分数 (0-1)。

        Args:
            query: 用户查询文本
            chunks: 候选 chunk 列表
            top_k: 返回 Top-K

        Returns:
            精排后的 Top-K chunks (原对象被原地修改 score)

        精排原理 (Cross-Encoder):
          不同于向量检索用的 Bi-Encoder（query 和 chunk 分别编码再算余弦），
          Cross-Encoder 把 query 和 chunk 拼在一起:
            "[CLS] 肺炎住院能赔吗 [SEP] 被保险人在保险期间内因疾病住院... [SEP]"
          然后整个序列输入模型，逐 token 计算匹配度。
          这种交互式计算精度远高于向量相似度，但速度慢 50-100 倍。
          所以只对 Top-30 做，不能全量做。
        """
        if not self._loaded:
            self.load()

        if not chunks:
            logger.warning("Reranker 输入为空")
            return []

        # ── 提取文本和分数 ──
        chunk_texts = []
        for c in chunks:
            if hasattr(c, "text"):
                chunk_texts.append(c.text)
            elif isinstance(c, dict):
                chunk_texts.append(c.get("text", ""))
            elif isinstance(c, str):
                chunk_texts.append(c)
            else:
                chunk_texts.append("")

        if not any(chunk_texts):
            return chunks[:top_k]

        # ── 构造 (query, chunk) 对 ──
        # Reranker 需要成对的 query 和 document
        pairs = [[query, text] for text in chunk_texts]

        # ── 计算相关性分数 ──
        logger.info(f"Reranker 精排: {len(pairs)} 个候选 chunks")

        if self._is_flag:
            # FlagEmbedding: compute_score 返回 float 列表
            scores = self._model.compute_score(
                pairs,
                batch_size=len(pairs),     # 一次性的，不用分批
                normalize=True,             # 归一化到 [0, 1]
            )
            # compute_score 可能返回单个 float (单条时) 或 list
            if not isinstance(scores, list):
                scores = [scores]
        else:
            # HuggingFace CrossEncoder: predict 返回 numpy array
            scores = self._model.predict(
                pairs,
                show_progress_bar=False,
            )
            scores = scores.tolist()

        # ── 更新分数并排序 ──
        for i, chunk in enumerate(chunks):
            if i < len(scores):
                if hasattr(chunk, "score"):
                    chunk.score = float(scores[i])
                elif isinstance(chunk, dict):
                    chunk["rerank_score"] = float(scores[i])

        # 按分数降序排列 (分数越高越相关)
        scored = list(zip(scores, chunks))
        scored.sort(key=lambda x: x[0], reverse=True)

        # 取 Top-K
        reranked = [chunk for _, chunk in scored[:top_k]]

        # ── 记录精排前后对比 (用于监控 Reranker 的效果) ──
        if len(reranked) >= 2:
            logger.info(
                f"Reranker 完成: {len(chunks)} → {len(reranked)} "
                f"(Top-1 score: {scores[0]:.4f})"
            )

        return reranked

    def rerank_with_threshold(
        self,
        query: str,
        chunks: list,
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> list:
        """
        精排 + 阈值过滤

        和 rerank 一样，但额外过滤掉 rerank 分数 < min_score 的 chunk。
        如果精排后 Top-1 的分数都低于 min_score，说明所有候选都不够相关，
        返回空列表（触发 RetrievalQualityGuard 拒绝）。

        Args:
            query: 用户查询
            chunks: 候选列表
            top_k: 返回 Top-K
            min_score: 最低相关度阈值 (0-1)
                       0.3 = 较宽松，0.5 = 中等，0.7 = 严格

        Returns:
            过滤后的 Top-K chunks
        """
        reranked = self.rerank(query, chunks, top_k)

        # 阈值过滤
        filtered = []
        for chunk in reranked:
            score = getattr(chunk, "score", None) if hasattr(chunk, "score") else \
                    chunk.get("rerank_score", 0) if isinstance(chunk, dict) else 0
            if score >= min_score:
                filtered.append(chunk)

        if len(filtered) < len(reranked):
            logger.info(
                f"Reranker 阈值过滤: {len(reranked)} → {len(filtered)} "
                f"(min_score={min_score})"
            )

        return filtered


# ============================================================
# 单例
# ============================================================

_reranker: Optional[Reranker] = None


def get_reranker() -> Reranker:
    """获取 Reranker 单例"""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
        _reranker.load()
    return _reranker
