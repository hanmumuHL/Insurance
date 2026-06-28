# -*- coding: utf-8 -*-
"""
BGE-M3 编码器 — 将文本转换为向量 (Dense + Sparse)

BGE-M3 是 BAAI 发布的多语言 Embedding 模型，特点:
  1. Dense 向量 (1024维): 语义相似度，捕捉同义表述
     例: "肺炎住院" 和 "肺部感染住院" → Dense 向量很近
  2. Sparse 向量 (词汇级别): 关键词精确匹配，类似 BM25
     例: "住院" 这个词在文档中出现 → Sparse 向量有非零权重
  3. 一条文本可以同时输出 Dense 和 Sparse 向量，互不冲突

混合检索原理:
  Dense 检索: 语义匹配，"肺炎" 可以匹配到"肺部感染"
  Sparse 检索: 关键词匹配，"免赔额"精确命中条款
  RRF 融合: 两路结果按排名融合，兼顾语义和精确

为什么本地部署 BGE-M3:
  - 保险条款文档不需要出内网（合规考虑）
  - 本地编码延迟 ~20ms，API 调用的延迟 ~100ms+
  - 高并发场景下本地 GPU 推理成本更低

模型加载:
  首次加载需要下载模型 (~2GB)，后续缓存到本地。
  支持 CPU 推理（慢一些但不需要 GPU）。
"""

import numpy as np
from typing import Optional, Tuple
from pathlib import Path

from base.logger import logger
from config.settings import settings


class BGEM3Encoder:
    """
    BGE-M3 编码器 — Dense + Sparse 双向量输出

    用法:
        encoder = BGEM3Encoder()
        dense, sparse = encoder.encode("肺炎住院能赔吗")

        # Dense: (1024,) float32, 已 L2 归一化
        # Sparse: dict {token_id: weight}, 用于 BM25 式检索
    """

    def __init__(self, model_path: str = None, use_fp16: bool = False):
        """
        Args:
            model_path: 模型路径，默认从 settings 读取
            use_fp16: 是否使用半精度 (减少显存占用，仅 GPU 有效)
        """
        self.model_path = model_path or settings.models.bge_m3
        self.use_fp16 = use_fp16
        self._model = None
        self._loaded = False
        self._is_flag = False  # 是否为 FlagEmbedding 模式（支持 Sparse）

    def load(self):
        """
        加载 BGE-M3 模型到内存

        首次调用时自动下载模型 (~2GB)，后续从缓存加载。
        使用 FlagEmbedding 库（BAAI 官方封装）加载模型。

        加载策略:
          GPU 可用 → 加载到 GPU (推理 ~5ms)
          GPU 不可用 → 加载到 CPU (推理 ~50ms)
        """
        if self._loaded:
            return

        with _load_lock:
            # 双重检查：可能另一个线程已完成加载
            if self._loaded:
                return

            logger.info(f"加载 BGE-M3 模型: {self.model_path} ...")

            try:
                # 方法 1: FlagEmbedding (BAAI 官方，推荐)
                # Dense 编码: BGEM3FlagModel
                # Sparse 编码: 自动输出词汇权重
                from FlagEmbedding import BGEM3FlagModel

                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"

                self._model = BGEM3FlagModel(
                    self.model_path,
                    use_fp16=self.use_fp16 and device == "cuda",
                    device=device,
                )

                self._loaded = True
                self._is_flag = True  # 标记为 FlagEmbedding 模式（支持 Sparse）
                logger.info(f"BGE-M3 加载完成 (device={device})")

            except ImportError:
                # 方法 2: sentence-transformers (备选)
                # 只输出 Dense 向量，不支持 Sparse
                logger.warning("FlagEmbedding 未安装，降级为 sentence-transformers")
                self._load_fallback()

    def _load_fallback(self):
        """
        降级方案: 使用 sentence-transformers 加载

        注意: 这种方式只支持 Dense 向量，缺少 Sparse 向量
        混合检索退化为纯 Dense 检索。
        """
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_path)
            self._loaded = True
            self._is_flag = False  # 标记为降级模式
            logger.info("BGE-M3 sentence-transformers 加载完成 (无 Sparse 支持)")
        except ImportError:
            raise ImportError(
                "请安装 FlagEmbedding 或 sentence-transformers:\n"
                "  pip install FlagEmbedding>=1.2.10\n"
                "  或 pip install sentence-transformers>=2.7.0"
            )

    # ============================================================
    # 编码接口
    # ============================================================

    def encode(self, text: str, normalize: bool = True) -> Tuple[np.ndarray, dict]:
        """
        将单条文本编码为 Dense + Sparse 向量

        Args:
            text: 输入文本
            normalize: 是否 L2 归一化 Dense 向量 (Milvus 余弦相似度需要)

        Returns:
            dense: (1024,) float32 numpy array
            sparse: dict {token_id: weight} 或空 dict (降级模式)

        使用示例:
            encoder = BGEM3Encoder()
            encoder.load()
            dense, sparse = encoder.encode("肺炎住院能赔吗")
            # dense 用于 Milvus Dense 检索
            # sparse 用于 Milvus Sparse 检索
        """
        if not self._loaded:
            self.load()

        if self._is_flagembedding():
            # ── FlagEmbedding 模式: 输出 Dense + Sparse ──
            output = self._model.encode(
                [text],                          # 输入是列表
                return_dense=True,               # 输出 Dense 向量
                return_sparse=True,              # 输出 Sparse 向量
                return_colbert_vecs=False,       # ColBERT 向量不需要
            )

            dense = output["dense_vecs"][0]      # (1024,) array
            sparse = output["lexical_weights"][0]  # dict {token_id: weight}

            if normalize:
                # L2 归一化: 使向量模为 1，适合余弦相似度
                norm = np.linalg.norm(dense)
                if norm > 0:
                    dense = dense / norm

            return dense.astype(np.float32), sparse

        else:
            # ── sentence-transformers 降级: 只输出 Dense ──
            dense = self._model.encode(
                text,
                normalize_embeddings=normalize,
            )
            return dense.astype(np.float32), {}

    def encode_batch(
        self,
        texts: list[str],
        normalize: bool = True,
        batch_size: int = 32,
    ) -> Tuple[np.ndarray, list[dict]]:
        """
        批量编码 — 适用于文档摄取时批量处理 chunks

        批量处理的效率远高于逐条编码 (GPU 利用率更高)。

        Args:
            texts: 文本列表
            normalize: 是否归一化
            batch_size: 批量大小 (GPU 可用时建议 32-64)

        Returns:
            dense_vecs: (N, 1024) float32 array
            sparse_dicts: N 个 dict 的列表 [{token_id: weight}, ...]
        """
        if not self._loaded:
            self.load()

        logger.info(f"BGE-M3 批量编码: {len(texts)} 条文本, batch_size={batch_size}")

        if self._is_flagembedding():
            output = self._model.encode(
                texts,
                batch_size=batch_size,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
                show_progress_bar=(len(texts) > 100),  # 数据量大时显示进度条
            )

            dense_vecs = output["dense_vecs"]
            sparse_dicts = output["lexical_weights"]

            if normalize:
                norms = np.linalg.norm(dense_vecs, axis=1, keepdims=True)
                norms[norms == 0] = 1.0  # 防止除零
                dense_vecs = dense_vecs / norms

            logger.info(f"BGE-M3 批量编码完成: {dense_vecs.shape}")
            return dense_vecs.astype(np.float32), sparse_dicts

        else:
            # sentence-transformers 降级
            dense_vecs = self._model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=(len(texts) > 100),
            )
            sparse_dicts = [{} for _ in texts]
            return dense_vecs.astype(np.float32), sparse_dicts

    # ============================================================
    # 维度信息
    # ============================================================

    @property
    def dim(self) -> int:
        """
        Dense 向量维度

        BGE-M3 固定输出 1024 维，这是模型设计决定的，不可更改。
        Milvus Collection 创建时需要这个值。
        """
        return 1024

    # ============================================================
    # 辅助方法
    # ============================================================

    def _is_flagembedding(self) -> bool:
        """判断是否使用 FlagEmbedding (支持 Sparse 向量)"""
        return self._is_flag

    def __repr__(self):
        device = getattr(self._model, 'device', 'unknown') if self._model else 'none'
        return (
            f"BGEM3Encoder(model={self.model_path}, "
            f"device={device}, "
            f"sparse={'no' if not self._is_flag else 'yes'})"
        )


# ============================================================
# 单例 — 整个应用共享一个编码器
# ============================================================

import threading as _threading

_encoder: Optional[BGEM3Encoder] = None
_load_lock = _threading.Lock()
_encoder_lock = _threading.Lock()


def get_encoder() -> BGEM3Encoder:
    """
    获取 BGE-M3 编码器单例

    首次调用时加载模型 (~2GB)，后续复用。
    避免每次请求都加载模型。
    """
    global _encoder
    if _encoder is None:
        with _encoder_lock:
            if _encoder is None:
                _encoder = BGEM3Encoder()
                _encoder.load()
    return _encoder
