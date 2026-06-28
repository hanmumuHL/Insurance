# -*- coding: utf-8 -*-
"""
文档分块器 — 将解析后的条款文本拆分为适合检索的 chunks

分块策略: Parent-Child (父子块)

  为什么不能简单地按固定长度切:
    1. 切太短 (~256字): 检索精度高，但上下文不完整
       → LLM 只看到一个片段，无法给出完整回答
    2. 切太长 (~2000字): 上下文完整，但检索精度低
       → 向量是整个长文本的"平均语义"，和 query 匹配不精确

  Parent-Child 策略:
    子块 (~512字): 用于检索，精度高
    父块 (~2000字): 4个子块合并，用于返回上下文，信息完整
    检索时先找到匹配的子块，再回查父块获取完整上下文

  重叠 (overlap):
    相邻子块之间保留 64 字的重叠，
    防止关键信息正好被切在两个 chunk 的边界上而丢失
"""

import re
from dataclasses import dataclass, field

from base.logger import logger


# ============================================================
# 分块结果数据结构
# ============================================================

@dataclass
class Chunk:
    """
    单个文本块

    Attributes:
        chunk_id: 唯一标识，格式 {doc_id}_child_0001 或 {doc_id}_parent_0001
        text: 文本内容
        chunk_type: "child" (子块) 或 "parent" (父块)
        parent_id: 父块的 chunk_id (子块才有，父块为 None)
        parent_text: 父块的完整文本 (检索后回查填充)
        chunk_index: 在文档中的顺序位置
        metadata: 附加元数据 (保司、产品名、文档类型、条款章节等)
    """
    chunk_id: str
    text: str
    chunk_type: str = "child"     # "child" 或 "parent"
    parent_id: str = None
    parent_text: str = None
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)

    # ── 属性代理：从 metadata 中读取常用字段 ──
    # KG 的 build_from_clauses() 通过属性访问 product_name 等字段，
    # 但实际数据存储在 metadata dict 中。这些 property 确保两种访问方式兼容。

    @property
    def product_name(self) -> str:
        return self.metadata.get("product_name", "")

    @property
    def product_code(self) -> str:
        return self.metadata.get("product_code", "")

    @property
    def clause_type(self) -> str:
        return self.metadata.get("clause_type", "")


# ============================================================
# 文档分块器
# ============================================================

class DocumentChunker:
    """
    文档分块器 — Parent-Child 策略

    参数:
        chunk_size: 子块目标长度 (字符数)，默认 512
        overlap: 相邻子块的重叠长度 (字符数)，默认 64
        parent_ratio: 每个父块包含多少个子块，默认 4
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64, parent_ratio: int = 4, segmenter=None):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.parent_ratio = parent_ratio
        self.segmenter = segmenter

    def chunk(self, text: str, doc_id: str, metadata: dict = None) -> list[Chunk]:
        """
        将文本分块，生成 Parent-Child 结构的 chunks

        Args:
            text: 完整的条款文本
            doc_id: 文档唯一标识（用于生成 chunk_id）
            metadata: 文档元数据（保司、产品名等），会附加到每个 chunk

        Returns:
            list[Chunk]: 所有子块 + 所有父块

        两种父块策略:
          - BERT 语义分割 (segmenter 可用): 达摩院模型切出语义边界 → 每个段落=1个父块
          - 固定比例 (segmenter 不可用): 每 parent_ratio 个子块合并为 1 个父块
        """
        if metadata is None:
            metadata = {}

        if self.segmenter:
            return self._chunk_with_segmenter(text, doc_id, metadata)
        else:
            return self._chunk_fixed_ratio(text, doc_id, metadata)

    # ── BERT 语义父块 ──────────────────────

    def _chunk_with_segmenter(self, text: str, doc_id: str, metadata: dict) -> list[Chunk]:
        """BERT 语义分割 → 每个段落=1个父块 → 父块内独立生成了块"""
        try:
            parent_texts = self.segmenter.segment(text)
        except Exception as e:
            logger.warning(f"BERT 分割失败，降级为固定比例: {e}")
            return self._chunk_fixed_ratio(text, doc_id, metadata)

        if not parent_texts:
            return []

        logger.info(f"BERT 语义分割: {len(parent_texts)} 个父块")

        chunks = []
        child_index = 0

        for parent_idx, parent_text in enumerate(parent_texts):
            parent_chunk_id = f"{doc_id}_parent_{parent_idx:04d}"

            # ── 在父块内: 句子切分 → 合并子块 ──
            sentences = self._split_sentences(parent_text)
            child_texts = self._merge_to_chunks(sentences, self.chunk_size, self.overlap)

            # ── 构造子块 ──
            for child_text in child_texts:
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}_child_{child_index:04d}",
                    text=child_text,
                    chunk_type="child",
                    parent_id=parent_chunk_id,
                    parent_text=parent_text,
                    chunk_index=child_index,
                    metadata={**metadata},
                ))
                child_index += 1

            # ── 构造父块 ──
            chunks.append(Chunk(
                chunk_id=parent_chunk_id,
                text=parent_text,
                chunk_type="parent",
                parent_id=None,
                parent_text=parent_text,
                chunk_index=-1,
                metadata={**metadata},
            ))

        logger.info(
            f"分块完成: {child_index} 子块 + {len(parent_texts)} 父块 = {len(chunks)} 总计"
        )
        return chunks

    # ── 固定比例父块 (原逻辑) ────────────

    def _chunk_fixed_ratio(self, text: str, doc_id: str, metadata: dict) -> list[Chunk]:
        """固定比例父块切分（原逻辑，segmenter 不可用时降级）"""
        if metadata is None:
            metadata = {}

        # ── 步骤 1: 按句子边界切分 ──
        sentences = self._split_sentences(text)
        logger.info(f"句子切分完成: {len(sentences)} 个句子")

        # ── 步骤 2: 合并为子块 (~chunk_size 字符) ──
        child_texts = self._merge_to_chunks(sentences, self.chunk_size, self.overlap)
        logger.info(f"子块生成: {len(child_texts)} 个子块")

        # ── 步骤 3: 合并为父块 (每 parent_ratio 个子块 = 1 个父块) ──
        parent_texts = []
        for i in range(0, len(child_texts), self.parent_ratio):
            group = child_texts[i:i + self.parent_ratio]
            parent_texts.append("".join(group))
        logger.info(f"父块生成: {len(parent_texts)} 个父块")

        # ── 步骤 4: 构造 Chunk 对象 ──
        chunks = []

        for i, child_text in enumerate(child_texts):
            parent_idx = i // self.parent_ratio
            parent_chunk_id = f"{doc_id}_parent_{parent_idx:04d}"

            chunks.append(Chunk(
                chunk_id=f"{doc_id}_child_{i:04d}",
                text=child_text,
                chunk_type="child",
                parent_id=parent_chunk_id,
                parent_text=parent_texts[parent_idx] if parent_idx < len(parent_texts) else child_text,
                chunk_index=i,
                metadata={**metadata},
            ))

        for i, parent_text in enumerate(parent_texts):
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_parent_{i:04d}",
                text=parent_text,
                chunk_type="parent",
                parent_id=None,
                parent_text=parent_text,
                chunk_index=-1,
                metadata={**metadata},
            ))

        logger.info(
            f"分块完成: {len(child_texts)} 子块 + {len(parent_texts)} 父块 = {len(chunks)} 总计"
        )

        return chunks

    # ============================================================
    # 内部方法
    # ============================================================

    def _split_sentences(self, text: str) -> list[str]:
        """
        按中文句子边界切分文本

        切分符号: 。！？；\n
        使用正则的后瞻断言 (?<=...)，保留标点符号在句尾

        为什么不按字数硬切:
          硬切会把一句话切成两半，破坏语义完整性。
          例如: "本保险合同的保险责任|包括住院医疗费用"
          切分后两个片段都不完整，向量编码的语义不准确。
        """
        # 按中文句末标点切分，保留标点
        sentences = re.split(r"(?<=[。！？；\n])", text)

        # 过滤空句子
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def _merge_to_chunks(
        self,
        sentences: list[str],
        chunk_size: int,
        overlap: int,
    ) -> list[str]:
        """
        将句子列表合并为目标长度的 chunks

        合并策略:
          1. 逐个添加句子，直到累计长度 >= chunk_size
          2. 当前 chunk 完成，开始新 chunk
          3. 新 chunk 保留上一个 chunk 的最后几句作为 overlap
             (防止关键信息正好被切在边界上)

        Args:
            sentences: 句子列表
            chunk_size: 目标 chunk 长度 (字符数)
            overlap: overlap 长度 (字符数)

        Returns:
            list[str]: chunk 文本列表
        """
        chunks = []
        current_sentences = []  # 当前 chunk 的句子列表
        current_length = 0      # 当前 chunk 的累计字符数

        for sent in sentences:
            sent_len = len(sent)

            # 如果加上这个句子会超过 chunk_size，且当前已有内容
            if current_length + sent_len > chunk_size and current_sentences:
                # 当前 chunk 完成
                chunks.append("".join(current_sentences))

                # ── overlap 处理 ──
                # 保留最后几个句子，作为下一个 chunk 的开头
                # 这样边界处的信息不会丢失
                overlap_sentences = []
                overlap_length = 0
                # 从后往前取句子，直到 overlap 长度
                for s in reversed(current_sentences):
                    if overlap_length + len(s) > overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_length += len(s)

                # 新 chunk 以 overlap 句子开头
                current_sentences = overlap_sentences + [sent]
                current_length = overlap_length + sent_len
            else:
                current_sentences.append(sent)
                current_length += sent_len

        # 最后一个 chunk（可能不足 chunk_size）
        if current_sentences:
            chunks.append("".join(current_sentences))

        return chunks
