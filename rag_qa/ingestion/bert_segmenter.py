"""
达摩院 BERT 文档语义分割器

模型: nlp_bert_document-segmentation_chinese-base
原理: Token 级 B-EOP 标签 → 句子级密度聚合 → 段落边界判定
  - 章节标题句子: B-EOP 密度高 (>0.3)
  - 正文句子: B-EOP 密度低 (<0.1)
  - 段落边界: B-EOP 密度从高→低的转换点

用途: 替代 document_chunker 中固定 parent_ratio 的父块切分

使用:
  segmenter = get_segmenter()
  paragraphs = segmenter.segment("保险条款全文...")
"""

import re
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from base.logger import logger
from config.settings import settings

# 句子级 B-EOP 密度阈值
B_EOP_THRESHOLD = 0.3      # 高于此值视为结构性句子
BODY_THRESHOLD = 0.1        # 低于此值视为正文

# 段落长度约束
MIN_PARAGRAPH_LEN = 100     # 最短段落 (字)，短文本向左合并
MAX_PARAGRAPH_LEN = 4000    # 最长段落 (字)，超长内部再切

# 滑动窗口参数（用于超过 512 token 的长文本）
WINDOW_SIZE = 400
OVERLAP = 50


class BERTDocumentSegmenter:
    """达摩院 BERT 文档语义分割器 — 句子级 B-EOP 密度方案"""

    def __init__(self, model_path: str = None):
        self.model_path = model_path or settings.models.bert_segmenter
        self.model: Optional[AutoModelForTokenClassification] = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded = False

    # ── 加载 ──────────────────────────────

    def load(self):
        """加载模型和 tokenizer"""
        if self._loaded:
            return

        try:
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            logger.info(f"[Segmenter] 模型加载完成: {self.model_path} device={self.device}")
        except Exception as e:
            logger.error(f"[Segmenter] 模型加载失败: {e}")
            raise

    # ── 核心: 语义分割 ─────────────────────

    def segment(self, text: str) -> list[str]:
        """
        将文本按语义分割为段落

        流程:
          1. 句子切分
          2. Token 级 B-EOP 预测
          3. 句子级 B-EOP 密度计算
          4. 按密度阈值判定段落边界
          5. 边界修正（短段落合并、长段落再切）
        """
        if not self._loaded:
            self.load()

        if not text or len(text) < MIN_PARAGRAPH_LEN:
            return [text] if text else []

        # ── 句子切分 ──
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [text]

        # ── Token 级 B-EOP 预测 + 句子级密度 ──
        ratios = self._compute_sentence_ratios(text, sentences)

        # ── 按密度阈值判定段落边界 ──
        paragraphs = self._group_into_paragraphs(sentences, ratios)

        # ── 边界修正 ──
        paragraphs = self._merge_short(paragraphs, MIN_PARAGRAPH_LEN)
        paragraphs = self._split_long(paragraphs, MAX_PARAGRAPH_LEN)

        logger.info(
            f"[Segmenter] 分割完成: {len(text)}字 → {len(paragraphs)} 个段落 "
            f"(avg={sum(len(p) for p in paragraphs)//max(len(paragraphs),1)}字)"
        )
        return paragraphs

    # ── Token 级预测 + 句子级聚合 ──────────

    def _compute_sentence_ratios(self, text: str, sentences: list[str]) -> list[float]:
        """计算每个句子的 B-EOP 密度"""
        # Tokenize
        encoding = self.tokenizer(
            text, return_offsets_mapping=True, return_tensors="pt"
        )
        input_ids = encoding["input_ids"]
        offsets = encoding["offset_mapping"][0].numpy()

        # 纯文本过长 → 滑动窗口推理
        if len(input_ids[0]) > WINDOW_SIZE:
            preds = self._predict_long(text, len(offsets), offsets)
        else:
            preds = self._predict_short(input_ids, offsets)

        # 每个句子计算 B-EOP 密度
        ratios = []
        char_pos = 0
        for sent in sentences:
            sent_len = len(sent)
            b_eop_count = 0
            total = 0
            for i, (cs, ce) in enumerate(offsets):
                if cs >= ce:
                    continue
                if cs >= char_pos and ce <= char_pos + sent_len:
                    total += 1
                    if preds[i] == 0:
                        b_eop_count += 1
            ratio = b_eop_count / total if total > 0 else 0
            ratios.append(ratio)
            char_pos += sent_len

        return ratios

    def _predict_short(self, input_ids, offsets) -> list[int]:
        """短文本单次推理"""
        input_ids = input_ids.to(self.device)
        attention_mask = torch.ones_like(input_ids).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()
        return preds.tolist()

    def _predict_long(self, text: str, total_tokens: int, all_offsets) -> list[int]:
        """长文本滑动窗口推理 + 投票"""
        import numpy as np

        encoding = self.tokenizer(
            text, return_offsets_mapping=True, add_special_tokens=False
        )
        all_input_ids = encoding["input_ids"]
        local_offsets = encoding["offset_mapping"]

        votes = np.zeros(total_tokens, dtype=int)
        total_passes = np.zeros(total_tokens, dtype=int)

        stride = WINDOW_SIZE - OVERLAP
        for start in range(0, len(all_input_ids), stride):
            end = min(start + WINDOW_SIZE, len(all_input_ids))
            window_ids = [self.tokenizer.cls_token_id] + all_input_ids[start:end] + [self.tokenizer.sep_token_id]
            input_ids = torch.tensor([window_ids]).to(self.device)
            attention_mask = torch.ones_like(input_ids).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()

            for i, pred in enumerate(preds[1:]):
                if i >= len(window_ids) - 2:
                    break
                global_idx = start + i
                if global_idx < total_tokens:
                    total_passes[global_idx] += 1
                    if pred == 0:
                        votes[global_idx] += 1

        # 投票: ≥50% → B-EOP, 否则 → O
        final_preds = []
        for i in range(total_tokens):
            if total_passes[i] > 0 and votes[i] / total_passes[i] >= 0.5:
                final_preds.append(0)
            else:
                final_preds.append(1)

        return final_preds

    # ── 段落分组 ──────────────────────────

    def _group_into_paragraphs(self, sentences: list[str], ratios: list[float]) -> list[str]:
        """
        按 B-EOP 密度将句子分组为段落

        规则: 遇到结构句 (ratio >= threshold) → 开启新段落
              正文句跟随在前面最近的段落中
        """
        if not sentences:
            return []

        paragraphs = []
        current = sentences[0]

        for i in range(1, len(sentences)):
            if ratios[i] >= B_EOP_THRESHOLD:
                # 结构句：结束当前段落，开启新段落
                paragraphs.append(current)
                current = sentences[i]
            else:
                current += sentences[i]

        if current:
            paragraphs.append(current)

        return paragraphs

    # ── 句子切分 ──────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """按中文句末标点切分（不含分号，分号不构成段落边界）"""
        sentences = re.split(r"(?<=[。！？\n])", text)
        return [s.strip() for s in sentences if s.strip()]

    # ── 边界修正 ──────────────────────────

    @staticmethod
    def _merge_short(paragraphs: list[str], min_len: int) -> list[str]:
        """合并过短段落，但保留以章节头开头的段落"""
        import re

        header_pattern = re.compile(r"第[一二三四五六七八九十百千\d]+[章节条]")

        if not paragraphs:
            return paragraphs

        if len(paragraphs) >= 2 and len(paragraphs[0]) < min_len and not header_pattern.match(paragraphs[0]):
            paragraphs[1] = paragraphs[0] + paragraphs[1]
            paragraphs = paragraphs[1:]

        merged = [paragraphs[0]]
        for para in paragraphs[1:]:
            if len(para) < min_len and merged and not header_pattern.match(para):
                merged[-1] += para
            else:
                merged.append(para)
        return merged

    @staticmethod
    def _split_long(paragraphs: list[str], max_len: int) -> list[str]:
        """拆分过长段落（按句号边界）"""
        result = []
        for para in paragraphs:
            if len(para) <= max_len:
                result.append(para)
            else:
                sentences = re.split(r"(?<=[。！？])", para)
                sentences = [s.strip() for s in sentences if s.strip()]
                current = ""
                for s in sentences:
                    if len(current) + len(s) > max_len and current:
                        result.append(current)
                        current = s
                    else:
                        current += s
                if current:
                    result.append(current)
        return result


# ── 单例工厂 ──────────────────────────────

_segmenter: Optional[BERTDocumentSegmenter] = None


def get_segmenter() -> Optional[BERTDocumentSegmenter]:
    """获取文档分割器单例"""
    global _segmenter
    if _segmenter is None:
        try:
            _segmenter = BERTDocumentSegmenter()
            _segmenter.load()
        except Exception as e:
            logger.warning(f"[Segmenter] 初始化失败，父块将使用固定比例切分: {e}")
            _segmenter = None
    return _segmenter
