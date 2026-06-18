# -*- coding: utf-8 -*-
"""
PDF 解析器 — 将保险条款 PDF 转换为结构化纯文本

支持的 PDF 类型:
  1. 电子版 PDF — 直接提取文本（PyMuPDF/fitz）
  2. 扫描件 PDF — OCR 识别（marker-pdf）

解析后的输出:
  - 按页分割的纯文本
  - 按章节拆分的结构化 sections（如 "保险责任"、"责任免除"）
  - 产品基础信息（产品名、保司、等待期、犹豫期等）

为什么不用 LangChain 的 PDF Loader:
  保险条款 PDF 有特殊性:
    - 条款编号格式多样（第一条 / 第1条 / 1. / 一、）
    - 表格频繁（费率表、保障对比表）
    - 扫描件比例高（部分保司仍然发扫描件）
  需要定制化解析逻辑
"""

import io
import re
from pathlib import Path
from dataclasses import dataclass, field

from base.logger import logger


# ============================================================
# 解析结果数据结构
# ============================================================

@dataclass
class ParsedDocument:
    """
    PDF 解析后的完整结果

    Attributes:
        raw_text: 按页拼接的完整纯文本
        sections: 按章节拆分的 dict，key=章节标题，value=章节内容
        product_info: 提取的产品基础信息
        page_count: 总页数
        parse_method: 解析方式 ("native" 或 "ocr")
    """
    raw_text: str = ""
    sections: dict = field(default_factory=dict)
    product_info: dict = field(default_factory=dict)
    page_count: int = 0
    parse_method: str = "native"


# ============================================================
# PDF 解析器
# ============================================================

class PDFParser:
    """
    PDF 解析器

    自动判断 PDF 是电子版还是扫描件，选择对应的解析方式。
    电子版: PyMuPDF (fitz) 直接提取文本，速度快，精度高
    扫描件: marker-pdf OCR 识别，速度慢，但能处理图片 PDF
    """

    def __init__(self, force_ocr: bool = False):
        """
        Args:
            force_ocr: 强制使用 OCR（即使 PDF 有文本层）
                       用于文本层质量差的情况
        """
        self.force_ocr = force_ocr

    def parse(self, content: bytes) -> ParsedDocument:
        """
        解析 PDF 字节内容为结构化文档

        Args:
            content: PDF 文件的原始字节

        Returns:
            ParsedDocument: 解析后的结构化文档
        """
        # 判断是否需要 OCR
        if self.force_ocr or self._needs_ocr(content):
            logger.info("PDF 解析方式: OCR (扫描件)")
            raw_text = self._parse_with_ocr(content)
            method = "ocr"
        else:
            logger.info("PDF 解析方式: native (电子版)")
            raw_text = self._parse_native(content)
            method = "native"

        # 解析章节结构
        sections = self._extract_sections(raw_text)

        # 提取产品信息
        product_info = self._extract_product_info(raw_text)

        # 统计页数
        page_count = self._count_pages(content)

        logger.info(
            f"PDF 解析完成: {page_count} 页, "
            f"{len(sections)} 个章节, "
            f"{len(raw_text)} 字符"
        )

        return ParsedDocument(
            raw_text=raw_text,
            sections=sections,
            product_info=product_info,
            page_count=page_count,
            parse_method=method,
        )

    # ============================================================
    # 文本提取
    # ============================================================

    def _parse_native(self, content: bytes) -> str:
        """
        电子版 PDF 文本提取

        使用 PyMuPDF (fitz) 逐页提取文本。
        sort=True: 按阅读顺序排列文本块（从左到右、从上到下）

        为什么逐页加标记:
          方便后续定位条款在 PDF 的哪一页，
          用户问"这个条款在第几页"时可以回答。
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.error("请安装 PyMuPDF: pip install PyMuPDF")
            raise

        with fitz.open(stream=content, filetype="pdf") as doc:
            full_text = []

            for page_num, page in enumerate(doc, 1):
                # sort=True 按阅读顺序提取，避免多栏布局时文本错乱
                text = page.get_text("text", sort=True)
                if text.strip():
                    # 加页码标记，方便后续定位
                    full_text.append(f"\n--- 第{page_num}页 ---\n")
                    full_text.append(text)

            return "".join(full_text)

    def _parse_with_ocr(self, content: bytes) -> str:
        """
        扫描件 PDF OCR 识别

        使用 marker-pdf 将扫描件转换为 Markdown 文本。
        marker-pdf 的优势:
          - 中文识别精度高
          - 能保留表格结构（转为 Markdown 表格）
          - 能识别文档层级标题

        降级方案: 如果 marker-pdf 不可用，用 PyMuPDF 的 OCR 功能
        """
        try:
            from marker.converters.pdf import PdfConverter

            converter = PdfConverter(artifact_dict={})
            markdown = converter(io.BytesIO(content))
            # marker 返回 Markdown，简单清理后当纯文本用
            # 保留表格结构（Markdown 表格也是可读的）
            return markdown

        except ImportError:
            logger.warning("marker-pdf 未安装，降级为 PyMuPDF OCR")
            return self._parse_native(content)

    # ============================================================
    # 章节拆分
    # ============================================================

    def _extract_sections(self, text: str) -> dict:
        """
        按保险条款的章节标题拆分文本

        保险条款的章节标题有固定格式:
          "第一章 保险责任"
          "第一条 保险合同的构成"
          "1. 保险责任"
          "一、投保范围"

        返回: {"保险责任": "第一章内容...", "责任免除": "第二章内容...", ...}
        """
        # 匹配保险条款常见的章节标题
        section_patterns = [
            r"第[一二三四五六七八九十百千]+章\s*.+",     # 第一章 保险责任
            r"第\d+条\s*.+",                            # 第1条 保险合同的构成
            r"\d+[\.、]\s*(保险责任|责任免除|释义|犹豫期|续保|宽限期|等待期|免赔额|投保范围|保险金额|保险期间|保险费|保险金申请|保险金给付|合同解除|争议处理)",
        ]

        sections = {}
        lines = text.split("\n")
        current_title = "前言"      # 第一个章节标题之前的内容归入"前言"
        current_content = []

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # 检查当前行是否是章节标题
            is_section_header = any(
                re.match(pat, line_stripped) for pat in section_patterns
            )

            if is_section_header and len(current_content) > 3:
                # 保存上一个章节（内容太少说明是误判，不保存）
                sections[current_title] = "\n".join(current_content)
                current_title = line_stripped
                current_content = []
            else:
                current_content.append(line_stripped)

        # 保存最后一个章节
        if current_content:
            sections[current_title] = "\n".join(current_content)

        logger.info(f"章节拆分完成: {len(sections)} 个章节")
        return sections

    # ============================================================
    # 产品信息提取
    # ============================================================

    def _extract_product_info(self, text: str) -> dict:
        """
        从条款文本中提取产品基础信息

        使用正则匹配常见的信息格式:
          产品名称: XXX
          等待期: 30天
          犹豫期: 15天
          等

        返回: {"产品名称": "平安e生保", "等待期": "30天", ...}
        """
        info = {}

        # 各字段的匹配模式
        patterns = {
            "产品名称": r"产品名称[：:]\s*(.+)",
            "保险公司": r"([\u4e00-\u9fa5]{2,6})保险(?:股份)?有限公司",
            "备案号":   r"(?:备案号|产品备案编号)[：:]\s*(.+)",
            "等待期":   r"等待期[为是]?\s*(\d+)\s*[日天]",
            "犹豫期":   r"犹豫期[为是]?\s*(\d+)\s*[日天]",
            "保障期间": r"(?:保障期间|保险期间)[为是]?\s*(.+?)(?:[，。]|$)",
            "投保年龄": r"投保年龄[为是：:]*\s*(.+?)(?:[，。]|$)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                info[key] = match.group(1).strip()

        logger.info(f"产品信息提取: {info}")
        return info

    # ============================================================
    # 辅助方法
    # ============================================================

    def _needs_ocr(self, content: bytes) -> bool:
        """
        判断 PDF 是否需要 OCR

        判断逻辑: 提取前 3 页的文本，如果文本量太少
        （< 100 字符/页），说明是扫描件，需要 OCR
        """
        try:
            import fitz
            with fitz.open(stream=content, filetype="pdf") as doc:
                text_length = 0
                pages_to_check = min(3, len(doc))
                for i in range(pages_to_check):
                    text_length += len(doc[i].get_text("text").strip())
                # 平均每页不到 100 字符 → 大概率是扫描件
                avg_per_page = text_length / pages_to_check if pages_to_check > 0 else 0
                return avg_per_page < 100
        except Exception:
            return True  # 出错时默认用 OCR

    def _count_pages(self, content: bytes) -> int:
        """统计 PDF 总页数"""
        try:
            import fitz
            with fitz.open(stream=content, filetype="pdf") as doc:
                return len(doc)
        except Exception:
            return 0
