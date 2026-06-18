# -*- coding: utf-8 -*-
"""
文档摄取编排器 — 新产品 PDF 上线的完整流程

完整流程:
  PDF 文件 (API拉取 / SFTP同步 / 人工上传)
  → MD5 去重检查 (同一 PDF 不重复处理)
  → PDF 解析 (PyMuPDF / marker-pdf)
  → 章节拆分 + 产品信息提取
  → 文档分块 (Parent-Child)
  → BGE-M3 向量编码 (Dense + Sparse)
  → Milvus 写入 (向量 + 12 个标量字段)
  → MySQL 写入 (元数据 + 全文备份)
  → 旧版本标记失效

三种文档来源渠道:
  1. API 拉取 — 保司提供文档下发接口
  2. SFTP 同步 — 保司定期推文件到 SFTP 目录
  3. 人工上传 — 管理后台上传到临时目录
"""

import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

from base.logger import logger
from base.encoder import get_encoder
from rag_qa.ingestion.pdf_parser import PDFParser
from rag_qa.ingestion.document_chunker import DocumentChunker


# ============================================================
# 文档模型 — 统一三种来源的文档格式
# ============================================================

@dataclass
class ProductDocument:
    """
    保司产品文档 — 统一模型

    无论来源是 API、SFTP 还是人工上传，
    都转换为这个统一格式后进入处理管道。

    Attributes:
        insurer: 保司名称，如 "平安健康"
        product_name: 产品名称，如 "平安e生保2025"
        product_code: 产品编码，如 "PA-ESB-2025"
        doc_type: 文档类型: 条款 / 投保须知 / 费率表 / 理赔指南
        file_name: 原始文件名
        file_content: 文件原始字节
        file_md5: MD5 值（用于去重）
        version: 文档版本号
        received_at: 接收时间
    """
    insurer: str
    product_name: str
    product_code: str
    doc_type: str = "条款"
    file_name: str = ""
    file_content: bytes = b""
    file_md5: str = ""
    version: str = "1.0"
    received_at: datetime = field(default_factory=datetime.now)


# ============================================================
# 摄取编排器
# ============================================================

class IngestionOrchestrator:
    """
    文档摄取编排器 — 串联 PDF 解析 → 分块 → 编码 → 入库

    使用方式:
        orchestrator = IngestionOrchestrator(vector_store, mysql_session)
        orchestrator.ingest(documents)

    设计原则:
      - MD5 去重: 同一个 PDF 只处理一次
      - 版本管理: 新版本上线时旧版本标记失效
      - 错误隔离: 单个文档失败不影响其他文档
    """

    def __init__(self, vector_store=None, mysql_session=None):
        """
        Args:
            vector_store: VectorStore 实例（Milvus 写入）
            mysql_session: MySQL 会话（元数据写入）
        """
        self.parser = PDFParser()
        from rag_qa.ingestion.bert_segmenter import get_segmenter

        self.chunker = DocumentChunker(
            chunk_size=512, overlap=64, segmenter=get_segmenter()
        )
        self.vector_store = vector_store
        self.mysql = mysql_session

    def ingest(self, documents: list[ProductDocument]) -> dict:
        """
        批量处理文档 — 完整摄取管道

        Args:
            documents: ProductDocument 列表

        Returns:
            dict: 处理结果统计
                {"total": 10, "success": 8, "skipped": 1, "failed": 1}
        """
        stats = {"total": len(documents), "success": 0, "skipped": 0, "failed": 0}

        for doc in documents:
            try:
                result = self._process_single(doc)
                if result == "success":
                    stats["success"] += 1
                elif result == "skipped":
                    stats["skipped"] += 1
            except Exception as e:
                # 单个文档失败不影响其他文档
                logger.error(f"文档处理失败: {doc.file_name} - {e}", exc_info=True)
                stats["failed"] += 1

        logger.info(
            f"摄取完成: 总共 {stats['total']} 个, "
            f"成功 {stats['success']}, 跳过 {stats['skipped']}, "
            f"失败 {stats['failed']}"
        )
        return stats

    def _process_single(self, doc: ProductDocument) -> str:
        """
        处理单个文档的完整流程

        Returns:
            "success" / "skipped"
        """
        # ── 步骤 1: 计算 MD5 + 去重检查 ──
        if not doc.file_md5:
            doc.file_md5 = hashlib.md5(doc.file_content).hexdigest()

        if self._already_exists(doc.file_md5):
            logger.info(f"跳过重复文档: {doc.file_name} (MD5={doc.file_md5})")
            return "skipped"

        # ── 步骤 2: PDF 解析 ──
        parsed = self.parser.parse(doc.file_content)

        # 检查解析质量：文本太短说明解析可能失败
        if len(parsed.raw_text) < 100:
            logger.warning(f"文档文本过短 ({len(parsed.raw_text)} 字符): {doc.file_name}")
            # 不 raise，继续尝试（可能是很短的文档）

        # ── 步骤 3: 生成 doc_id ──
        # 格式: {产品名}_{MD5前8位}，确保唯一
        doc_id = f"{doc.product_code}_{doc.file_md5[:8]}"

        # ── 步骤 4: 文档分块 (Parent-Child) ──
        # metadata 会附加到每个 chunk，写入 Milvus 标量字段
        metadata = {
            "insurer": doc.insurer,
            "product_name": doc.product_name,
            "product_code": doc.product_code,
            "doc_type": doc.doc_type,
            "version": doc.version,
        }
        chunks = self.chunker.chunk(parsed.raw_text, doc_id, metadata)

        # ── 步骤 5: 按章节标注 clause_type ──
        # 根据章节拆分结果，给每个 chunk 标注属于哪个条款章节
        self._annotate_clause_type(chunks, parsed.sections)

        # ── 步骤 6: BGE-M3 批量编码 ──
        # 将所有 chunk 的 text 列表一次性批量编码
        # 比逐条编码效率高 10x+ (GPU 利用率)
        encoder = get_encoder()
        chunk_texts = [c.text for c in chunks]
        dense_all, sparse_all = encoder.encode_batch(chunk_texts)

        for i, chunk in enumerate(chunks):
            chunk.metadata["dense_vector"] = dense_all[i]
            chunk.metadata["sparse_vector"] = sparse_all[i]

        # ── 步骤 7: 写入 Milvus ──
        if self.vector_store:
            milvus_entities = []
            for chunk in chunks:
                milvus_entities.append({
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "dense_vector": chunk.metadata.get("dense_vector"),
                    "sparse_vector": chunk.metadata.get("sparse_vector", {}),
                    "insurer": chunk.metadata.get("insurer", ""),
                    "product_name": chunk.metadata.get("product_name", ""),
                    "product_code": chunk.metadata.get("product_code", ""),
                    "doc_type": chunk.metadata.get("doc_type", ""),
                    "clause_type": chunk.metadata.get("clause_type", ""),
                    "chunk_type": chunk.chunk_type,
                    "parent_id": chunk.parent_id or "",
                    "is_valid": True,
                    "version": chunk.metadata.get("version", "1.0"),
                })
            self.vector_store.insert(milvus_entities)
            self.vector_store.flush()

        # ── 步骤 8: 写入 MySQL 元数据 ──
        if self.mysql:
            self._save_to_mysql(doc, parsed, doc_id)

        # ── 步骤 9: 旧版本标记失效 ──
        if self.vector_store:
            self.vector_store.invalidate_by_product(
                doc.insurer, doc.product_code, doc.version
            )

        logger.info(
            f"✅ 文档摄取成功: {doc.product_name} ({doc.doc_type}) "
            f"| {len(chunks)} chunks | doc_id={doc_id}"
        )
        return "success"

    # ============================================================
    # 内部方法
    # ============================================================

    def _already_exists(self, md5: str) -> bool:
        """
        检查文档是否已入库（MD5 去重）

        查 MySQL 的 documents 表，如果 file_md5 已存在则跳过。
        同一个 PDF 文件（内容完全相同）的 MD5 是相同的。
        """
        if self.mysql is None:
            return False

        try:
            result = self.mysql.execute(
                "SELECT 1 FROM documents WHERE file_md5 = %s",
                (md5,),
            )
            return result.fetchone() is not None
        except Exception as e:
            logger.warning(f"MD5 去重查询失败: {e}")
            return False

    def _annotate_clause_type(self, chunks: list, sections: dict):
        """
        根据章节拆分结果，给每个 chunk 标注 clause_type

        方法: 检查 chunk 文本的前 50 字是否出现在某个章节的内容中。
        如果在，就把该章节标题作为 chunk 的 clause_type。

        clause_type 会写入 Milvus 的标量字段，
        支持按条款章节过滤检索（如只检索"保险责任"章节）。
        """
        for chunk in chunks:
            if chunk.chunk_type != "child":
                continue  # 父块不需要标注

            # 取 chunk 文本的前 50 字作为定位标记
            marker = chunk.text[:50]

            for section_title, section_content in sections.items():
                if marker in section_content:
                    chunk.metadata["clause_type"] = section_title
                    break
            else:
                # 没有匹配到任何章节 → 标记为"其他"
                chunk.metadata["clause_type"] = "其他"

    def _save_to_mysql(self, doc: ProductDocument, parsed, doc_id: str):
        """
        将文档元数据写入 MySQL

        存储内容:
          - documents 表: 文档级元数据 (产品名、MD5、解析信息)
          - document_chunks 表: chunk 级数据 (文本、类型、关联关系)
        """
        try:
            # 写入 documents 表
            self.mysql.execute(
                """INSERT INTO documents
                   (doc_id, insurer, product_name, product_code,
                    doc_type, file_name, file_md5, version,
                    page_count, raw_text, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                   ON DUPLICATE KEY UPDATE version=VALUES(version)""",
                (doc_id, doc.insurer, doc.product_name, doc.product_code,
                 doc.doc_type, doc.file_name, doc.file_md5, doc.version,
                 parsed.page_count, parsed.raw_text[:10000]),  # 全文截断存储
            )

            logger.info(f"MySQL documents 表写入成功: {doc_id}")

        except Exception as e:
            logger.error(f"MySQL 写入失败: {e}")
            # MySQL 写入失败不阻断 Milvus 写入（向量检索仍然可用）


# ============================================================
# 便捷入口: 从本地 PDF 文件列表摄取
# ============================================================

def ingest_local_pdfs(
    pdf_paths: list[str],
    insurer: str,
    product_name: str,
    product_code: str,
    vector_store=None,
    mysql_session=None,
) -> dict:
    """
    便捷函数: 从本地 PDF 文件列表摄取新产品

    用法:
        from rag_qa.ingestion.ingestion_orchestrator import ingest_local_pdfs
        stats = ingest_local_pdfs(
            pdf_paths=["/data/平安e生保_条款.pdf", "/data/平安e生保_投保须知.pdf"],
            insurer="平安健康",
            product_name="平安e生保2025",
            product_code="PA-ESB-2025",
        )

    Args:
        pdf_paths: PDF 文件路径列表
        insurer: 保司名称
        product_name: 产品名称
        product_code: 产品编码
        vector_store: VectorStore 实例 (可选)
        mysql_session: MySQL 会话 (可选)

    Returns:
        dict: 处理统计 {"total": N, "success": N, "skipped": N, "failed": N}
    """
    documents = []

    for path_str in pdf_paths:
        path = Path(path_str)
        if not path.exists():
            logger.warning(f"文件不存在，跳过: {path}")
            continue

        content = path.read_bytes()
        md5 = hashlib.md5(content).hexdigest()

        # 从文件名推断文档类型
        doc_type = "条款"  # 默认
        if "投保须知" in path.name:
            doc_type = "投保须知"
        elif "费率" in path.name:
            doc_type = "费率表"
        elif "理赔" in path.name:
            doc_type = "理赔指南"

        documents.append(ProductDocument(
            insurer=insurer,
            product_name=product_name,
            product_code=product_code,
            doc_type=doc_type,
            file_name=path.name,
            file_content=content,
            file_md5=md5,
        ))

    orchestrator = IngestionOrchestrator(vector_store, mysql_session)
    return orchestrator.ingest(documents)
