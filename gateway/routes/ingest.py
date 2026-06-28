# -*- coding: utf-8 -*-
"""
POST /admin/ingest 端点 — 文档摄取（仅 ADMIN）

新产品 PDF 上线流程:
  PDF 文件 → MD5 去重 → 解析 → 分块 → BGE-M3 编码 → Milvus + MySQL 写入
"""

from fastapi import APIRouter, Depends, HTTPException

from base.logger import logger
from gateway.models import IngestRequest
from gateway.auth import UserContext, ADMIN_ROLES, get_current_user

router = APIRouter()


@router.post("/admin/ingest")
async def ingest_documents(
    request: IngestRequest,
    user: UserContext = Depends(get_current_user),
):
    """
    文档摄取端点 — 新产品 PDF 上线

    仅 ADMIN 角色可调用。
    管理后台调用此端点，将新产品的条款 PDF 导入知识库。
    生产环境建议改为异步任务队列 (Celery / RQ)。
    """
    if user.role not in ADMIN_ROLES:
        logger.warning(
            f"[SECURITY] 非 ADMIN 用户尝试调用 /admin/ingest: "
            f"user={user.display_name} role={user.role.value}"
        )
        raise HTTPException(
            status_code=403,
            detail=f"权限不足: 仅 ADMIN 可执行文档导入，当前角色: {user.role.value}",
        )

    logger.info(
        f"[ADMIN] 文档摄取请求: user={user.display_name} "
        f"product={request.product_name} ({len(request.pdf_paths)} 个文件)"
    )

    try:
        from rag_qa.ingestion.ingestion_orchestrator import ingest_local_pdfs

        stats = ingest_local_pdfs(
            pdf_paths=request.pdf_paths,
            insurer=request.insurer,
            product_name=request.product_name,
            product_code=request.product_code,
        )

        return {
            "status": "success",
            "stats": stats,
            "message": (
                f"摄取完成: 成功 {stats['success']} 个, "
                f"跳过 {stats['skipped']} 个, 失败 {stats['failed']} 个"
            ),
        }

    except Exception as e:
        logger.error(f"[ADMIN] 文档摄取失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"摄取失败: {e}")
