# -*- coding: utf-8 -*-
"""
API 请求/响应模型

从 gateway/app.py 中抽离，方便 auth.py 和 app.py 共用。
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """聊天请求"""

    query: str = Field(..., description="用户问题", min_length=1, max_length=2000)
    session_id: str = Field(default="", description="会话 ID（多轮对话时传入）")


class ChatResponse(BaseModel):
    """聊天响应（支持双通道）"""

    answer: str = Field(description="回答内容")
    intent: str = Field(default="", description="识别的意图")
    strategy: str = Field(default="", description="使用的检索策略")
    sources: list[dict] = Field(default_factory=list, description="引用的条款来源")
    latency_ms: float = Field(default=0, description="处理耗时（毫秒）")
    cache_hit: bool = Field(default=False, description="是否命中缓存")

    # ── 双通道新增字段 ──
    channel: str = Field(default="agent", description="通道标识: 'agent' 或 'customer'")
    route_mode: str = Field(default="", description="路由模式: 'rag' 或 'multi_agent'")
    agents_used: list[str] = Field(default_factory=list, description="使用的 Agent 列表")
    disclaimer: str = Field(default="", description="合规声明")


class IngestRequest(BaseModel):
    """文档摄取请求"""

    pdf_paths: list[str] = Field(description="PDF 文件路径列表")
    insurer: str = Field(description="保险公司名称")
    product_name: str = Field(description="产品名称")
    product_code: str = Field(description="产品编码")
