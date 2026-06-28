# -*- coding: utf-8 -*-
"""闲聊/投诉 Stage — 不走检索，直接 LLM 回复或转人工"""

import time
from rag_qa.core.pipeline.stage import Stage
from rag_qa.core.pipeline.context import PipelineContext, StageResult


class ChitchatStage(Stage):
    """处理闲聊和投诉意图 — 不走检索管线"""

    name = "chitchat"

    def __init__(self, llm_client=None):
        self._llm_client = llm_client

    def can_execute(self, ctx: PipelineContext) -> bool:
        # 仅在闲聊或投诉意图时执行
        return ctx.intent in ("闲聊寒暄", "投诉建议")

    def execute(self, ctx: PipelineContext) -> StageResult:
        t0 = time.time()
        try:
            if ctx.intent == "投诉建议":
                answer = (
                    "非常抱歉给您带来不好的体验。我已记录您的反馈，"
                    "正在为您转接人工客服，请稍候。"
                )
            else:
                # 闲聊: LLM 直答（无检索上下文）
                answer = self._chitchat_reply(ctx.query)

            timing = round((time.time() - t0) * 1000, 1)
            ctx.pipeline[self.name] = timing
            ctx.generated_answer = answer
            ctx.complexity = 0  # 标记为已处理

            return StageResult(
                status="skip", data={"intent": ctx.intent}, timing_ms=timing
            )

        except Exception as e:
            timing = round((time.time() - t0) * 1000, 1)
            ctx.pipeline[self.name] = timing
            ctx.generated_answer = "抱歉，系统暂时繁忙，请稍后再试。"
            return StageResult(status="degraded", timing_ms=timing,
                               data={"warning": str(e)})

    def _chitchat_reply(self, query: str) -> str:
        """闲聊场景的 LLM 直答"""
        try:
            if self._llm_client is None:
                from base.llm_client import get_llm_client
                self._llm_client = get_llm_client()

            messages = [
                {
                    "role": "system",
                    "content": (
                        "你是保险智能客服，友好地回应用户的闲聊，"
                        "并引导到保险话题。回复简洁，不超过100字。"
                    ),
                },
                {"role": "user", "content": query},
            ]
            response = self._llm_client.chat(messages=messages)
            if response.error:
                return "您好！我是保险智能客服，有什么可以帮您的吗？"
            return response.content
        except Exception:
            return "您好！我是保险智能客服，有什么可以帮您的吗？"
