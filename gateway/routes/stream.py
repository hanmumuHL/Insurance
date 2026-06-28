# -*- coding: utf-8 -*-
"""
GET /chat/stream 端点 — SSE 流式问答（按角色自动分流）

Customer 通道: RAG Pipeline 流式输出 + 逐句推送 + 免责声明
Agent 通道:    RAG Pipeline 流式输出（内部使用，无冗余声明）
"""

import json
import asyncio
from fastapi import APIRouter, Depends, Query
from sse_starlette.sse import EventSourceResponse

from base.logger import logger
from gateway.auth import UserContext, get_current_user
from gateway.services import get_rag_system

router = APIRouter()


@router.get("/chat/stream")
async def chat_stream(
    query: str = Query(..., description="用户问题"),
    session_id: str = Query(default="", description="会话 ID"),
    user: UserContext = Depends(get_current_user),
):
    """
    SSE 流式问答端点 — 按角色自动分流

    Headers 同 POST /chat。
    """

    async def event_generator():
        channel = user.channel
        user_role = user.role.value
        rag_system = get_rag_system()

        # ── 是否启用 Stage 级事件（旧调试模式，走同步 run()）──
        stage_events_enabled = session_id and "stage_events=1" in session_id

        try:
            # ── 推送状态: 开始处理 ──
            yield {
                "event": "status",
                "data": json.dumps(
                    {"stage": "analyzing", "message": "正在分析问题...",
                     "channel": channel},
                    ensure_ascii=False,
                ),
            }

            if stage_events_enabled:
                # ── 旧路径: 同步 run() + 逐句拆分（用于调试 Stage 耗时）──
                stage_events = []
                def on_stage_complete(stage_name: str, stage_result):
                    stage_events.append({
                        "stage": stage_name,
                        "status": stage_result.status,
                        "timing_ms": stage_result.timing_ms,
                    })

                if rag_system._pipeline is not None:
                    result = await asyncio.to_thread(
                        rag_system._pipeline.run,
                        query=query,
                        session_id=session_id,
                        user_role=user_role,
                        on_stage_complete=on_stage_complete,
                    )
                else:
                    result = await asyncio.to_thread(
                        rag_system.query,
                        raw_query=query,
                        session_id=session_id,
                        user_role=user_role,
                    )

                for evt in stage_events:
                    yield {
                        "event": "status",
                        "data": json.dumps(
                            {
                                "stage": evt["stage"],
                                "message": f"{evt['stage']} ({evt['status']}, {evt['timing_ms']}ms)",
                                "channel": channel,
                            },
                            ensure_ascii=False,
                        ),
                    }
                    await asyncio.sleep(0.01)

                yield {
                    "event": "status",
                    "data": json.dumps(
                        {
                            "stage": "searching",
                            "message": f"检索完成，找到 {len(result.sources)} 条相关条款",
                            "channel": channel,
                        },
                        ensure_ascii=False,
                    ),
                }

                sentences = result.answer.split("。")
                for i, sentence in enumerate(sentences):
                    if sentence.strip():
                        text = sentence.strip() + ("。" if i < len(sentences) - 1 else "")
                        yield {
                            "event": "answer",
                            "data": json.dumps({"text": text}, ensure_ascii=False),
                        }
                        await asyncio.sleep(0.05)

                if user.is_customer:
                    disclaimer = (
                        "⚠️ 以上信息基于保险条款解读，不构成投保或理赔建议。"
                        "具体以保险合同约定和保险公司审核为准。"
                    )
                    for sentence in disclaimer.split("。"):
                        if sentence.strip():
                            yield {
                                "event": "answer",
                                "data": json.dumps(
                                    {"text": sentence.strip() + "。"},
                                    ensure_ascii=False,
                                ),
                            }
                            await asyncio.sleep(0.03)

                if result.sources:
                    yield {
                        "event": "sources",
                        "data": json.dumps({"sources": result.sources}, ensure_ascii=False),
                    }

                yield {
                    "event": "done",
                    "data": json.dumps(
                        {
                            "latency_ms": result.latency_ms,
                            "intent": result.intent,
                            "strategy": result.strategy,
                            "cache_hit": result.cache_hit,
                            "channel": channel,
                        },
                        ensure_ascii=False,
                    ),
                }

            else:
                # ── 新路径: astream() → 真 token 级流式 ──
                pipeline = rag_system._pipeline
                if pipeline is None:
                    from rag_qa.core.pipeline.smart_pipeline import SmartPipeline
                    pipeline = SmartPipeline(rag_system=rag_system)

                async for evt in pipeline.astream(
                    query=query,
                    session_id=session_id,
                    user_role=user_role,
                ):
                    if evt.type == "status":
                        yield {
                            "event": "status",
                            "data": json.dumps(
                                {**evt.data, "channel": channel},
                                ensure_ascii=False,
                            ),
                        }
                    elif evt.type == "answer":
                        yield {
                            "event": "answer",
                            "data": json.dumps(
                                {"text": evt.data["text"]},
                                ensure_ascii=False,
                            ),
                        }
                    elif evt.type == "replace":
                        yield {
                            "event": "replace",
                            "data": json.dumps(
                                {"text": evt.data["text"]},
                                ensure_ascii=False,
                            ),
                        }
                    elif evt.type == "sources":
                        yield {
                            "event": "sources",
                            "data": json.dumps(
                                {"sources": evt.data["sources"]},
                                ensure_ascii=False,
                            ),
                        }
                    elif evt.type == "done":
                        if user.is_customer:
                            yield {
                                "event": "answer",
                                "data": json.dumps(
                                    {
                                        "text": (
                                            "⚠️ 以上信息基于保险条款解读，不构成投保或理赔建议。"
                                            "具体以保险合同约定和保险公司审核为准。"
                                        ),
                                    },
                                    ensure_ascii=False,
                                ),
                            }
                        yield {
                            "event": "done",
                            "data": json.dumps(
                                {**evt.data, "channel": channel},
                                ensure_ascii=False,
                            ),
                        }
                    elif evt.type == "error":
                        yield {
                            "event": "error",
                            "data": json.dumps(
                                {"error": evt.data.get("error", "处理失败")},
                                ensure_ascii=False,
                            ),
                        }

        except Exception as e:
            logger.error(f"[{channel.upper()}] 流式处理失败: {e}", exc_info=True)
            yield {
                "event": "error",
                "data": json.dumps(
                    {"error": "处理失败，请稍后再试"}, ensure_ascii=False
                ),
            }

    return EventSourceResponse(event_generator())
