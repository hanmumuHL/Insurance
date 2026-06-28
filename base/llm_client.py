# -*- coding: utf-8 -*-
"""
LLM 客户端 — DeepSeek (主) + 千问 (降级) 双通道调用

设计背景:
  DeepSeek API 偶尔会超时或限流，需要一个降级方案。
  主通道: DeepSeek-V3 (deepseek-chat) — 性价比高，中文能力强
  降级通道: 通义千问 (qwen-max) — 阿里云生态，稳定性好

  两个 API 都兼容 OpenAI 接口，用同一个 openai Python SDK 调用。

调用策略:
  1. 先尝试 DeepSeek，成功则返回
  2. DeepSeek 失败 (超时/429/5xx) → 切换到千问
  3. 千问也失败 → 重试一次 DeepSeek
  4. 全部失败 → 返回兜底话术

熔断机制:
  DeepSeek 连续 5 次失败 → 熔断 30 秒
  熔断期间直接走千问，不再尝试 DeepSeek
  30 秒后自动恢复，试探性调用 DeepSeek
"""

import time
import json
import threading
from typing import Optional
from dataclasses import dataclass, field

from openai import OpenAI, AsyncOpenAI, APIError, APITimeoutError, RateLimitError, APIStatusError
from base.logger import logger
from config.settings import settings


# ============================================================
# LLM 调用结果
# ============================================================

@dataclass
class LLMResponse:
    """
    LLM 调用的返回结果

    Attributes:
        content: LLM 生成的文本
        model: 实际使用的模型名 (可能是主模型或降级模型)
        provider: 实际使用的提供商 ("deepseek" / "qwen")
        is_fallback: 是否使用了降级通道
        tokens_used: token 用量 (prompt + completion)
        latency_ms: 调用耗时 (毫秒)
        error: 如果失败，记录错误信息
    """
    content: str = ""
    model: str = ""
    provider: str = ""
    is_fallback: bool = False
    tokens_used: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    error: str = ""


# ============================================================
# 流式异常 — 用于通知调用方流式中断
# ============================================================

class LLMStreamError(Exception):
    """
    LLM 流式调用中途失败时抛出的异常

    与普通 Exception 的区别:
      partial_content 保存了失败前已经产出的 token 文本，
      调用方可以决定是用已产出的内容结束流，还是丢弃。

    使用场景:
      - async for token in client.astream_chat(...):
          ...
      - 如果 DeepSeek 在生成一半时断连，抛出此异常
      - Gateway 捕获后，可以用 partial_content 结束回答
    """

    def __init__(self, message: str, partial_content: str = ""):
        super().__init__(message)
        self.partial_content = partial_content


# ============================================================
# 熔断器 — 连续失败时自动切换
# ============================================================

class CircuitBreaker:
    """
    简单的熔断器实现

    工作原理:
      正常状态: 每次调用都尝试
      熔断状态: 连续失败 N 次后，直接跳过，不再尝试
      恢复: 等待 cool_down 秒后，试探性调用一次

    为什么要熔断:
      如果 DeepSeek 服务器宕机，每次都等超时 (30秒) 才降级，
      用户体验极差。熔断后直接走千问，延迟只有几百毫秒。
    """

    def __init__(self, failure_threshold: int = 5, cool_down_seconds: int = 30):
        """
        Args:
            failure_threshold: 连续失败多少次触发熔断
            cool_down_seconds: 熔断冷却时间 (秒)
        """
        self.failure_threshold = failure_threshold
        self.cool_down_seconds = cool_down_seconds
        self.failure_count = 0           # 连续失败计数
        self.last_failure_time = 0.0     # 上次失败时间戳
        self.is_open = False             # 是否处于熔断状态
        self._lock = threading.Lock()    # 线程安全锁

    def should_skip(self, has_fallback: bool = True) -> bool:
        """
        判断是否应该跳过 (熔断中)

        Args:
            has_fallback: 是否有降级通道可用。若无降级通道，即使熔断也不跳过。

        Returns:
            True: 熔断中，应该跳过直接用降级通道
            False: 正常状态，可以尝试调用
        """
        with self._lock:
            if not self.is_open:
                return False

            # 无降级通道时不跳过——跳过就是硬失败
            if not has_fallback:
                logger.info("无降级通道可用，熔断器暂不拦截 primary")
                return False

            # 检查冷却时间是否已过
            elapsed = time.time() - self.last_failure_time
            if elapsed >= self.cool_down_seconds:
                # 冷却期已过，半开状态，允许试探性调用
                logger.info(f"熔断器半开: 已冷却 {elapsed:.0f}s，试探性调用")
                self.is_open = False
                return False

            return True

    def record_success(self):
        """记录成功 — 重置失败计数，关闭熔断"""
        with self._lock:
            self.failure_count = 0
            self.is_open = False

    def record_failure(self):
        """记录失败 — 累计失败计数，达到阈值则熔断"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.is_open = True
                logger.warning(
                    f"熔断器触发: 连续失败 {self.failure_count} 次，"
                    f"冷却 {self.cool_down_seconds}s"
                )


# ============================================================
# LLM 客户端主体
# ============================================================

class LLMClient:
    """
    LLM 客户端 — 双通道 + 熔断 + 重试

    用法:
        client = LLMClient()

        # 普通聊天
        response = client.chat(messages=[
            {"role": "system", "content": "你是保险客服"},
            {"role": "user", "content": "肺炎住院能赔吗？"},
        ])
        print(response.content)

        # 指定参数
        response = client.chat(
            messages=messages,
            temperature=0,        # 分类任务用 0
            max_tokens=100,       # 限制输出长度
            model="deepseek-reasoner",  # 用 R1 做推理
        )
    """

    def __init__(self):
        """
        初始化两个 OpenAI 兼容客户端

        DeepSeek 和千问都实现了 OpenAI 兼容 API，
        所以用同一个 openai Python SDK，只是 base_url 和 api_key 不同。
        """
        cfg = settings.llm

        # ── 主通道: DeepSeek ──
        self.primary_client = OpenAI(
            api_key=cfg.deepseek_api_key,
            base_url=cfg.deepseek_base_url,
            timeout=30.0,           # 单次请求超时 30 秒
            max_retries=0,          # 不用 SDK 内置重试，我们自己控制
        )
        self.primary_model = cfg.primary_model  # "deepseek-chat"

        # ── 降级通道: 千问 ──
        # 只有在配置了 QWEN_API_KEY 时才初始化
        self.fallback_client = None
        self.fallback_model = cfg.fallback_model  # "qwen-max"
        if cfg.qwen_api_key and cfg.qwen_api_key != "sk-your-qwen-key":
            self.fallback_client = OpenAI(
                api_key=cfg.qwen_api_key,
                base_url=cfg.qwen_base_url,
                timeout=30.0,
                max_retries=0,
            )
            logger.info(f"降级通道已配置: {cfg.fallback_model}")
        else:
            logger.info("降级通道未配置 (QWEN_API_KEY 为空)")

        # ── 规划模型: DeepSeek-R1 ──
        self.planner_model = cfg.planner_model  # "deepseek-reasoner"

        # ── 异步客户端 (用于流式 SSE) ──
        self.async_primary = AsyncOpenAI(
            api_key=cfg.deepseek_api_key,
            base_url=cfg.deepseek_base_url,
            timeout=30.0,
            max_retries=0,
        )

        self.async_fallback = None
        if cfg.qwen_api_key and cfg.qwen_api_key != "sk-your-qwen-key":
            self.async_fallback = AsyncOpenAI(
                api_key=cfg.qwen_api_key,
                base_url=cfg.qwen_base_url,
                timeout=30.0,
                max_retries=0,
            )

        # ── 熔断器 ──
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            cool_down_seconds=30,
        )

        logger.info(f"LLM 客户端初始化完成: 主={self.primary_model}")

    def chat(
        self,
        messages: list[dict],
        temperature: float = None,
        max_tokens: int = None,
        model: str = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """
        调用 LLM — 主备切换 + 熔断 + 重试

        Args:
            messages: OpenAI 格式的消息列表
                [{"role": "system", "content": "..."},
                 {"role": "user", "content": "..."}]
            temperature: 温度参数 (0=确定性, 1=创造性)
                         None 时使用默认值 0.3
            max_tokens: 最大输出 token 数
                        None 时使用默认值 2048
            model: 指定模型名 (覆盖默认)
                   None 时: 普通对话用 primary_model, 规划用 planner_model
            json_mode: 是否要求 JSON 格式输出
                       True 时会添加 response_format={"type": "json_object"}

        Returns:
            LLMResponse: 包含生成文本、使用的模型、耗时等信息
        """
        cfg = settings.llm
        temperature = temperature if temperature is not None else cfg.temperature
        max_tokens = max_tokens if max_tokens is not None else cfg.max_tokens

        # ── 尝试主通道 (DeepSeek) ──
        has_fallback = self.fallback_client is not None
        if not self.circuit_breaker.should_skip(has_fallback=has_fallback):
            response = self._call_primary(
                messages, temperature, max_tokens, model, json_mode
            )
            if response and not response.error:
                self.circuit_breaker.record_success()
                return response
            else:
                self.circuit_breaker.record_failure()

        # ── 降级到千问 ──
        if self.fallback_client:
            response = self._call_fallback(
                messages, temperature, max_tokens, json_mode
            )
            if response and not response.error:
                return response

        # ── 重试一次主通道 (可能是偶发性故障) ──
        if self.circuit_breaker.failure_count < self.circuit_breaker.failure_threshold:
            logger.info("重试主通道...")
            response = self._call_primary(
                messages, temperature, max_tokens, model, json_mode
            )
            if response and not response.error:
                self.circuit_breaker.record_success()
                return response

        # ── 全部失败 ──
        logger.error("LLM 主备通道全部失败")
        return LLMResponse(
            content="抱歉，系统暂时繁忙，请稍后再试。",
            error="所有 LLM 通道均不可用",
        )

    def chat_for_planning(self, messages: list[dict]) -> LLMResponse:
        """
        规划专用调用 — 使用 DeepSeek-R1 (推理能力更强)

        R1 模型的特点:
          - 有 "思考链" (Chain of Thought)，会先推理再回答
          - 适合复杂的多步规划任务
          - 比普通 chat 模型慢一些 (~1-2秒)，但推理质量高

        在 Agent 的 Planner 节点中使用此方法。
        """
        return self.chat(
            messages=messages,
            temperature=0.1,    # 规划任务需要确定性
            max_tokens=4096,    # 规划输出可能较长
            model=self.planner_model,
        )

    def chat_json(self, messages: list[dict], temperature: float = 0, max_tokens: int = None) -> dict:
        """
        调用 LLM 并解析 JSON 返回

        用于需要结构化输出的场景 (意图分类、策略选择等)。
        自动设置 json_mode=True，并解析返回的 JSON。

        Args:
            messages: 消息列表
            temperature: 温度 (JSON 输出建议用 0)

        Returns:
            dict: 解析后的 JSON 对象
            如果解析失败，返回 {"error": "JSON解析失败", "raw": "..."}
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens or 2048,
            json_mode=True,
        )

        if response.error:
            return {"error": response.error}

        # 尝试解析 JSON
        try:
            # LLM 可能在 JSON 外面包了 ```json ... ```，需要清理
            text = response.content.strip()
            if text.startswith("```"):
                # 去掉 markdown 代码块包裹
                lines = text.split("\n")
                text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

            return json.loads(text)

        except json.JSONDecodeError as e:
            logger.warning(f"LLM JSON 解析失败: {e}\n原始输出: {response.content[:200]}")
            return {"error": "JSON解析失败", "raw": response.content}

    # ============================================================
    # 内部方法: 实际 API 调用
    # ============================================================

    def _call_primary(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        model: str = None,
        json_mode: bool = False,
    ) -> Optional[LLMResponse]:
        """
        调用主通道 (DeepSeek)

        Returns:
            LLMResponse 或 None (调用失败时)
        """
        use_model = model or self.primary_model

        try:
            start = time.time()

            # 构造请求参数
            kwargs = {
                "model": use_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            # JSON 模式: 强制 LLM 输出合法 JSON
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            # 调用 API
            completion = self.primary_client.chat.completions.create(**kwargs)

            latency = (time.time() - start) * 1000

            # 提取结果
            content = completion.choices[0].message.content or ""
            usage = completion.usage

            logger.info(
                f"DeepSeek 调用成功: model={use_model}, "
                f"tokens={usage.total_tokens if usage else '?'}, "
                f"latency={latency:.0f}ms"
            )

            return LLMResponse(
                content=content,
                model=use_model,
                provider="deepseek",
                is_fallback=False,
                tokens_used={
                    "prompt": usage.prompt_tokens if usage else 0,
                    "completion": usage.completion_tokens if usage else 0,
                    "total": usage.total_tokens if usage else 0,
                },
                latency_ms=latency,
            )

        except APITimeoutError:
            logger.warning("DeepSeek 调用超时")
            return LLMResponse(error="DeepSeek timeout")

        except RateLimitError:
            logger.warning("DeepSeek 限流 (429)")
            return LLMResponse(error="DeepSeek rate limit")

        except APIStatusError as e:
            logger.warning(f"DeepSeek API 错误: {e.status_code} {e.message}")
            return LLMResponse(error=f"DeepSeek {e.status_code}")

        except APIError as e:
            logger.warning(f"DeepSeek API 异常: {e}")
            return LLMResponse(error=f"DeepSeek error: {e}")

        except Exception as e:
            logger.error(f"DeepSeek 未知错误: {e}", exc_info=True)
            return LLMResponse(error=f"DeepSeek unknown: {e}")

    def _call_fallback(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        json_mode: bool = False,
    ) -> Optional[LLMResponse]:
        """
        调用降级通道 (千问)

        逻辑和主通道一样，只是用不同的 client 和 model。
        """
        try:
            start = time.time()

            kwargs = {
                "model": self.fallback_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            completion = self.fallback_client.chat.completions.create(**kwargs)

            latency = (time.time() - start) * 1000

            content = completion.choices[0].message.content or ""
            usage = completion.usage

            logger.info(
                f"千问降级调用成功: model={self.fallback_model}, "
                f"tokens={usage.total_tokens if usage else '?'}, "
                f"latency={latency:.0f}ms"
            )

            return LLMResponse(
                content=content,
                model=self.fallback_model,
                provider="qwen",
                is_fallback=True,
                tokens_used={
                    "prompt": usage.prompt_tokens if usage else 0,
                    "completion": usage.completion_tokens if usage else 0,
                    "total": usage.total_tokens if usage else 0,
                },
                latency_ms=latency,
            )

        except Exception as e:
            logger.error(f"千问降级也失败: {e}")
            return LLMResponse(error=f"Qwen error: {e}")

    # ============================================================
    # 异步流式方法: 用于 SSE token 级流式输出
    # ============================================================

    async def astream_chat(
        self,
        messages: list[dict],
        temperature: float = None,
        max_tokens: int = None,
        model: str = None,
    ):
        """
        异步流式调用 LLM — 逐 token yield，用于 SSE 实时推送

        与 chat() 共享同一套主备切换 + 熔断逻辑：
          1. 先尝试 DeepSeek 流式
          2. 第一个 token 前失败 → 切换千问流式
          3. 中途失败 → 抛出 LLMStreamError（含已产出文本）
          4. 全部失败 → 抛出 LLMStreamError

        Args:
            messages:    OpenAI 格式的消息列表
            temperature: 温度参数（None 则用默认值）
            max_tokens:  最大输出 token 数（None 则用默认值）
            model:       指定模型名（None 则用 primary_model）

        Yields:
            str: 每个 token 片段（delta.content）

        Raises:
            LLMStreamError: 所有通道不可用，或流式中途断连
        """
        cfg = settings.llm
        temperature = temperature if temperature is not None else cfg.temperature
        max_tokens = max_tokens if max_tokens is not None else cfg.max_tokens

        has_fallback = self.async_fallback is not None

        # ── 尝试主通道流式 ──
        if not self.circuit_breaker.should_skip(has_fallback=has_fallback):
            try:
                async for token in self._astream_primary(
                    messages, temperature, max_tokens, model
                ):
                    yield token
                self.circuit_breaker.record_success()
                return
            except LLMStreamError as e:
                self.circuit_breaker.record_failure()
                if e.partial_content:
                    # 已产出部分内容 → 无法恢复，向上抛出
                    raise
                # 第一个 token 之前就失败了 → 继续走降级

        # ── 降级到千问流式 ──
        if self.async_fallback:
            try:
                async for token in self._astream_fallback(
                    messages, temperature, max_tokens
                ):
                    yield token
                return
            except LLMStreamError as e:
                if e.partial_content:
                    raise
                # 继续走重试

        # ── 重试一次主通道 ──
        if self.circuit_breaker.failure_count < self.circuit_breaker.failure_threshold:
            logger.info("流式重试主通道...")
            try:
                async for token in self._astream_primary(
                    messages, temperature, max_tokens, model
                ):
                    yield token
                self.circuit_breaker.record_success()
                return
            except LLMStreamError as e:
                if e.partial_content:
                    raise

        # ── 全部失败 ──
        logger.error("LLM 流式主备通道全部失败")
        raise LLMStreamError("所有 LLM 通道均不可用", partial_content="")

    async def _astream_primary(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        model: str = None,
    ):
        """
        流式调用主通道 (DeepSeek)

        Yields:
            str: token 文本片段

        Raises:
            LLMStreamError: 调用失败（含已产出文本）
        """
        use_model = model or self.primary_model
        content_so_far = ""

        try:
            kwargs = {
                "model": use_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            }

            stream = await self.async_primary.chat.completions.create(**kwargs)

            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    content_so_far += delta.content
                    yield delta.content

            logger.info(
                f"DeepSeek 流式完成: model={use_model}, "
                f"tokens≈{len(content_so_far)} chars"
            )

        except APIError as e:
            logger.warning(f"DeepSeek 流式 API 错误: {e}")
            raise LLMStreamError(
                f"DeepSeek stream error: {e}",
                partial_content=content_so_far,
            )
        except Exception as e:
            logger.warning(f"DeepSeek 流式异常: {e}")
            raise LLMStreamError(
                f"DeepSeek stream error: {e}",
                partial_content=content_so_far,
            )

    async def _astream_fallback(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ):
        """
        流式调用降级通道 (千问)

        Yields:
            str: token 文本片段

        Raises:
            LLMStreamError: 调用失败（含已产出文本）
        """
        content_so_far = ""

        try:
            kwargs = {
                "model": self.fallback_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            }

            stream = await self.async_fallback.chat.completions.create(**kwargs)

            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    content_so_far += delta.content
                    yield delta.content

            logger.info(
                f"千问流式完成: model={self.fallback_model}, "
                f"tokens≈{len(content_so_far)} chars"
            )

        except Exception as e:
            logger.warning(f"千问流式异常: {e}")
            raise LLMStreamError(
                f"Qwen stream error: {e}",
                partial_content=content_so_far,
            )


# ============================================================
# 单例 — 整个应用共享一个 LLM 客户端
# ============================================================

_llm_client: Optional[LLMClient] = None
_llm_lock = threading.Lock()


def get_llm_client() -> LLMClient:
    """
    获取 LLM 客户端单例

    第一次调用时创建，后续复用。
    避免每次请求都创建新的 HTTP 连接。
    """
    global _llm_client
    if _llm_client is None:
        with _llm_lock:
            if _llm_client is None:
                _llm_client = LLMClient()
    return _llm_client
