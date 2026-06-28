# -*- coding: utf-8 -*-
"""
Stage 抽象基类 — 智能管道的最小处理单元

每个 Stage 代表管道中的一个处理步骤（FAQ 检查、意图分类、KG 推理等）。
Stage 接口统一为 execute(ctx) -> StageResult，使得:
  - 每个 Stage 可独立单元测试（构造 PipelineContext → 调用 execute → 验证结果）
  - Stage 可组合为不同管道（complexity=1 vs complexity=2 跑不同 Stage 组合）
  - 新增 Stage 不影响现有 Stage（开闭原则）

can_execute() 实现复杂度门控:
  - KGStage 仅在 complexity>=2 时执行
  - DomainGuardStage 仅在 complexity>=1 时执行
  - FAQStage 始终可执行（由管道路由决定是否调用）
"""

from abc import ABC, abstractmethod
from rag_qa.core.pipeline.context import PipelineContext, StageResult


class Stage(ABC):
    """
    处理阶段抽象基类

    子类必须实现:
      - name: 阶段名称（用于日志和 pipeline timing）
      - execute(): 核心处理逻辑

    子类可选覆盖:
      - can_execute(): 复杂度门控（默认始终可执行）
    """

    name: str = ""

    @abstractmethod
    def execute(self, ctx: PipelineContext) -> StageResult:
        """
        执行本阶段的处理逻辑

        从 ctx 读取所需数据，处理后将结果写回 ctx。
        返回 StageResult 表示执行状态。

        Args:
            ctx: 管道上下文（读写）

        Returns:
            StageResult: 执行结果状态
        """
        ...

    def can_execute(self, ctx: PipelineContext) -> bool:
        """
        判断本阶段是否应该执行

        用于复杂度门控——简单查询跳过重阶段（KG 推理等）。

        Args:
            ctx: 管道上下文（只读）

        Returns:
            True: 应该执行；False: 跳过本阶段
        """
        return True
