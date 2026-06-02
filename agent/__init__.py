# -*- coding: utf-8 -*-
"""
Agent 智能体系统 — LangGraph Planner-Executor 架构

核心设计:
  Agent 不是一股脑把所有工具调一遍，而是先规划再执行:
    Planner: LLM 分析用户意图，拆解子任务，选择工具
    Executor: 按规划执行工具调用，收集结果
    Reflector: 检查结果是否完整，不完整则回到 Planner 补充
    Synthesizer: 整合所有工具结果，生成最终答案

为什么用 LangGraph 而不是简单的 LangChain Agent:
  1. LangChain Agent 是 ReAct 循环（想一步做一步），不适合复杂多步任务
  2. LangGraph 是状态图，可以显式定义 plan → exec → reflect → synthesize 的流程
  3. LangGraph 支持 Checkpoint（中断恢复），用户追加信息时不用从头开始
"""
