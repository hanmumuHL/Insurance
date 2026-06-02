# -*- coding: utf-8 -*-
"""
Agent 工具集 — 7 个 LangChain Tool

每个工具继承 LangChain 的 BaseTool，实现 _run 方法。
Planner 通过工具名 + 参数调用工具，Executor 负责执行。

7 个工具:
  1. policy_query      — 保单查询
  2. claim_eligibility — 理赔资格预检
  3. clause_search     — 条款检索 (复用 RAG 的 Milvus 检索)
  4. premium_calc      — 保费试算
  5. product_compare   — 多产品对比
  6. claim_tracking    — 理赔进度追踪
  7. human_handoff     — 人工转接
"""
