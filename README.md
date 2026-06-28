# 保险聚合平台 — RAG 智能问答 + 多智能体编排 · 双通道架构

面向保险聚合场景（对接多家保司多个产品）的 AI 助手系统。支持**内外双通道**：
- **Customer 通道**：外部客户自助问答（RAG Pipeline）— 轻量、快速、严格合规
- **Agent 通道**：内部保险顾问 AI 助手（Multi-Agent 编排）— 完整数据、专业决策

从 RAG 问答到多智能体编排，历经三次架构迭代，形成一套生产就绪的 AI 服务框架。对标明亚 MyBA、人保 AI 保宝等行业主流双通道架构。

---

## 核心亮点

|  |  |  |
|---|---|---|
| **双通道架构** | **14 阶段 RAG Pipeline** | **多智能体编排** |
| 同一 AI 底座，按角色分流：外部客户走轻量 RAG，内部顾问走 Multi-Agent | BGE-M3 混合检索 + Reranker 重排序 + 质量门控 + 合规审查 | Orchestrator 统一调度 4 个领域 Agent |
| **双模型故障转移** | **角色感知合规** | **10 种设计模式** |
| DeepSeek 主 + 通义千问降级 + Circuit Breaker | Customer 严格合规 + 免责声明；Agent 宽松合规 | Singleton、Strategy、Template Method、Chain of Responsibility 等 |

---

## 技术栈

| 类别 | 技术 |
|---|---|
| **LLM** | DeepSeek-V3（主） / 通义千问（降级） / DeepSeek-R1（Agent 规划） |
| **Embedding** | BGE-M3（本地，Dense 1024d + Sparse 双向量） |
| **Reranker** | bge-reranker-large（本地 Cross-Encoder） |
| **意图分类** | BERT-base-Chinese（9 分类含 OOD）+ 规则兜底 |
| **向量库** | Milvus（IVF_FLAT + 稀疏倒排索引，RRF 融合） |
| **关系库** | MySQL（产品/FAQ/保单缓存/费率表） |
| **缓存** | Redis（FAQ/Embedding/查询/产品四级缓存） |
| **Agent** | LangGraph StateGraph + LangChain Tools |
| **API** | FastAPI + SSE 流式推送 |
| **测试** | pytest + pytest-asyncio |

---

## 架构演进

```
Phase 1 ── 单 Agent 系统
  ├── LangGraph Planner-Executor 单 Agent
  ├── 5 轮 Plan-Execute 循环 + Reflector
  └── 7 个 LangChain Tool 任意调用

Phase 2 ── RAG 智能客服
  ├── 14 阶段串行 Pipeline
  ├── 6 种检索策略（DIRECT / HYDE / SUB_QUERY / COMPARE / CONDITIONAL / FALLBACK）
  ├── Parent-Child 分块（子块检索 + 父块增强）
  └── 质量门控 + 合规审查

Phase 3 ── 多 Agent 编排
  ├── Orchestrator 统一路由与任务编排
  ├── 4 个领域 SubAgent（保险/核保/理赔/客服）
  ├── 每个 Agent 独立 LangGraph、Tools、Compliance Rules
  └── 依赖式多 Agent 协作（上游结果注入下游 Context）

Phase 4 ── 双通道架构（当前）
  ├── Customer 通道: RAG Pipeline → 轻量快速、仅看自己数据、严格合规
  ├── Agent 通道: Multi-Agent → 完整编排、全量数据、角色感知工具集
  └── 共享 AI 底座: Milvus · MySQL · Redis · 双 LLM 故障转移
```

---

## RAG Pipeline

```
用户查询
  │
  ├─ ① FAQ 缓存命中 ─────────── 直接返回（<1ms）
  │
  ├─ ② PII 脱敏 ────────────── 6 类敏感信息正则 + NER 替换
  ├─ ③ 领域守卫 ────────────── 关键词白名单 + 黑名单过滤
  ├─ ④ 意图分类 ────────────── 规则（60%）→ BERT（35%）→ LLM（5%）
  ├─ ⑤ 查询缓存 ────────────── 归一化查询 + 意图维度的 Redis 缓存
  ├─ ⑥ 闲聊/投诉旁路
  │
  ├─ ⑧ 检索策略选择 ─────────── 6 种策略动态决定
  ├─ ⑨ Milvus 混合检索 ──────── Dense + Sparse 双路召回，RRF 融合
  ├─ ⑩ Parent-Child 增强 ────── 子块匹配 → 父块上下文回填
  ├─ ⑪ Reranker 重排序 ──────── bge-reranker-large，Top-30 → Top-5
  ├─ ⑫ 质量门控 ────────────── 相似度阈值 + 块数 + 长度检查
  │
  ├─ ⑬ LLM 生成 ────────────── DeepSeek → 通义千问故障转移
  ├─ ⑭ 合规审查 ────────────── 医疗建议/敏感词/竞品贬低/金额验证
  │
  └─ ⑮ PII 恢复 ────────────── 占位符回填原始数据
       │
       └─ 返回答案 + 来源 + 各阶段耗时
```

每个阶段记录独立耗时，支持精细化性能监控。任何异常由顶层兜底捕获，永不暴露原始错误给用户。

---

## 多智能体系统

### Orchestrator 流程

```
用户查询 → 路由意图 → 构建任务 → 执行任务 → 聚合结果 → 合规审查 → 返回
                │            │           │
                │            │           └─ 单 Agent：直接返回
                │            │              多 Agent：LLM 合并结果
                │            │
                │            └─ 依次执行，依赖的上游 Agent 结果注入下游 Context
                │
                └─ 12 种意图 → 4 个 Agent + RAG 兜底
```

### 4 个领域 SubAgent

| Agent | 模型 | Tools | 职责 |
|---|---|---|---|
| **保险顾问** | DeepSeek-V3 | 产品对比、保费计算、条款搜索、人工转接 | 产品推荐、投保指导 |
| **核保助手** | DeepSeek-R1 | 条款搜索、人工转接 | 健康核保、风险评估 |
| **理赔专员** | DeepSeek-R1 | 保单查询、条款搜索、理赔资格、进度追踪、保费计算 | 理赔核定、赔付估算 |
| **客服管家** | DeepSeek-V3 | 保单查询、人工转接 | 保单查询、FAQ、转人工 |

每个 Agent 继承自 `SubAgent` 抽象基类，模板方法提供统一接口：

```
SubAgent.invoke(task) → SubAgentResult
  ├── _get_system_prompt()  ── 领域定制 Prompt
  ├── _get_tools()          ── 领域过滤 Tool 子集
  ├── _get_compliance_rules() ── 领域合规规则
  └── LangGraph Plan → Exec → Check (循环至多 3 次) → Synthesize
```

### 7 个 Tool 一览

| Tool | 数据源 | 功能 |
|---|---|---|
| PolicyQueryTool | MySQL | 保单信息查询 |
| ClaimEligibilityTool | 向量库 + 规则引擎 | 理赔资格判定 |
| ClauseSearchTool | 向量库 | 条款混合检索 |
| PremiumCalcTool | MySQL | 费率表计算 |
| ProductCompareTool | 向量库 + LLM | 多产品对比 |
| ClaimTrackingTool | MySQL | 理赔进度查询 |
| HumanHandoffTool | MySQL | 人工转接记录 |

---

## 项目结构

```
Insurance/
├── config/           # 配置管理（类型化 dataclass）
├── base/             # 基础设施：LLM 客户端 / BGE-M3 / Reranker / 数据库会话 / 日志
├── cache/            # 三级缓存：Redis + Milvus + MySQL，含 Guard 防护
├── rag_qa/           # RAG 智能问答核心
│   ├── core/         #   Pipeline 14 阶段：脱敏 / 守卫 / 分类 / 检索 / 门控 / 合规
│   └── ingestion/    #   PDF 摄取：解析 → 分块 → 编码 → 入库
├── agent/            # 多智能体编排
│   ├── tools/        #   7 个 LangChain Tool
│   ├── sub_agents/   #   4 个领域 SubAgent 实现
│   └── orchestrator.py  # 统一编排引擎
├── gateway/          # FastAPI API 层（/chat、SSE、/admin/ingest、/health）
└── tests/            # pytest 测试（多 Agent / 缓存 / 数据库 / LLM）
```

---

## 工程亮点

| 模式/策略 | 说明 |
|---|---|
| **Singleton** | LLM 客户端、Encoder、Reranker、Redis 客户端全局唯一实例 |
| **Template Method** | `SubAgent` 抽象基类定义 invoke 骨架，子类实现领域定制 |
| **Strategy** | `StrategySelector` 根据意图和查询特征动态切换 6 种检索策略 |
| **Chain of Responsibility** | RAG Pipeline 14 阶段串行处理，每阶段独立可测 |
| **Orchestrator** | 集中编排多 Agent，不直接交互，保持每个 Agent 无状态可独立测试 |
| **Circuit Breaker** | LLM 连续 5 次失败 → 30s 冷却 → 自动切换降级模型 |
| **Cache-Aside** | 缓存优先读，缺失则查库回填，随机 TTL 抖动防止雪崩 |
| **分布式互斥锁** | SET NX 实现缓存重建锁，单请求回源，其余等待 50ms |
| **Parent-Child 分块** | 子块（512 字符）精确检索 + 父块（2000 字符）上下文增强 |
| **优雅降级** | 任何组件故障不影响整体服务：LLM 故障 → 降级模型 → "暂时繁忙"返回 |
| **全链路监控** | Pipeline 每阶段记录 ms 级耗时，随响应返回 |
| **文档版本化** | 产品更新时旧块标记 is_valid=false，不物理删除 |

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 复制并修改配置
cp config/.env.example config/.env
# 编辑 .env 填入 Milvus/MySQL/Redis/DeepSeek 等配置

# 3. 启动服务
uvicorn gateway.app:app --host 0.0.0.0 --port 8000

# 4. 运行测试
pytest tests/ -v

# 5. 文档导入（可选）
curl -X POST http://localhost:8000/admin/ingest \
  -H "Content-Type: application/json" \
  -d '{"pdf_paths": ["./docs/示例条款.pdf"], "insurer": "示例保险", "product_name": "示例产品"}'
```

---

## API 接口

| 方法 | 路径 | 说明 | 权限 |
|---|---|---|---|
| POST | `/chat` | 统一问答入口，按 X-User-Role 自动分流 | Customer / Agent |
| GET | `/chat/stream` | SSE 流式推送（Customer 通道追加免责声明） | Customer / Agent |
| POST | `/admin/ingest` | PDF 文档导入 | **仅 ADMIN** |
| GET | `/health` | 健康检查 | 公开 |

### 双通道 Headers

| Header | 说明 | 示例 |
|---|---|---|
| `X-User-Id` | 用户唯一标识 | `cust_001` / `agent_042` |
| `X-User-Role` | 角色 | `customer` / `agent` / `underwriter` / `admin` |
| `X-Org-Id` | 所属机构（可选） | `pingan` / `zhongan` |

### 双通道架构图

```
         X-User-Id / X-User-Role（上游 API Gateway 注入）
                          │
               ┌──────────▼──────────┐
               │   Auth Middleware     │
               │   get_current_user()  │
               └──────────┬──────────┘
                          │
         ┌────────────────┼────────────────┐
         │                                 │
┌────────▼────────┐              ┌────────▼────────┐
│ CUSTOMER 通道    │              │ AGENT 通道       │
│ role=customer    │              │ role=agent/uw    │
├─────────────────┤              ├─────────────────┤
│ · RAG Pipeline  │              │ · Multi-Agent    │
│ · 只看自己保单   │              │ · 查询任意客户   │
│ · 严格合规+免责  │              │ · 宽松合规       │
│ · 通俗回答       │              │ · 完整数据       │
│ · 无内部工具     │              │ · 全量工具集     │
└────────┬────────┘              └────────┬────────┘
         │                                 │
         └────────────────┬────────────────┘
                          │
               ┌──────────▼──────────┐
               │  共享 AI 基础设施    │
               │  Milvus · MySQL ·    │
               │  Redis · LLM Pool    │
               └─────────────────────┘
```

---
