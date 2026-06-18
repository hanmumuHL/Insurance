# 保险聚合平台 RAG+Agent 项目 — 面试复习指南

> 生成时间: 2026-06-03
> 项目路径: ~/code/code/pythonCode/Insurance/

---

## 目录

1. [核心数字速查](#一核心数字速查)
2. [项目全景](#二项目全景)
3. [基础设施层](#三基础设施层)
4. [基础服务层 (base/)](#四基础服务层-base)
5. [缓存层 (cache/)](#五缓存层-cache)
6. [RAG 核心 (rag_qa/)](#六rag-核心-rag_qa)
7. [多 Agent 架构 (agent/)](#七多-agent-架构-agent)
8. [网关与测试](#八网关与测试)
9. [分阶段应答策略](#九分阶段应答策略)
10. [追问预判表](#十追问预判表)
11. [高频技术问题速答](#十一高频技术问题速答)

---

## 一、核心数字速查

```
8 家保险公司        200+ 产品           7 个工具
4 个领域 Agent      2 个推理模型(R1)    2 个检索模型(V3)
14 阶段 RAG 管线    6 种检索策略        12 个 Milvus 标量字段
~350ms 总延迟       22 个集成测试       5 道合规检查
3 级缓存            5 次熔断阈值        30 秒熔断冷却
50+ Python 文件     7000+ 行代码
```

---

## 二、项目全景

### 业务定位

保险聚合平台智能客服，对接 8 家主流保险公司（平安、众安、太平洋、人保等）、200+ 产品。用户一站式完成 **投保咨询 → 核保审核 → 理赔指导 → 保单查询**。

### 技术栈总览

```
应用层   FastAPI (gateway/app.py)
        LangGraph (Agent 状态图)
模型层   DeepSeek-V3 (日常对话/轻推理)
        DeepSeek-R1 (核保/理赔深度推理)
        千问 (降级通道)
        BGE-M3 (Dense+Sparse 双向量编码)
        BGE-Reranker (Cross-Encoder 精排)
存储层   Milvus 2.4.4 (向量库, 19530端口)
        MySQL 8.4 (持久层, 3306端口)
        Redis 7.x (缓存, 6379端口)
        MinIO (对象存储, 9000端口)
部署     Docker Compose (~/milvus_redis)
        WSL systemd (MySQL)
```

### 项目演进

```
Phase 1  RAG 智能客服     Milvus + BGE-M3 + DeepSeek 纯检索增强
Phase 2  Agent 智能体      LangGraph Plan-Execute + 7工具
Phase 3  多 Agent 协作     Orchestrator + 4领域子Agent (投保/核保/理赔/客服)
```

### 请求全链路

```
用户 query
→ PII 脱敏 → 领域守卫 → 三层意图路由 → 6种检索策略自适应
→ Milvus 混合检索 (Dense语义 + Sparse关键词 + RRF融合)
→ Reranker 精排 (Top-30→Top-5) → 质量检查 → LLM生成 → 合规检查
→ PII还原 → 缓存写入 → 返回答案 (总耗时 <350ms)
```

---

## 三、基础设施层

### Docker Compose (~/milvus_redis/docker-compose.yml)

| 容器 | 端口 | 职责 | 依赖 |
|------|------|------|------|
| milvus-etcd | 2379 | Milvus 元数据 | — |
| milvus-minio | 9000 | Milvus 对象存储 | — |
| milvus-standalone | 19530,9091 | 向量引擎 | etcd+minio |
| milvus-redis | 6379 | 缓存 | — |

### MySQL 8.4 — 8 张业务表

WSL 原生 systemd 运行。数据库 `insurance_platform`：

```
documents       — PDF 文档元数据
document_chunks — chunk 级映射
faq_questions   — FAQ 问答对
products        — 产品注册信息
policy_cache    — 保单缓存(脱敏)
rate_table      — 费率表
claim_records   — 理赔记录
handoff_requests — 人工转接请求
```

---

## 四、基础服务层 (base/)

### LLM 双通道 + 熔断器 (llm_client.py, 482行)

```
主通道: DeepSeek-V3 (deepseek-chat)  — 性价比高，中文能力强
降级通道: 通义千问 (qwen-max)        — 阿里云生态，稳定性好

调用策略:
  DeepSeek 成功 → 返回
  DeepSeek 失败 → 千问
  千问也失败 → 重试 DeepSeek 一次
  全部失败 → 兜底话术 "系统繁忙"

熔断机制:
  连续 5 次失败 → 熔断 30 秒
  熔断期间直接走千问，不等超时
  冷却期过 → 半开试探 → 成功后自动恢复
```

### BGE-M3 双向量编码 (encoder.py)

```
输入文本 → BGE-M3 模型
  ├── Dense  (1024维): 语义相似度  "肺炎"↔"肺部感染"
  └── Sparse (词汇级): 关键词精确匹配 "免赔额"↔条款原文

性能: GPU ~5ms, CPU ~50ms
```

### Reranker 精排 (reranker.py)

```
Cross-Encoder 架构:
  query + chunk 拼成一条序列 → 逐 token 计算匹配度
  精度远超 Bi-Encoder 余弦相似度

管线: Milvus 粗排 Top-30 → Reranker 精排 → Top-5
精度提升: +10-15%，代价: 慢 50-100 倍 (只在 T30 上做)
```

---

## 五、缓存层 (cache/)

### 三级缓存

```
L0: FAQ 精确匹配 (1h)
  query 标准化 → Redis → 命中秒回
  命中 → ZINCRBY 热度统计
  未命中 → 查 MySQL FAQ 表 → 回填 Redis

L1: Query 结果缓存 (10min)
  query+intent → MD5 → Redis → 完整回答
  同一问题高峰期秒回

L2: Embedding 缓存 (24h)
  query → MD5 → Redis 存 float32 字节
  命中 → 跳过 BGE-M3 编码 (~50ms)
```

### 三大防护 (cache_guard.py)

| 问题 | 原因 | 方案 |
|------|------|------|
| 穿透 | 恶意查询不存在的 key | 空值也缓存 (null, TTL=60s) |
| 击穿 | 热点 key 过期瞬间大量请求 | SET NX 互斥锁，只让一个请求查 DB |
| 雪崩 | 大量 key 同时过期 | 随机 TTL ±20% |

---

## 六、RAG 核心 (rag_qa/)

### 14 阶段管线

```
阶段 1:  FAQ 缓存检查         → 命中直接返回 (<1ms)
阶段 2:  PII 脱敏             → 6类敏感信息正则检测 + 姓名上下文匹配
阶段 3:  领域边界守卫          → 白名单60+词/黑名单10词，0ms拦截 ~70%
阶段 4:  三层意图路由          → 规则(60%)→BERT(35%)→LLM(5%)
阶段 5:  意图拒绝判断          → 9分类，差异化置信度阈值
阶段 6:  Query 结果缓存        → 相同 query+intent 秒回
阶段 7-8: 闲聊/投诉快捷路径     → 直接 LLM 回复或转人工
阶段 9:  检索策略选择          → 6种策略自适应
阶段 10: Milvus 混合检索       → Dense+Sparse+标量过滤+RRF融合 (~30ms)
阶段 11: Reranker 精排         → Top-30→Top-5 (~20ms)
阶段 12: 检索质量检查          → Top-1/Top-5平均/高相似度数量 3项检查
阶段 13: LLM 生成答案          → DeepSeek 主 + 千问降级 (~200ms)
阶段 14: 合规检查              → 5道检查 (医疗建议/监管词/贬低/金额引用/(预留))
阶段 15: PII 还原 + 写入缓存   → 占位符还原真实值
```

### 6 种检索策略

| 策略 | 触发条件 | 检索方式 |
|------|---------|---------|
| direct | 默认 | 语义向量 |
| hyde | query<8字且无实体 | LLM先生成假设文档再检索 |
| sub_query | ≥2个产品名 | 子查询拆分并行检索 |
| compare | 产品对比意图 | 多路并行+Rerank交叉对比 |
| conditional | 保费试算意图 | MySQL结构化查询 |
| fallback | Top-K不足 | 放宽过滤条件二次检索 |

### 合规守卫 — 5 道检查

```
1. 医疗建议拦截    "医生"+"建议"同时出现 → 拦截 (引用条款除外)
2. 监管敏感词       "保证理赔""一定能赔""保本保息" → 删除
3. 竞品贬低         保司名+不好/坑/骗/垃圾 → 替换为中立表述
4. 金额引用         涉及金额但无条款引用 → 追加免责说明
5. (预留)

修复策略: 自动修复而非简单拒绝
  - 医疗建议 → 追加 "⚠️ 不构成医疗建议"
  - 贬低竞品 → 替换为 "不同产品适合不同需求"
  - 金额无引用 → 追加 "📋 以合同条款约定为准"
```

---

## 七、多 Agent 架构 (agent/)

### 架构全景

```
POST /chat
    │
    ▼
Orchestrator (agent/orchestrator.py, 724行)
    │ 1. 查路由表 (INTENT_ROUTING, 12条规则)
    │ 2. 降级判断 (短query/投诉 → RAG)
    │ 3. 任务拆解 (single_agent / multi_agent)
    │ 4. 按序执行 (上游结果 → 下游 context)
    │ 5. 结果聚合 (单Agent直返 / 多Agent LLM整合)
    │ 6. 合规终审 (确定性承诺替换)
    │
    ├── InsuranceAgent     投保 (v3) 4工具: product_compare, premium_calc, clause_search, human_handoff
    ├── UnderwritingAgent  核保 (r1) 2工具: clause_search, human_handoff
    ├── ClaimAgent         理赔 (r1) 6工具: 全部
    └── ServiceAgent       客服 (v3) 2工具: policy_query, human_handoff
```

### 通信模式

子 Agent 之间**不直接通信**。Orchestrator 中心化调度：

```
task_0 (主Agent) → 执行 → 结果
                              ↓
task_1 (辅助Agent) context["upstream_result"] = task_0 的结果
                              ↓
task_1 基于上游结果做推理 → 结果
                              ↓
Orchestrator LLM 聚合两个结果 → 合规终审 → 最终答案
```

### 意图路由表 (精简)

| 意图 | 主 Agent | 辅助 | 模式 |
|------|---------|------|------|
| 产品咨询/对比/保费 | insurance | — | single |
| 投保流程 | insurance | underwriting | multi |
| 核保咨询 | underwriting | — | single |
| 理赔咨询 | claim | underwriting | multi |
| 理赔进度 | claim | service | multi |
| 保单查询 | service | — | single |
| 投诉/闲聊 | service | — | rag_fallback (降级) |

### 7 个工具

记法：7 个动词 = 用户旅程

```
查: policy_query      保单查询       MySQL
搜: clause_search     条款检索       Milvus
算: premium_calc      保费试算       MySQL
比: product_compare   产品对比       Milvus+LLM
判: claim_eligibility 理赔资格预检   MySQL+Milvus
追: claim_tracking    理赔进度       MySQL
转: human_handoff     转人工        MySQL+Redis
```

### 模型选型

```
V3 (快速): InsuranceAgent, ServiceAgent
  → 信息检索和对比展示，不需要推理链

R1 (推理): UnderwritingAgent, ClaimAgent
  → 核保: 健康告知→规则匹配→风险评估→结论 (4步推理)
  → 理赔: 查保单→查条款→等待期→免责→赔付 (5步推理)
```

### RAG 与 Agent 的结合

```
RAG 是底座，Agent 是上层调度。不是二选一。

具体结合:
  - Agent 的 clause_search 底层走 BGE-M3+Milvus+RRF (和纯RAG相同)
  - gateway 先用 RAG 的 QueryClassifier 拿 intent，再交给 Orchestrator
  - 闲聊/投诉/Agent失败 → 自动降级纯 RAG
  - Agent 和 RAG 共用同一套 VectorStore / MySQL / Redis 实例
```

### LangGraph 在哪

每个子 Agent 内部是 LangGraph 状态图：

```
plan → exec → check ─┬→ synthesize → END
    ↑                 │
    └── 3次迭代 ──────┘  (should_continue 条件边)

Orchestrator 做 Agent 间调度 (纯 Python)
LangGraph 做 Agent 内循环 (Conditional Edges)
```

---

## 八、网关与测试

### FastAPI 三端点

```
POST /chat            统一入口 → Orchestrator 自动路由
POST /chat/multi_agent 向后兼容别名
GET  /chat/stream      SSE流式问答 (逐句推送, 50ms间隔)
POST /admin/ingest     文档摄取
GET  /health           健康检查
```

### 4 个测试文件

```
test_mysql.py        585行  数据库连接+建表+CRUD+并发
test_redis_milvus.py 498行  Redis读写/过期+Milvus检索/过滤
test_llm_client.py   127行  DeepSeek API+降级+JSON输出
test_multi_agent.py  420行  6类测试 (路由/任务/执行/聚合/合规/E2E)
```

---

## 九、分阶段应答策略

### 开场介绍 (90秒)

> 这是我做的一个保险聚合平台智能客服，对接 8 家保司 200+ 产品。核心链路是：用户问题先走 14 阶段的 RAG 管线——PII 脱敏、三层意图路由、6 种检索策略自适应选择、Milvus 混合检索、Cross-Encoder 精排、合规检查后生成答案。复杂场景做了多 Agent 架构——Orchestrator 调度 4 个领域 Agent，理赔场景下理赔 Agent 和核保 Agent 串行协作。Agent 的每个工具底层复用同一套 RAG 基础设施，不是两套系统。

### "为什么拆成 4 个 Agent"

> 三个触发标准。第一，7 个工具全量给一个 Agent，Planner 选择范围太大不稳定。拆领域后每个 Agent 只 2-6 个工具。第二，知识域冲突——投保 Agent 的 Prompt 要"推荐产品"，核保 Agent 要"审慎评估"，塞一个 Prompt 里互相打架。第三，独立迭代——理赔规则随监管频繁更新，不能影响客服的稳定性。

### "Agent 体现在哪里"

> 三个维度。自主规划——Planner 自动决定调哪些工具、什么顺序。容错自愈——失败 Re-plan 重试、降级部分工具、全失败走 RAG 兜底。状态记忆——多轮对话自动加载用户已购产品和理赔记录。

### "RAG 和 Agent 怎么结合"

> RAG 是底座，Agent 是上层调度。Agent 的 clause_search 底层走和纯 RAG 相同的 BGE-M3+Milvus+RRF 管线。路由层面，闲聊直接走 RAG 降级。Agent 初始化失败自动退化纯 RAG——服务不中断。

### "最大难点"

> 双通道 LLM 的熔断设计。DeepSeek 偶尔超时 30 秒，不等就熔断直走千问。35 行自研 CircuitBreaker，但正常→熔断→半开三状态边界测试了很多轮。另一个是多 Agent 上游结果注入协议——哪些信息传递、格式怎么定义，迭代了三版。

---

## 十、追问预判表

| 你说 | 面试官追问 | 回答方向 |
|------|-----------|---------|
| "8 家保险公司" | 多保司数据怎么处理 | 12 个标量字段 + expr 动态拼接多维过滤 |
| "混合检索" | Dense 和 Sparse 怎么融合 | RRF 排名融合，不看分数看排名，天然归一化 |
| "合规守卫" | 具体检查什么 | 5 道：医疗建议/监管词/贬低竞品/金额引用/(预留) |
| "意图路由三层" | 为什么不用纯 LLM | 成本：规则 0ms 命中 60%，LLM 200ms 才兜底 5% |
| "模型选 R1" | R1 和 V3 什么区别 | R1 有思维链适合多步推理，V3 快适合检索生成 |
| "缓存三级" | 防击穿怎么做的 | SET NX 互斥锁，只让一个请求查 DB，其余等 50ms 读缓存 |
| "工具 7 个" | 怎么选型不用 LangChain | 依赖注入模式，测试可 mock，不需要 AgentExecutor 黑盒 |
| "PII 脱敏" | 脱敏后怎么还原 | mapping 字典保存，LLM 返回后 restore() 还原。LLM API 只看到占位符 |
| "子 Agent 通信" | 为什么不直接通信 | 中心化调度，加 Agent 不影响其他，协议简单 |
| "6 种检索策略" | fallback 怎么触发 | Top-K 不足→放宽过滤→二次检索→融合去重 |

---

## 十一、高频技术问题速答

### BGE-M3 为什么选它？

同时输出 Dense(语义) + Sparse(关键词) 双向量。Dense 捕捉 "肺炎↔肺部感染" 的同义关系，Sparse 做 "免赔额" 精确关键词匹配。两路 RRF 融合，召回率高于纯 Dense。

### 为什么不用 LangChain 的 AgentExecutor？

while 循环对简单任务够用，但保险场景需要结构化中间结果传递和精细降级。自研 260 行 LLMClient 比 LangChain 的 30+ 文件轻量可控。

### 为什么 Cross-Encoder 只对 Top-30 做？

Cross-Encoder 逐 token 计算匹配度，精度高但慢 50-100 倍。全量做延迟不可接受。Top-30 是工程上的甜点——兼顾精度和速度。

### 怎么保证 Agent 挂了系统不崩？

三层：gateway 懒加载 Agent 失败 → 降级纯 RAG；单次工具调用失败 → Re-plan 重试；全部失败 → 兜底话术 "联系人工客服"。

### SQL 注入怎么防？

policy_query 工具使用参数化查询 `%s` 占位符，不用字符串拼接。Milvus 的 expr 过滤也有字段名校验。

### 产品下架怎么处理？

`vector_store.invalidate_by_product()` 标记旧版本 `is_valid=False`，检索时 `filter='is_valid==true'` 自动过滤。

---

> 结尾钩子 (每个回答后留一个):
> "Retrieval 这块可以展开讲下为什么不用纯关键字"
> "合规层是国内保险场景独有的挑战，和通用 RAG 很不一样"
> "多 Agent 协议设计踩了不少坑，可以仔细聊聊"

---

*全部 9 步梳理完毕。项目路径: ~/code/code/pythonCode/Insurance/*
