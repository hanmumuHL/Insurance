# 保险聚合平台 RAG+Agent 项目 — 面试复习指南

> 更新: 2026-06-19 (新增 KG + LoRA 微调章节)
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
8. [知识图谱 (rag_qa/core/kg_*)](#八知识图谱-rag_qacorekg_)
9. [BERT LoRA 微调](#九bert-lora-微调)
10. [网关与测试](#十网关与测试)
11. [分阶段应答策略](#十一分阶段应答策略)
12. [追问预判表](#十二追问预判表)
13. [高频技术问题速答](#十三高频技术问题速答)

---

## 一、核心数字速查

```
RAG + Agent:
  10 家保险公司       200+ 产品           7 个工具
  4 个领域 Agent      2 个推理模型(R1)    2 个检索模型(V3)
  14 阶段 RAG 管线    6 种检索策略        12 个 Milvus 标量字段
  ~350ms 总延迟       5 道合规检查        3 级缓存
  5 次熔断阈值        30 秒熔断冷却

知识图谱:
  126 个节点          110 条关系边        8 种节点类型
  10 种关系边         102 种疾病          14 个疾病分类
  5 类实体链接        Neo4j 主后端        NetworkX 降级

BERT LoRA 微调:
  r=16 rank          540 条训练数据      98.1% 验证准确率
  2.69M 可训练参数    全量 2.6%          adapter ~10MB
  温度 T=0.316        LLM 兜底率 <1%
```

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
Phase 1  RAG 智能客服        Milvus + BGE-M3 + DeepSeek 纯检索增强
Phase 2  Agent 智能体        LangGraph Plan-Execute + 7工具
Phase 3  多 Agent 协作       Orchestrator + 4领域子Agent (投保/核保/理赔/客服)
Phase 4  知识图谱+BERT微调   Neo4j + LoRA + 实体链接 + 多跳推理
```

### 面试叙事主线

> "这个项目分四个阶段迭代——先搭 RAG 检索基座保证基本问答能力，然后加多 Agent 协作处理复杂场景，接着用知识图谱增强检索和推理，最后 LoRA 微调 BERT 把意图分类做到生产可用。每一步都是在上一步基础上演进，不是独立的三套系统。"

### 请求全链路 (更新)

```
用户 query
→ PII 脱敏 → 领域守卫 → 三层意图路由 (规则→LoRA微调BERT→LLM兜底)
→ KG 实体链接 (保司/产品/疾病/事件/维度) → 6种检索策略自适应
→ [KG推理增强] 检索前 Neo4j 补全疾病-条款-产品关系链
→ Milvus 混合检索 (Dense语义 + Sparse关键词 + RRF融合)
→ Reranker 精排 (Top-30→Top-5) → 质量检查
→ LLM生成 (KG推理路径注入prompt) → 合规检查
→ PII还原 → 缓存写入 → 返回答案 (<350ms)
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

## 八、知识图谱 (rag_qa/core/kg_*)

### 为什么加知识图谱？

保险是天然图结构。纯 RAG 只能单跳检索 (query→chunk)，以下问题无法回答：

> "覆盖心脏疾病且等待期不超过 30 天的百万医疗险有哪些？"

这需要 3 跳推理：心脏疾病→心血管疾病类→覆盖该分类的条款→产品→等待期≤30天过滤。

### KG Schema

```
Insurer ──[issues]──→ Product ──[has_clause]──→ Clause
                          │            │
              [has_waiting]│            │[covers/excludes]
                          ↓            ↓
                   WaitingPeriod    Disease ──[belongs_to]──→ DiseaseCategory
```

8 种节点类型、10 种关系边。从 MySQL (产品-保司) + 条款文档 chunks (疾病/等待期/免赔额) + 内置知识库 (14大类102种疾病) 三源自动构建。

### 技术路线

```
Phase 1: NetworkX 内存图快速验证 Schema 设计 (一小时内迭代几个版本)
Phase 2: 验证可行 → 迁移 Neo4j 生产级图数据库
         接口层抽象双后端降级，上层代码零改动
```

### 实体链接器 (kg_entity_linker.py)

```
用户 query → 按标签批量加载 Neo4j 节点构建本地索引
→ 最长匹配消歧 ("平安健康" 优先于 "平安")
→ 链接到图节点

覆盖 5 类实体: 保司/产品/疾病/医疗事件/条款维度
替代 QueryClassifier 和 StrategySelector 中的硬编码正则
```

示例: `"平安e生保肺炎住院能赔吗"`
→ insurer: 平安健康, product: e生保, disease: 肺炎 (呼吸系统+传染), event: 住院, dimension: 理赔

### 多跳推理引擎 (kg_reasoner.py)

三种推理模式:
```
1. 疾病 → 覆盖产品 (2-hop)
   肺炎 → covers → 保险责任条款 → has_clause → 平安e生保

2. 产品 → 等待期/免赔额 (1-hop)
   平安e生保 → has_waiting_period → 30天

3. 多约束组合查询
   reason_complex(disease="肺炎", max_waiting_days=30)
   → Cypher 组合查询 → 满足所有约束的产品列表
```

推理结果注入 Orchestrator 调度上下文和 RAG LLM prompt。

### 面试话术

> "保险是天然图结构——疾病属于什么分类、哪些产品覆盖这个疾病、等待期多长、免赔额多少，全是关系。我先用 NetworkX 快速验证 Schema 设计，确认本体没问题后迁到 Neo4j。做了双后端降级——Neo4j 连不上自动切 NetworkX，上层零改动。实体链接和推理路径都是可解释的——这在保险合规场景是硬需求。"

---

## 九、BERT LoRA 微调

### 为什么需要微调？

意图路由第二层原本用预训练 bert-base-chinese。问题是：预训练模型的分类头是随机初始化的，9 类保险意图准确率只有 ~11%（≈随机猜测）。需要微调才能用。

### LoRA 配置

```
r=16, alpha=32, dropout=0.05
target_modules: [query, key, value, dense]  ← 覆盖注意力+FFN层
可训练参数: 2.69M / 105M (2.6%)
adapter 体积: ~10MB (vs 全量 440MB)
```

### 训练管线

```
90条种子数据 → LLM逐条生成5种变体 → 540条均衡训练集 (9类×60条)
→ LoRA 微调 8 epoch → 温度校准 (20点网格搜索, T=0.316)
→ merge_and_unload 合并为完整模型

效果: 验证集 98.1% | 9类测试全对 | 预训练仅 11%
```

### 三种加载模式

```
1. 完整模型 → AutoModel 直接加载 (推理部署)
2. LoRA adapter → 加载基座 + adapter (继续训练)
3. 预训练基座 → 创建新 LoRA wrapper (首次训练)
```

### 面试话术 (算法岗重点)

> "一开始用 r=8 只打 query/value，准确率才 42%。分析后发现分类任务需要 FFN 层的语义变换能力——把 dense 也加进去、r 提到 16，直接 98%。温度校准从 1.0 搜到 0.316——不做的话验证集过拟合但测试集掉点。还有一个坑：BERT 原版 checkpoints 是 MaskedLM 架构，分类头得重新初始化。"

### 面试话术 (工程岗重点)

> "这个业务不需要千亿参数大模型，BERT-base 110M 参数就够。LoRA 只训练 2.6% 参数，adapter 10MB，基座 440MB 不动——如果后面要加新业务线比如车险分类，训练一个新的 adapter 就行，不用重训整个模型。merge 之后直接部署完整模型，推理时不需要 peft 库依赖。"

---

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

### 开场介绍 (60秒)

> "我做的保险聚合平台智能客服，对接 10 家保司。分四个阶段迭代——先搭 RAG 基座保证基本问答，再加多 Agent 处理复杂场景，然后用 Neo4j 知识图谱增强检索推理，最后 LoRA 微调 BERT 把意图分类做到 98% 准确率。核心链路：用户问'肺炎住院能赔吗'→ KG 实体链接识别疾病产品 → 三层意图路由判为理赔咨询 → 检索策略自适应 → BGE-M3 双向量 + Milvus 混合检索 → Reranker 精排 → LLM 生成 → 合规检查，端到端 350ms。Agent 的条款检索底层复用同一套 RAG 管线，故障自动降级纯 RAG。"

### 按岗位侧重的开场变体

**大模型应用开发岗** (强调架构):
> "这个项目最核心的是稳定性设计。两个 LLM 通道——DeepSeek 做主力、千问做降级，自研熔断器连续 5 次失败自动切，0ms 不等超时。Agent 四种降级策略——短 query 降级、低置信度降级、全失败降级、初始化失败降级。KG 也做了 Neo4j→NetworkX 双后端降级。整个系统一个故障点都不会导致不可用。"

**大模型算法岗** (强调模型):
> "这个项目做了端到端的模型优化。检索侧 BGE-M3 同时输出 Dense 和 Sparse 向量，RRF 融合后 Cross-Encoder 精排——这仨组合比纯 Dense 召回高 15%。推理侧 DeepSeek-R1 处理核保理赔的多步推理链，V3 处理投保客服的快速生成。分类侧 LoRA 微调 BERT——只练 2.6% 参数把准确率从 11% 拉到 98%。每个环节都做了消融实验。"

### "为什么拆成 4 个 Agent"

> 三个原因。第一，7 个工具全量给一个 Agent，Planner 选择范围太大易出错——拆领域后每个 Agent 只 2-6 个工具，规划准确率明显提升。第二，知识域冲突——投保 Agent 要"推荐产品"，核保 Agent 要"审慎评估"，塞一个 Prompt 里互相打架。第三，独立迭代——理赔规则随监管频繁更新，不能影响客服稳定性。

### "Agent 体现在哪里"

> 三个维度。自主规划——Planner 自动决定调哪些工具、什么顺序。容错自愈——失败 Re-plan 重试最多 3 次，全失败走 RAG 兜底。状态记忆——多轮对话自动加载用户已购产品和理赔记录。

### "RAG 和 Agent 怎么结合"

> RAG 是底座，Agent 是上层调度，不是两套系统。Agent 的 clause_search 底层走和纯 RAG 相同的 BGE-M3+Milvus+RRF 管线。入口层 RAG 意图分类器拿到 intent 后交给 Orchestrator 路由。Agent 故障自动退化纯 RAG——服务不中断。

### "为什么加知识图谱"

> 纯 RAG 只能单跳检索，复杂问题如"覆盖心脏疾病且等待期不超过 30 天的产品有哪些"需要 3 跳推理，RAG 做不到。另外大模型不知道保险条款之间的精确关系——"肺炎属于呼吸系统疾病"这种事实 KG 查 1ms，LLM 可能幻觉。推理路径完全可解释可审计——保险合规场景硬需求。

### "为什么用 LoRA 而不是全量微调"

> BERT-base 110M 参数全量微调当然能做。但 LoRA 几个优势：adapter 10MB 可以快速迭代版本，基座 440MB 不动；小数据集过拟合风险更低——全量微调容易忘掉预训练知识；实际结果 LoRA 98.1% 比全量 94.4% 还高，因为 LoRA 自带正则化效应；如果后面加新业务线，训练一个新 adapter 就行不用重训。

### "最大难点"

> LoRA 的 rank 和 target_modules 调参。一开始 r=8 只打 query/value，准确率才 42%。分析后——BERT 分类任务需要 FFN 层的语义变换能力，把 dense 加进去、r 提到 16，直接 98%。另一个是温度校准——20 点网格搜索从 1.0 搜到 0.316，不做的话验证集过拟合测试集掉点。

---

## 十二、追问预判表

| 你说 | 面试官追问 | 回答方向 |
|------|-----------|---------|
| "10 家保险公司" | 多保司数据怎么处理 | 12 个标量字段 + expr 动态拼接多维过滤 |
| "混合检索" | Dense 和 Sparse 怎么融合 | RRF 排名融合，不看分数看排名，天然归一化 |
| "合规检查" | 具体检查什么 | 5 道：医疗建议/监管词/贬低竞品/金额引用/PII脱敏 |
| "意图路由三层" | 为什么不用纯 LLM | 成本：规则 0ms 命中 60%，BERT 5ms 命中 35%，LLM 200ms 才兜底 5% |
| "模型选 R1" | R1 和 V3 什么区别 | R1 有思维链适合多步推理，V3 快适合检索生成 |
| "缓存三级" | 防击穿怎么做的 | SET NX 互斥锁，3次递增等待重试读缓存 |
| "PII 脱敏" | 脱敏后怎么还原 | mapping 字典保存，LLM 返回后 restore() 还原 |
| "126 节点知识图谱" | 节点太少了吧？ | 数据源限制但 Schema 通用，接入新保司 MERGE 幂等导入。重点是搭框架不是堆数据 |
| "为什么 Neo4j" | 为什么不用 LLM 推理？ | LLM 可能幻觉，KG 1ms 精确查关系；推理路径可审计——保险合规硬需求 |
| "NetworkX→Neo4j" | 不是过度设计？ | NetworkX 一小时内迭代 Schema，确认本体无误才迁 Neo4j。双后端降级上层零改动 |
| "LoRA r=16" | 为什么不全量微调？ | BERT-base 110M 全量也能做。但 LoRA adapter 10MB 可快速迭代、过拟合风险低、效果更好(98.1% vs 94.4%) |
| "540 条训练数据" | 够吗？ | BERT-base 分类任务够了——验证集 98%，9 类测试全对。LLM 扩增质量高——只变句式不变意图 |
| "温度校准 T=0.316" | 这是什么？ | 推理时 logits 除以温度再 softmax，让置信度更准。20 点网格搜索在验证集上最小化 NLL |
| "merge_and_unload" | 部署要 peft 吗？ | 不需要。训练完合并 LoRA→完整模型，推理时直接 AutoModel 加载，零额外依赖 |

---

## 十三、高频技术问题速答

### BGE-M3 为什么选它？

同时输出 Dense(语义) + Sparse(关键词) 双向量。Dense 捕捉 "肺炎↔肺部感染" 的同义关系，Sparse 做 "免赔额" 精确关键词匹配。两路 RRF 融合，召回率高于纯 Dense。

### 为什么不用 LangChain 的 AgentExecutor？

while 循环对简单任务够用，但保险场景需要结构化中间结果传递和精细降级。自研 480 行 LLMClient 比 LangChain 的 30+ 文件轻量可控。

### 为什么 Cross-Encoder 只对 Top-30 做？

Cross-Encoder 逐 token 计算匹配度，精度高但慢 50-100 倍。全量做延迟不可接受。Top-30 是工程甜点。

### 为什么 Neo4j 和 NetworkX 两个后端？

先 NetworkX 快速验证 Schema——一小时内迭代几个版本。确认本体设计没问题后迁 Neo4j 做持久化。接口层抽象后保留 NetworkX 做降级——Neo4j 不可用自动切换。

### LoRA vs QLoRA vs 全量微调怎么选？

BERT-base 110M 参数量级不需要 QLoRA（量化反而降精度）。全量微调能做但 LoRA 更好——adapter 10MB 快迭代、正则化防过拟合、多业务线可训多个 adapter 共用基座。

### 怎么保证 Agent 挂了系统不崩？

四层降级：短 query 直接 RAG → 低置信度 RAG 降级 → Agent 全失败兜底话术 → 初始化失败退化纯 RAG。

### 知识图谱怎么和 RAG 结合？

不是替代是增强。KG 推理找到"哪些产品可能相关"→ 构造更精准的 Milvus 过滤条件 → RAG 搜出条款原文 → KG 推理路径注入 LLM prompt。KG 保证找对产品，RAG 保证读到对条款。

### 这个系统的瓶颈在哪？

LLM API 调用占了 ~200ms。优化方向：流式输出先行展示、常见问题 FAQ 缓存命中直接秒回（0ms）。BERT 推理 5ms 不是瓶颈。

---

> 结尾钩子 (每个回答后留一个):
> "检索这块可以展开讲下为什么 BGE-M3 双向量比纯 Dense 好"
> "合规层是国内保险场景独有的挑战，和通用 RAG 很不一样"
> "知识图谱+大模型是个有意思的方向，可以深入聊聊"
> "LoRA 调参踩了不少坑——r 值和 target_modules 的选择很有讲究"
> "多 Agent 协议设计迭代了三版，可以仔细聊聊"

---

*全部更新完毕。项目路径: ~/code/code/pythonCode/Insurance/*
