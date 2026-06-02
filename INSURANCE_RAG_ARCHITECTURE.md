# 保险聚合平台 — RAG 智能客服 + Agent 系统架构设计

> **场景定位**：保险科技平台（聚合方），对接多家保险公司，将各保司产品登录 App 开放投保。
> 不是持牌保险机构，是渠道平台。面向 C 端用户，需要高并发、低成本、快速迭代。
>
> 基于 `integrated_qa_system` 的优秀基因，面向保险聚合场景的独立架构设计。
> 分两期：Phase 1 = RAG 智能客服，Phase 2 = Agent 智能体。

---

## 〇、场景定位（与单一保险公司的关键区别）

```
┌─────────────────────────────────────────────────────────────────┐
│                    你是「聚合平台」不是「持牌机构」                 │
├──────────────────────────┬──────────────────────────────────────┤
│ 保险公司自建系统          │ 保险聚合平台（你的场景）               │
├──────────────────────────┼──────────────────────────────────────┤
│ 自营产品，知识库固定       │ N家保司产品，频繁上下架，知识持续更新   │
│ 有保单核心数据            │ 无保单数据（或仅缓存脱敏数据）          │
│ 等保三级/银保监强监管      │ 等保二级/三级，平台方合规责任较轻       │
│ 数据绝不能出内网          │ 用户PII敏感但保险条款是公开信息         │
│ 7×24自建运维团队          │ 团队规模有限，偏好托管/API             │
│ 预算高（牌照值钱）         │ 预算敏感（C端App毛利薄）               │
└──────────────────────────┴──────────────────────────────────────┘

由此推导的核心决策:
  ✅ LLM 走 API → 零GPU硬件 + 弹性扩缩 + 自动升级
  ✅ Embedding 本地 → 保险条款不出内网（但条款本身是公开信息，非必须）
  ✅ PII 脱敏层 → 用户数据传给API前必须脱敏
  ✅ 多保司知识库 → 支持按保司/产品/险种多维过滤
```

---

## 一、总体架构全景图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         接入层 (Gateway)                             │
│   Web Chat │ 微信小程序 │ APP SDK │ 企微/飞书 │ OpenAPI (第三方)      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                      API 层 (FastAPI)                                │
│   /chat/stream (WebSocket SSE)  │  /agent/task (Agent任务)           │
│   /chat/history                 │  /admin/documents (文档管理)       │
│   /chat/feedback (用户反馈)      │  /admin/eval (评测管理)            │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  PII 脱敏层  │  ← 用户数据传给LLM API前强制脱敏
                    └──────┬──────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                      编排层 (Orchestrator)                           │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Query Router (意图路由)                     │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │   │
│  │  │ 闲聊寒暄  │  │ 条款咨询  │  │ 保单查询  │  │  理赔咨询     │ │   │
│  │  │ →LLM直答 │  │  → RAG   │  │→FuncCall  │  │→RAG+FuncCall │ │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────┘ │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │   │
│  │  │ 产品对比  │  │ 保费计算  │  │ 退保咨询  │  │  投诉建议     │ │   │
│  │  │→RAG+对比 │  │→FuncCall  │  │→RAG+计算 │  │  →转人工     │ │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────┘ │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Compliance Guard (合规守卫)                  │   │
│  │  • 置信度 < 阈值 → 拒绝回答 + 转人工                            │   │
│  │  • 涉及赔付金额 → 必须引用条款原文                              │   │
│  │  • 医疗建议类 → 自动拦截 + 引导咨询专业医生                      │   │
│  │  • 监管敏感词 → 自动拦截                                        │   │
│  │  • 多保司场景: 不得贬低竞品、不得推荐未授权产品                  │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  RAG Pipeline │  │ Agent Engine │  │ 人工客服路由  │
│  (知识检索)    │  │ (工具调用)    │  │ (Human Loop) │
└──────┬───────┘  └──────┬───────┘  └──────────────┘
       │                 │
       ▼                 ▼
┌──────────────────────────────────────────┐
│              基础设施层                    │
│  Milvus │ MySQL │ Redis │ MinIO          │
│  BGE-M3(本地) │ Reranker(本地) │ BERT   │
│                                          │
│  LLM API ──── DeepSeek / 通义千问 (云端) │
│  (保险条款上下文 + 脱敏query 传出)        │
└──────────────────────────────────────────┘
```

---

## 二、部署架构（混合模式 — 平台方最优解）

### 2.0 核心原则：LLM API + 本地 Embedding

```
┌─────────────────────────────────────────────────────────────────┐
│              本地 vs 云端 分工                                    │
├────────────────────────────┬────────────────────────────────────┤
│ 云端 API（不碰敏感数据）     │ 本地部署（数据闭环）                │
├────────────────────────────┼────────────────────────────────────┤
│ LLM 生成答案                │ Embedding (BGE-M3)                 │
│  → 传入的是脱敏query+条款    │  → 保险条款向量不过网              │
│                             │                                    │
│ 策略选择 (LLM)              │ Reranker (bge-reranker-large)      │
│  → 轻量任务，成本极低        │  → 排序逻辑本地闭环                │
│                             │                                    │
│ 意图分类 (可用LLM代替BERT)   │ BERT 分类器（可选）                │
│  → 省去训练维护成本          │  → 低延迟国产化方案                │
│                             │                                    │
│                             │ 向量数据库 (Milvus)                │
│                             │  → 保险文档向量本地存储            │
│                             │                                    │
│                             │ PII 脱敏引擎                      │
│                             │  → 用户数据脱敏后传给API           │
└────────────────────────────┴────────────────────────────────────┘

原则:
  ✓ 保险条款文档不出内网（Embedding本地编码）
  ✓ 用户PII不出内网（脱敏后再传给LLM API）
  ✓ LLM推理交给API（零GPU硬件成本，弹性伸缩）
  ✗ 不自己搭GPU集群（成本高、维护重、没意义）
```

### 2.1 PII 数据安全流转

```
用户输入: "我叫张三，身份证3301xxx，买的尊享e生保单号P2024xxx，上个月住院了能赔吗？"

         ↓  ┌────────────────────────────────────────┐
            │        PII 脱敏层（本地处理）            │
            │  · 姓名     → [姓名]                    │
            │  · 身份证   → [证件号]                  │
            │  · 手机号   → [手机号]                  │
            │  · 保单号   → [POLICY_001]              │
            │  · 银行卡   → [银行卡号]                │
            └────────────────┬───────────────────────┘
                             ↓
         "买的尊享e生保单号[POLICY_001]，上个月住院了能赔吗？"

         ↓  ┌────────────────────────────────────────┐
            │     知识检索（本地 Milvus）              │
            │  搜索: "尊享e生 住院 赔付条件"           │
            │  返回: 条款第2.3条、第2.5条（纯公开信息） │
            └────────────────┬───────────────────────┘
                             ↓
         ┌───────────────────────────────────────────┐
         │           LLM API 请求                     │
         │  {                                        │
         │    "system": "你是保险客服，基于条款回答",   │
         │    "user": "尊享e生保单[POLICY_001]，       │
         │            住院了能赔吗？",                  │
         │    "context": "条款2.3: ... 条款2.5: ..."   │
         │  }                                        │
         │  → 出境数据: 仅公开条款 + 脱敏query         │
         │  → 合规结论: 一般无合规风险                 │
         └───────────────────────────────────────────┘
                             ↓
         LLM API 返回答案
                             ↓
         ┌───────────────────────────────────────────┐
         │         PII 还原层（本地处理）              │
         │  "[姓名]先生，您的尊享e生保单P2024xxx..."    │
         └───────────────────────────────────────────┘
```

### 2.2 LLM API 选型与时间线

```
2023年（项目启动期）
  LLM:   通义千问 (DashScope) qwen-max / qwen-plus
  理由:  当时国产API最成熟，阿里云生态合规
  价格:  ~0.04元/千token（输入）, ~0.12元/千token（输出）
  并发:  阿里云弹性扩容

2024年（升级优化期）
  LLM:   通义千问 → DeepSeek-V2 API（年中迁移）
  理由:  价格骤降到千问的 1/20，中文保险场景效果不输
  价格:  ~0.001元/千token（输入）, ~0.002元/千token（输出）
  注意:  意图分类/策略选择等轻量任务可切到 DeepSeek-V3

2025年（成熟运营期）
  LLM:   DeepSeek-V3 / R1 API
  理由:  R1 推理能力更强，Agent多步规划更稳定
  架构:  主模型 DeepSeek-V3（对话）, 推理模型 R1（Agent规划）
  备选:  通义千问作为降级兜底

多Provider容灾:
  ┌─────────────┐     ┌─────────────┐
  │ DeepSeek    │────▶│  正常服务    │
  │ (主)        │     └─────────────┘
  └──────┬──────┘
         │ 超时/限流/故障
         ▼
  ┌─────────────┐     ┌─────────────┐
  │ 通义千问     │────▶│  降级服务    │
  │ (备)        │     │  略慢但可用   │
  └─────────────┘     └─────────────┘
```

### 2.3 硬件方案（本地部分）

```
┌─────────────────────────────────────────────────────────────┐
│              本地硬件（仅跑 Embedding + Reranker + BERT）     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  GPU 服务器:  1× NVIDIA L4 24GB (或 T4 16GB)                │
│                                                              │
│  显存分配:                                                   │
│    BGE-M3 Embedding     ~3GB   (常驻, 处理文档入向量库)      │
│    bge-reranker-large   ~3GB   (按需加载, 用完释放)          │
│    BERT 分类器          ~1GB   (常驻, 或切LLM API省掉)       │
│    ─────────────────────────                                │
│    峰值占用              ~7GB   (单卡 L4 24GB 绰绰有余)      │
│                                                              │
│  通用服务器:  2× 32C/64GB RAM/1TB NVMe                      │
│    · 服务器1: Milvus 向量库 + MySQL + Redis                  │
│    · 服务器2: FastAPI 业务逻辑 + PII脱敏 + API网关           │
│                                                              │
│  总硬件成本:  ~4万元（远低于纯本地LLM部署的20万+）            │
│                                                              │
│  对比:                                                       │
│    纯本地部署(含LLM):  L40S×2 + A10 ≈ 20万 + 运维人力       │
│    混合部署(API):      L4×1 + 2服务器 ≈ 4万 + API按量        │
│    节省:                约80%硬件成本，零LLM运维负担          │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、多保司知识库设计（聚合平台特有）

### 3.1 知识库架构

```
┌──────────────────────────────────────────────────────────────┐
│                 多保司产品知识库（统一索引）                    │
│                                                               │
│  保险条款文档源:                                               │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────┐ │
│  │平安条款  │ │人保条款  │ │泰康条款  │ │众安条款  │ │ ...   │ │
│  │重疾/医疗 │ │重疾/医疗 │ │重疾/医疗 │ │医疗/意外 │ │       │ │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └──┬────┘ │
│       └───────────┼───────────┼───────────┼─────────┘       │
│                   ▼           ▼           ▼                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Milvus 向量库 (统一索引)                 │   │
│  │                                                       │   │
│  │  metadata 多维标签:                                    │   │
│  │  {                                                    │   │
│  │    "company":       "众安保险",     ← 保司名称         │   │
│  │    "product_code":  "ZJX2024",      ← 产品唯一编码     │   │
│  │    "product_name":  "尊享e生2024",  ← 产品名称         │   │
│  │    "product_type":  "医疗险",       ← 险种类型         │   │
│  │    "clause_no":     "第2.3条",      ← 条款编号         │   │
│  │    "clause_type":   "保险责任",     ← 条款类型         │   │
│  │    "doc_type":      "条款",         ← 文档类型         │   │
│  │    "version":       "2024v1",       ← 版本号           │   │
│  │    "effective_date":"2024-01-01",   ← 生效日期         │   │
│  │    "status":        "active"        ← 状态(active/deprecated)│
│  │  }                                                   │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
│  检索时支持多维过滤:                                           │
│    "尊享e生2024的免赔额是多少"                                │
│      → filter: product_code = "ZJX2024"                      │
│    "平安和众安的百万医疗险哪个好"                               │
│      → multi-query:                                         │
│          query1: "百万医疗险 保障责任" filter:company="平安"    │
│          query2: "百万医疗险 保障责任" filter:company="众安"    │
│      → LLM 对比总结                                          │
│    "医疗险的等待期一般是多久"                                   │
│      → filter: clause_type = "等待期"                        │
│      → filter: product_type = "医疗险"                       │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 产品生命周期管理

```
保司产品频繁上下架是聚合平台的核心运维挑战：

┌─────────────────────────────────────────────────────────────┐
│              产品状态管理                                     │
│                                                              │
│  新产品上线:                                                  │
│    1. 保司提供条款PDF                                        │
│    2. 文档解析 → 层级切片 → 向量入库                          │
│    3. status = "active", effective_date = 生效日期            │
│    4. 同步更新 FAQ 精确匹配表                                 │
│                                                              │
│  产品下架/版本更新:                                           │
│    1. 旧版本: status = "deprecated"（不删除，保留作为历史参照）│
│    2. 新版本: status = "active", version = "2025v1"          │
│    3. 检索时默认只查 status = "active"                       │
│    4. 用户问旧版本产品时，仍可检索 deprecated 数据             │
│                                                              │
│  自动化运维:                                                  │
│    · 定时任务扫描 effective_date < now() → 自动激活           │
│    · 定时任务扫描保司通知的下架日期 → 自动标记 deprecated     │
│    · 产品变更 → 触发知识库增量更新（非全量重建）              │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Milvus Schema（多保司扩展版）

```
字段              类型           说明
──────────────────────────────────────────────────
id               VARCHAR(100)   主键 (MD5 hash)
text             VARCHAR(65535) 子块文本
dense_vector     FLOAT_VECTOR   稠密向量 (1024维)
sparse_vector    SPARSE_FLOAT   稀疏向量 (BGE-M3 token权重)
parent_id        VARCHAR(100)   父块ID
parent_content   VARCHAR(65535) 父块全文
company          VARCHAR(50)    保司名称 (如 "众安保险")
product_code     VARCHAR(50)    产品编码 (如 "ZJX2024")
product_name     VARCHAR(100)   产品名称 (如 "尊享e生2024版")
product_type     VARCHAR(50)    险种 (重疾/医疗/意外/寿险/车险)
clause_no        VARCHAR(50)    条款编号 (如 "第2.3条")
clause_type      VARCHAR(50)    条款类型 (保险责任/责任免除/释义/理赔)
doc_type         VARCHAR(50)    文档类型 (条款/投保须知/理赔指引/FAQ)
version          VARCHAR(20)    版本号
effective_date   VARCHAR(20)    生效日期
status           VARCHAR(20)    状态 (active/deprecated)
timestamp        VARCHAR(50)    索引时间戳
```

---

## 四、RAG Pipeline 详细设计（Phase 1）

### 4.1 检索架构（继承 integrated_qa_system 核心基因）

```
Query → 意图路由 ──→ [条款咨询] ──→ Strategy Selector(LLM API)
                                        │
                          ┌─────────────┼─────────────┐
                          ▼             ▼             ▼
                    直接检索        HyDE假设答案    子查询拆分
                          │             │             │
                          └─────────────┼─────────────┘
                                        ▼
                              Hybrid Search 混合检索
                              ┌─────────┴─────────┐
                              ▼                   ▼
                         Dense 语义检索       Sparse 关键词检索
                         (BGE-M3 dense)       (BGE-M3 sparse)
                              │                   │
                              └─────────┬─────────┘
                                        ▼
                              WeightedRanker 加权融合
                              (dense:1.0, sparse:0.7)
                                        │
                                        ▼
                              父子块去重 → 返回 Parent Docs
                              支持多保司过滤 (company/product_code)
                                        │
                                        ▼
                              BGE-Reranker 精排
                              (CrossEncoder pairwise)
                                        │
                                        ▼
                              Top-K 父块上下文
```

### 4.2 保险文档处理 Pipeline

```
原始文档                        结构化知识
────────                        ──────────
平安-重疾险条款.pdf ──┐
人保-医疗险条款.pdf ──┤
泰康-理赔指引.pdf ────┼──→ Document Loader ──→ 层级感知切分 ──→ 元数据标注
众安-投保须知.pdf ────┤    (PDF/OCR/MD)         (章→节→条)      (保司/产品/条款号/版本)
...                    │
                       │
                    ┌──▼──────────────────┐
                    │  自动化文档同步      │
                    │  · 保司SFTP拉取      │
                    │  · Webhook通知更新   │
                    │  · 人工上传兜底      │
                    └─────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
              向量化入库                结构化索引
          (BGE-M3→Milvus)          (MySQL 元数据+FAQ)
```

#### 切分策略（多保司保险优化）

| 文档类型 | 切分方法 | Chunk Size | Overlap | 特殊处理 |
|---------|---------|------------|---------|---------|
| 保险条款 | 层级感知切分 | 父1200/子300 | 50 | 保留"第X条"编号+保司+产品码 |
| 投保须知 | 按段落切分 | 800 | 100 | 保留"健康告知/免责声明"标签 |
| 理赔指引 | 按流程步骤切分 | 600 | 50 | 标注"报案/查勘/定损/理算/支付"阶段 |
| FAQ库 | 按问答对切分 | 原生QA | 0 | 不拆分, Q=chunk, A=metadata |
| 监管文件 | 按条款切分 | 1000 | 100 | 标注文号+发布机构 |
| 产品对比表 | 按产品维度切分 | 500 | 20 | 保留对比维度标签（保额/保费/免赔额） |

### 4.3 Prompt 工程体系

```
┌─────────────────────────────────────────────────┐
│                 System Prompt 层                  │
│  · 角色: 保险平台客服，中立客观介绍各保司产品       │
│  · 约束: 不确定就说不知道，严禁编造保险条款           │
│  · 合规: 不得贬低任何保司，不得推荐未上架产品        │
│  · 风格: 专业但不冰冷，解释通俗易懂                │
│  · 溯源: 所有回答必须引用具体条款来源（保司+条款号） │
│  · 兜底: 复杂理赔/退保金额计算 → 引导联系保司客服   │
└─────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
  条款查询Prompt    产品对比Prompt    理赔咨询Prompt
  template:         template:         template:
  - 上下文片段       - 多产品条款       - 理赔流程指引
  - 条款编号强制引用  - 维度对比表格     - 材料清单
  - 免责条款提示     - 中立不偏袒       - 理赔时效
  - 人工客服兜底     - 结论+依据       - 争议处理
                              │
                    多保司特殊约束:
                    · 不评价保司优劣
                    · 只陈述条款事实
                    · 引导用户自行对比决策
```

### 4.4 MySQL 表设计

```sql
-- FAQ 精确匹配表（多保司）
CREATE TABLE insurance_faq (
    id          INT PRIMARY KEY AUTO_INCREMENT,
    company     VARCHAR(50),        -- 保司名称
    product_code VARCHAR(50),       -- 产品编码
    question    TEXT NOT NULL,
    answer      TEXT NOT NULL,
    category    VARCHAR(50),        -- 险种/理赔/投保/退保
    priority    INT DEFAULT 0,
    is_active   BOOLEAN DEFAULT TRUE,
    created_at  DATETIME,
    updated_at  DATETIME,
    FULLTEXT idx_question (question)
);

-- 产品注册表（管理多保司产品生命周期）
CREATE TABLE product_registry (
    id              INT PRIMARY KEY AUTO_INCREMENT,
    company         VARCHAR(50) NOT NULL,
    product_code    VARCHAR(50) NOT NULL UNIQUE,
    product_name    VARCHAR(100) NOT NULL,
    product_type    VARCHAR(50),       -- 重疾/医疗/意外/寿险/车险
    status          ENUM('active','pending','deprecated'),
    effective_date  DATE,
    expired_date    DATE,
    created_at      DATETIME,
    updated_at      DATETIME,
    INDEX idx_company (company),
    INDEX idx_status (status)
);

-- 对话历史表
CREATE TABLE conversations (
    id          INT PRIMARY KEY AUTO_INCREMENT,
    session_id  VARCHAR(36) NOT NULL,
    user_id     VARCHAR(50),          -- 用户标识(脱敏)
    question    TEXT NOT NULL,
    answer      TEXT NOT NULL,
    intent      VARCHAR(50),
    citations   JSON,                 -- 引用的条款来源 [{company, clause_no, content}]
    confidence  FLOAT,
    feedback    TINYINT,              -- 0未评/1赞/-1踩
    timestamp   DATETIME,
    INDEX idx_session (session_id),
    INDEX idx_user (user_id)
);

-- 文档元数据表
CREATE TABLE document_registry (
    id              INT PRIMARY KEY AUTO_INCREMENT,
    file_name       VARCHAR(255),
    company         VARCHAR(50),
    product_code    VARCHAR(50),
    doc_type        VARCHAR(50),
    version         VARCHAR(20),
    chunk_count     INT,
    status          ENUM('processing','active','deprecated'),
    uploaded_at     DATETIME,
    INDEX idx_product (product_code),
    INDEX idx_company (company)
);

-- 评测数据集表
CREATE TABLE eval_dataset (
    id          INT PRIMARY KEY AUTO_INCREMENT,
    question    TEXT NOT NULL,
    expected_answer TEXT,
    expected_sources JSON,
    category    VARCHAR(50),
    difficulty  ENUM('easy','medium','hard'),
    created_at  DATETIME
);
```

---

## 五、Agent 引擎设计（Phase 2）

### 5.1 Agent 架构

```
┌─────────────────────────────────────────────────────────┐
│                    Insurance Agent                        │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │              Planner (规划器)                      │   │
│  │  LLM API分析任务 → 分解步骤 → 选择Tools → 编排    │   │
│  │  模型: DeepSeek-R1 (复杂推理)                     │   │
│  └──────────────────┬───────────────────────────────┘   │
│                     │                                    │
│  ┌──────────────────▼───────────────────────────────┐   │
│  │              Tool Registry (工具注册表)             │   │
│  │                                                    │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐          │   │
│  │  │RAG Search│ │Policy API│ │Claim API │          │   │
│  │  │多保司条款  │ │保单查询   │ │理赔查询   │          │   │
│  │  └──────────┘ └──────────┘ └──────────┘          │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐          │   │
│  │  │Premium   │ │Product   │ │Human    │          │   │
│  │  │保费试算   │ │多产品对比  │ │转人工    │          │   │
│  │  └──────────┘ └──────────┘ └──────────┘          │   │
│  │  ┌──────────┐                                    │   │
│  │  │Eligibility│                                    │   │
│  │  │理赔资格预检│                                    │   │
│  │  └──────────┘                                    │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │           Memory (记忆系统)                        │   │
│  │  · Short-term: 会话上下文 (最近N轮)                │   │
│  │  · Long-term: 用户画像 (偏好/已购产品/历史咨询)     │   │
│  │  · Working: 当前任务中间结果                       │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │         Safety Guard (安全守卫)                    │   │
│  │  · Tool调用前: 参数校验 + 权限检查                  │   │
│  │  · Tool调用后: 结果脱敏 + 合规过滤                  │   │
│  │  · 最终回答: 不贬低保司 + 溯源验证 + 转人工判断      │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 5.2 Agent Tools 详细设计

```
Tool 1: search_knowledge_base (多保司RAG检索)
  输入: query, company?, product_code?, clause_type?
  输出: [{content, clause_no, company, product_code, score}]
  说明: 封装 RAG Pipeline，支持按保司/产品/条款类型多维过滤

Tool 2: lookup_policy (保单查询)
  输入: policy_no, query_type (basic/cash_value/status)
  输出: {company, product_name, policy_info | cash_value | status_detail}
  说明: 对接各保司保单API，或Mock数据用于Demo
  注意: 作为平台方，可能只缓存脱敏保单摘要

Tool 3: query_claim (理赔查询)
  输入: claim_no, query_type (status/progress/materials)
  输出: {claim_status, progress_nodes, required_materials}
  说明: 对接保司理赔系统API，或Mock数据

Tool 4: calculate_premium (保费试算)
  输入: product_code, age, gender, coverage_amount, duration
  输出: {annual_premium, total_premium, payment_schedule}
  说明: 根据各保司费率表计算，或Mock规则引擎

Tool 5: compare_products (多产品对比)
  输入: [product_codes], compare_dimensions
  输出: {comparison_table, recommendation_notes}
  说明: RAG检索各产品条款 + LLM中立对比总结
  约束: 不评价保司优劣，只陈述条款事实

Tool 6: check_claim_eligibility (理赔资格预检)
  输入: policy_no, incident_type, incident_date
  输出: {eligible, reason, required_documents, deadline}
  说明: 规则匹配保单责任 + 等待期/免责条款检查

Tool 7: escalate_to_human (转人工)
  输入: session_context, reason, priority
  输出: {ticket_id, estimated_wait_time, target_company?}
  说明: 复杂问题转人工，复杂理赔可引导至对应保司客服
```

### 5.3 Agent 典型执行流程示例

```
用户: "我买了众安的尊享e生2024，上个月肺炎住院花了2万，能赔多少？
      另外平安的e生保是不是差不多？哪个划算？"

Agent 执行计划:

  Step 1 → search_knowledge_base("尊享e生2024 住院医疗 赔付 免赔额",
                                   company="众安保险")
           → 条款2.3条"住院医疗保险金"、条款2.5条"免赔额"

  Step 2 → lookup_policy(policy_no="...")
           → 保单: 保额300万, 年度免赔额1万, 赔付比例100%

  Step 3 → check_claim_eligibility(policy_no, "住院", "肺炎")
           → 检查: 等待期已过✓, 非免责疾病✓, 保障期内✓

  Step 4 → search_knowledge_base("e生保 住院医疗 赔付 免赔额",
                                   company="平安保险")
           → 平安e生保条款

  Step 5 → compare_products(["ZJX2024", "PINGAN-ESB2024"],
                             dimensions=["保额","免赔额","赔付比例","保费","等待期"])
           → 对比表格: 众安尊享e生 vs 平安e生保

  Step 6 → LLM 综合分析:
           【理赔分析】
           您的众安尊享e生2024，住院花费2万元，
           扣除1万免赔额后，预计可赔付1万元。
           需准备: 住院发票、费用清单、出院小结、身份证
           [引用: 众安尊享e生2024条款第2.3条、第2.5条]

           【产品对比】
           ┌──────────┬──────────┬──────────┐
           │          │众安尊享e生│平安e生保  │
           ├──────────┼──────────┼──────────┤
           │ 保额     │ 300万    │ 200万    │
           │ 免赔额   │ 1万      │ 1万      │
           │ 赔付比例 │ 100%     │ 100%     │
           │ 保费     │ 368元/年 │ 425元/年 │
           │ 等待期   │ 30天     │ 30天     │
           └──────────┴──────────┴──────────┘
           [引用: 众安尊享e生条款, 平安e生保条款]
           建议您根据自身需求选择，两产品保障范围接近，
           众安尊享e生保额更高且保费更低。
```

---

## 六、目录结构设计

```
insurance-ai-platform/
│
├── README.md
├── requirements.txt
├── docker-compose.yml              # Milvus + MySQL + Redis + MinIO
├── Makefile
├── .env.example
├── config/
│   ├── settings.py                 # 全局配置 (Pydantic Settings)
│   ├── model_config.yaml           # LLM API Provider/Embedding配置
│   │                                # provider: deepseek/aliyun
│   │                                # fallback: deepseek→aliyun
│   └── prompts/                    # 所有Prompt模板
│       ├── system_prompts.yaml     # System Prompt (多场景)
│       ├── rag_prompts.yaml        # RAG Prompt
│       ├── agent_prompts.yaml      # Agent Planner Prompt
│       └── strategy_prompts.yaml   # 检索策略Prompt
│
├── core/                           # 核心引擎 (RAG Pipeline)
│   ├── __init__.py
│   ├── document/
│   │   ├── loader.py              # 多格式文档加载器
│   │   ├── chunker.py             # 层级感知切片器
│   │   └── metadata.py            # 元数据提取(保司/产品/条款号/版本)
│   ├── embedding/
│   │   ├── bge_m3.py              # BGE-M3 Dense+Sparse编码 (本地)
│   │   └── reranker.py            # BGE-Reranker精排 (本地)
│   ├── retrieval/
│   │   ├── hybrid_search.py       # Milvus混合检索 (多保司过滤)
│   │   ├── bm25_search.py         # BM25关键词检索
│   │   └── fusion.py              # 结果融合(RRF/Weighted)
│   ├── strategies/
│   │   ├── base.py                # 策略基类
│   │   ├── direct.py              # 直接检索
│   │   ├── hyde.py                # 假设文档检索
│   │   ├── subquery.py            # 子查询拆分
│   │   └── backtracking.py        # 回溯简化
│   └── generator/
│       ├── llm_client.py          # LLM统一客户端(多Provider容灾)
│       └── answer_builder.py      # 答案构建(溯源+格式化+PII还原)
│
├── agent/                          # Agent引擎 (Phase 2)
│   ├── __init__.py
│   ├── planner.py                 # 任务规划器 (调用DeepSeek-R1 API)
│   ├── executor.py                # 工具执行器
│   ├── memory/
│   │   ├── short_term.py          # 会话上下文
│   │   └── long_term.py           # 用户画像
│   └── tools/                     # Agent工具集
│       ├── __init__.py
│       ├── base.py                # 工具基类 + 注册装饰器
│       ├── rag_search.py          # 多保司RAG检索工具
│       ├── policy_lookup.py       # 保单查询工具
│       ├── claim_query.py         # 理赔查询工具
│       ├── premium_calc.py        # 保费试算工具
│       ├── product_compare.py     # 多产品对比工具
│       └── human_escalate.py      # 转人工工具
│
├── router/                         # 意图路由
│   ├── __init__.py
│   ├── classifier.py              # BERT分类器 / LLM API few-shot 分类
│   ├── intent_map.py              # 意图→处理策略映射
│   └── guard.py                   # 合规守卫 (含多保司中立检查)
│
├── security/                       # 安全层（新增）
│   ├── __init__.py
│   ├── pii_detector.py            # PII识别 (姓名/身份证/手机号/银行卡)
│   ├── pii_masker.py              # PII脱敏 → 占位符
│   └── pii_restorer.py            # PII还原 ← 占位符
│
├── api/                            # API层
│   ├── __init__.py
│   ├── main.py                    # FastAPI入口
│   ├── routes/
│   │   ├── chat.py                # /chat/* 对话接口
│   │   ├── agent.py               # /agent/* Agent任务接口
│   │   ├── admin.py               # /admin/* 管理接口(含产品上下架)
│   │   └── eval.py                # /eval/* 评测接口
│   ├── middleware/
│   │   ├── auth.py                # 认证中间件
│   │   ├── logging.py             # 日志中间件
│   │   └── rate_limit.py          # 限流中间件
│   └── schemas/
│       ├── chat.py                # 对话请求/响应模型
│       └── agent.py               # Agent请求/响应模型
│
├── services/                       # 业务服务层
│   ├── __init__.py
│   ├── chat_service.py            # 对话服务编排
│   ├── agent_service.py           # Agent服务编排
│   ├── product_service.py         # 产品生命周期管理 (上下架/版本)
│   └── feedback_service.py        # 用户反馈收集
│
├── db/                             # 数据层
│   ├── __init__.py
│   ├── mysql_client.py            # MySQL连接管理
│   ├── redis_client.py            # Redis缓存管理
│   ├── milvus_client.py           # Milvus向量库管理
│   └── models/                    # ORM模型 (SQLAlchemy)
│       ├── conversation.py
│       ├── faq.py
│       ├── product.py             # 产品注册模型(新增)
│       └── document.py
│
├── evaluation/                     # 评测体系
│   ├── __init__.py
│   ├── metrics.py                 # 评测指标(RAGAS/自定义)
│   ├── dataset.py                 # 评测数据集管理
│   ├── runner.py                  # 评测执行器
│   └── report.py                  # 评测报告生成
│
├── observability/                  # 可观测性
│   ├── __init__.py
│   ├── tracing.py                 # 链路追踪
│   ├── metrics.py                 # 业务指标(Prometheus)
│   └── alerting.py                # 告警规则
│
├── data/                           # 数据目录
│   ├── documents/                 # 原始保险文档(按保司分目录)
│   │   ├── 平安保险/
│   │   │   ├── e生保2024-条款.pdf
│   │   │   └── e生保2024-投保须知.pdf
│   │   ├── 众安保险/
│   │   │   ├── 尊享e生2024-条款.pdf
│   │   │   └── 尊享e生2024-理赔指引.pdf
│   │   └── ...
│   ├── faq/                       # FAQ导入文件
│   └── eval/                      # 评测数据集
│
├── frontend/                       # 前端 (React/Next.js)
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatWindow.tsx     # 对话窗口
│   │   │   ├── SourceCitation.tsx # 溯源引用展示(保司+条款号悬浮)
│   │   │   ├── AgentThinking.tsx  # Agent思考过程可视化
│   │   │   ├── ProductCompare.tsx # 产品对比卡片
│   │   │   └── FeedbackButton.tsx # 点赞/踩反馈
│   │   ├── pages/
│   │   └── hooks/
│   └── package.json
│
├── scripts/                        # 运维脚本
│   ├── ingest_documents.py        # 文档批量导入
│   ├── sync_products.py           # 产品同步 (保司SFTP→入库)
│   ├── train_classifier.py        # BERT分类器训练
│   ├── run_evaluation.py          # 批量评测
│   └── export_data.py             # 数据导出
│
└── tests/                          # 测试
    ├── unit/
    ├── integration/
    └── e2e/
```

---

## 七、关键技术决策

### 7.1 与 integrated_qa_system 的继承与演化

| 组件 | integrated_qa_system | 重量方案 | 变化说明 |
|------|---------------------|---------|---------|
| 向量库 | Milvus | **Milvus** (保留) | 生产级，保留 |
| Embedding | BGE-M3本地 | **BGE-M3本地** (保留) | 核心能力不改 |
| Reranker | bge-reranker-large | **bge-reranker-large** (保留) | 不改 |
| 分类器 | BERT 2分类 | **BERT 8分类 或 LLM API** | 2→8类意图 |
| 检索策略 | 4种 | **6种** (扩展) | +对比检索 +条件检索 |
| 切片 | 父子块 | **层级感知父子块** (增强) | 保条款编号/章节目录 |
| LLM | 千问API | **DeepSeek API + 千问降级** | 多Provider容灾 |
| 前端 | 原生HTML+JS | **React/Next.js** (重写) | 组件化溯源展示 |
| 评测 | RAGAS基础 | **完整评测体系** (新增) | 离线+在线 |
| 溯源 | 无 | **强制溯源(保司+条款号)** (新增) | 保险合规刚需 |
| Agent | 无 | **工具调用Agent** (新增) | Phase 2核心 |
| PII安全 | 无 | **脱敏/还原层** (新增) | API模式刚需 |
| 知识库 | 单学科 | **多保司多维过滤** (新增) | 聚合平台核心 |
| 产品管理 | 无 | **产品生命周期管理** (新增) | 上下架/版本同步 |

### 7.2 LLM API 策略

```
场景                 模型              参数                         调用方式
─────────────────────────────────────────────────────────────────────────
意图分类             DeepSeek-V3        temp=0, max_tokens=50         API
检索策略选择          DeepSeek-V3        temp=0.1, max_tokens=100     API
答案生成(RAG)        DeepSeek-V3        temp=0.3, max_tokens=2048    API
Agent规划            DeepSeek-R1        temp=0.2, max_tokens=4096    API
闲聊兜底             DeepSeek-V3        temp=0.7, max_tokens=1024    API
─────────────────────────────────────────────────────────────────────────
Embedding编码        BGE-M3             本地                          本地GPU
Reranker精排         bge-reranker-large  本地                          本地GPU
BERT分类(可选)       bert-base-chinese   本地(省API调用延迟)            本地GPU

多Provider容灾:
  主: DeepSeek API → 超时/限流 → 备: 通义千问 API
  意图分类可用本地BERT代替LLM API，零依赖云端
```

### 7.3 保险意图分类体系（替代原 BERT 2分类）

```
原系统:  {通用知识, 专业咨询}  (教育场景)
↓
新系统:  8类保险意图

  1. 闲聊寒暄        → LLM API direct answer (不消耗RAG)
  2. 条款解读        → RAG + 强制溯源(保司+条款号)
  3. 产品对比        → RAG multi-query(多保司) + LLM中立对比
                      特殊约束: 不贬低任何保司
  4. 保单查询        → Function Call (Agent) 或引导至保司
  5. 理赔咨询        → RAG + Function Call
  6. 保费试算        → Function Call (计算工具 或 保司API)
  7. 退保咨询        → RAG + Function Call (现金价值)
  8. 投诉建议        → 转人工 (或引导至对应保司客服)

分类方式:
  方案A: BERT本地分类 (低延迟, 无API成本, 需训练)
  方案B: DeepSeek-V3 API few-shot (零训练, 有API调用延迟)
  推荐: 初期方案B快速验证 → 稳定后用方案A节省成本
```

---

## 八、评测体系设计

### 8.1 离线评测（上线前）

```
评测维度            指标              方法
─────────────────────────────────────────────
检索质量            Recall@5, MRR     BGE-M3 vs 纯Dense vs 纯BM25
多保司检索准确率     Company Accuracy   是否正确检索到目标保司的条款
答案准确性          RAGAS faithfulness  LLM-as-Judge
答案相关性          RAGAS relevancy    LLM-as-Judge
溯源准确率          Citation Accuracy  人工标注验证
合规通过率          Compliance Rate   规则自动检查
  - 中立性           Neutrality Check  是否对保司有不公正评价
意图分类准确率       Accuracy, F1      BERT/LLM API 分类器评估
```

### 8.2 在线监控（上线后）

```
指标              阈值告警           说明
─────────────────────────────────────────────
人工转接率         > 30%            过高说明RAG能力不足
用户点赞率         < 70%            过低说明答案质量差
平均响应时间       > 5s             过长影响体验（LLM API延迟）
API调用失败率      > 1%             触发Provider切换
溯源覆盖率         < 90%            过低有合规风险
拒答率(合规拦截)    > 15%            过高说明知识库不足或意图分类不准
保司分布均衡度     单保司>60%        可能RAG存在保司偏向
PII泄露告警        > 0              任何PII泄露都是P0事故
```

---

## 九、里程碑规划

```
Phase 1 — 保险RAG智能客服 (4周)
  Week 1: 项目脚手架 + 多保司文档处理Pipeline + BGE-M3向量入库
          + 产品注册表 + PII脱敏层
  Week 2: 混合检索 + Rerank + 多保司过滤 + Prompt工程(含溯源+中立)
          + LLM API对接(DeepSeek + 通义千问降级)
  Week 3: 意图分类(先用LLM API, 后期可选BERT) + FastAPI + 前端(溯源展示)
  Week 4: 评测体系 + 合规守卫 + 压测优化 + README

Phase 2 — 保险Agent智能体 (3周)
  Week 5: Agent框架(Planner/Executor) + 7个Tool开发
          + 多保司产品对比工具
  Week 6: 多工具编排 + 用户画像记忆 + Agent前端可视化(思考过程+产品对比卡)
  Week 7: 端到端测试 + 场景演示录制 + 简历文档

简历亮点打包:
  "自研保险聚合平台智能客服，三层检索架构(BM25→意图路由→BGE-M3混合检索+
   Reranker精排)，支持8家保险公司产品知识库统一索引，多保司条款溯源+中立对比。
   LLM混合部署(DeepSeek API+千问降级+本地Embedding)降低80%硬件成本，
   PII脱敏安全方案确保用户数据不出内网。
   扩展为Agent智能体，集成保单查询/理赔追踪/保费试算/多产品对比等7个工具，
   多步推理自动完成复杂保险咨询，支持中立的跨保司产品对比分析。"
```

---

## 十、与原项目的差异化总结（最终版）

| 维度 | integrated_qa_system | 重量方案 |
|------|---------------------|---------|
| **场景** | IT教育（单领域） | 保险聚合平台（多保司） |
| **部署模式** | 纯本地 | LLM API + 本地Embedding混合 |
| **LLM** | 千问API(单一) | DeepSeek主 + 千问备(容灾) |
| **Embedding** | BGE-M3本地 | BGE-M3本地(不变) |
| **意图分类** | 2类(BERT) | 8类(先API后BERT) |
| **检索策略** | 4种 | 6种(+对比/+条件) |
| **知识库** | 单学科过滤 | 多保司多维过滤(保司/产品/险种/条款型) |
| **溯源** | 无 | 强制溯源(保司名+产品+条款编号+原文) |
| **合规** | 无 | 置信度门禁+中立检查+敏感词拦截 |
| **PII安全** | 无（本地LLM不外传） | PII脱敏/还原层(API模式刚需) |
| **产品管理** | 无 | 产品生命周期(上下架/版本/同步) |
| **Agent** | 无 | Planner+7Tools+Memory |
| **评测** | RAGAS基础 | 离线6维+在线8指标(含中立性+保司分布) |
| **前端** | 原生JS | React组件化(溯源+对比+思考过程) |
| **工程化** | 脚本式 | Pydantic+docker-compose+Makefile+多Provider |
| **硬件成本** | ~20万(含LLM GPU) | ~4万(仅本地Embedding GPU) |
