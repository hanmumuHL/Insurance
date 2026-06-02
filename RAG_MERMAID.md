# 保险多保司 RAG 智能问答系统 — Mermaid 架构图集

---

## 1. 总体架构全景图

```mermaid
graph TB
    subgraph 接入层["接入层 Gateway"]
        A1[Web Chat]
        A2[微信小程序]
        A3[APP SDK]
        A4[企微/飞书]
        A5[OpenAPI]
    end

    subgraph API层["API 层 FastAPI"]
        B1["/chat/stream<br/>WebSocket SSE"]
        B2["/agent/task<br/>Agent 任务"]
        B3["/chat/history"]
        B4["/admin/documents"]
        B5["/admin/eval"]
    end

    subgraph 安全层["安全层"]
        C1[PII 脱敏引擎<br/>正则 + NER 双层]
        C2[速率限制]
        C3[合规守卫<br/>Compliance Guard]
    end

    subgraph 编排层["编排层 Orchestrator"]
        D1[Query Router<br/>意图路由<br/>规则→BERT→LLM]
        D2[Strategy Selector<br/>6种检索策略]
        D3[Response Generator<br/>答案生成 + 溯源]
    end

    subgraph 检索层["RAG Pipeline"]
        E1[向量编码<br/>BGE-M3 本地]
        E2[Milvus 混合检索<br/>Dense + Sparse + 过滤]
        E3[Reranker 精排<br/>bge-reranker-large]
        E4[上下文组装<br/>Parent-Child 窗口]
    end

    subgraph Agent层["Agent Engine Phase 2"]
        F1[Planner<br/>LangGraph 状态图]
        F2[Executor<br/>7 个工具]
        F3[Checkpoint<br/>对话记忆]
    end

    subgraph 存储层["基础设施层"]
        G1[(Milvus<br/>向量库)]
        G2[(MySQL<br/>元数据 + FAQ)]
        G3[(Redis<br/>缓存 + 会话)]
        G4[(MinIO<br/>原始PDF)]
    end

    subgraph 模型层["模型服务"]
        H1[DeepSeek API<br/>主 LLM]
        H2[千问 API<br/>降级兜底]
        H3[BGE-M3<br/>本地 Embedding]
        H4[BERT<br/>意图分类]
    end

    接入层 --> API层
    API层 --> 安全层
    安全层 --> 编排层
    编排层 -->|RAG 意图| 检索层
    编排层 -->|Agent 意图| Agent层
    编排层 -->|闲聊| H1
    检索层 --> G1
    检索层 --> G2
    检索层 --> H3
    Agent层 --> G2
    Agent层 --> G3
    G1 --> G4
    H1 --> G4
    H2 --> G4
```

---

## 2. 部署架构 — 本地/云端分工

```mermaid
graph LR
    subgraph 云端["云端 API 不碰敏感数据"]
        CA[DeepSeek-V3<br/>LLM 答案生成]
        CB[通义千问<br/>降级兜底]
        CC[LLM 策略选择<br/>轻量任务]
    end

    subgraph 本地["本地部署 数据闭环"]
        LA[BGE-M3<br/>Embedding 编码]
        LB[bge-reranker-large<br/>精排]
        LC[BERT-base-Chinese<br/>意图分类]
        LD[(Milvus<br/>向量存储)]
        LE[PII 脱敏引擎]
    end

    用户查询 --> LE
    LE -->|脱敏 query| CA
    LE -->|脱敏 query| LA
    LA --> LD
    LD --> LB
    LB -->|Top-K 条款| CA
    CA -->|生成答案| 用户
    CB -.->|降级| CA
    LC --> LE
```

---

## 3. Query 处理主流程

```mermaid
sequenceDiagram
    actor U as 用户
    participant GW as Gateway
    participant PII as PII 脱敏层
    participant QR as Query Router
    participant SS as Strategy Selector
    participant MIL as Milvus
    participant RER as Reranker
    participant LLM as DeepSeek API
    participant DB as MySQL

    U->>GW: "众安尊享e生肺炎住院免赔额是多少？"
    GW->>PII: 原始 query

    Note over PII: 正则扫描: 无PII → 放行<br/>耗时 < 1ms

    PII->>QR: 脱敏后 query

    Note over QR: 第1层: 关键词规则<br/>"免赔额" 在规则库 → 命中<br/>intent = 条款解读

    QR->>SS: intent=条款解读, entities={保司:众安, 产品:尊享e生}

    Note over SS: "条款解读" → 直接检索<br/>filter: clause_type IN [保险责任,释义]

    SS->>MIL: 编码 query → 混合检索<br/>filter='insurer=="众安" and product_name=="尊享e生"'

    MIL-->>SS: Top-30 chunks

    SS->>RER: 精排 Top-30 → Top-5

    RER-->>SS: Top-5 + parent chunks

    Note over SS: 组装上下文<br/>child → 补 parent → 去重 → 截断

    SS->>LLM: system: 保险客服<br/>context: 5个条款chunks<br/>user: 脱敏query

    LLM-->>SS: 生成答案 + 条款引用

    SS->>DB: 记录检索日志<br/>query / intent / chunks / latency

    SS->>U: "众安尊享e生住院医疗保障的免赔额为..."
```

---

## 4. Query Router — 三层意图路由

```mermaid
flowchart TD
    Q["用户 query 进入"] --> L1

    subgraph L1["第1层: 关键词规则 零延迟"]
        direction LR
        K1{"关键词匹配"}
        K2["命中率 ~60%"]
        K3["直接返回 intent"]
    end

    L1 -->|命中| K1
    K1 --> K2 --> K3
    K3 --> OUT1["intent + filter 条件"]

    L1 -->|未命中| L2

    subgraph L2["第2层: BERT 分类 5ms延迟"]
        direction LR
        B1["BERT-base-Chinese<br/>8分类微调"]
        B2["confidence ≥ 0.5?"]
        B3["命中率 ~35%"]
    end

    L2 --> B1 --> B2
    B2 -->|是| B3 --> OUT2["intent + confidence"]
    B2 -->|否| L3

    subgraph L3["第3层: LLM 兜底 200ms延迟"]
        direction LR
        L3A["LLM API 分类<br/>temp=0, JSON输出"]
        L3B["命中率 ~5%"]
    end

    L3 --> L3A --> L3B --> OUT3["intent + entities"]

    OUT1 --> FAN{"意图分发"}
    OUT2 --> FAN
    OUT3 --> FAN

    FAN -->|闲聊寒暄| A1["LLM 直答"]
    FAN -->|条款解读| A2["RAG 检索"]
    FAN -->|保单查询| A3["Function Call + DB"]
    FAN -->|理赔咨询| A4["RAG + FuncCall"]
    FAN -->|产品对比| A5["对比检索"]
    FAN -->|保费试算| A6["FuncCall + 计算"]
    FAN -->|退保咨询| A7["RAG + 计算"]
    FAN -->|投诉建议| A8["转人工"]
```

---

## 5. 6 种检索策略路由

```mermaid
flowchart TD
    START["intent + query + entities"] --> SWITCH{策略选择<br/>LLM 判断}

    SWITCH -->|条款解读| S1["直接检索<br/>query 直接编码 → Milvus"]
    SWITCH -->|模糊问题| S2["HyDE 检索<br/>LLM 生成假设文档 → 用假设文档编码检索"]
    SWITCH -->|复杂多条件| S3["子查询拆分<br/>LLM 拆为 N 个子query → N 路并行检索 → RRF 融合"]
    SWITCH -->|产品对比| S4["对比检索<br/>拆 ProductA + ProductB<br/>N路并行 → 各自 Rerank → LLM交叉"]
    SWITCH -->|保费/条件查询| S5["条件检索<br/>结构化条件 → Milvus 标量过滤<br/>无需语义检索"]
    SWITCH -->|召回不足| S6["回溯检索<br/>Top-K 不足 → 放宽过滤 → 二次检索<br/>→ 融合去重"]

    S1 --> ENCODE
    S2 --> ENCODE
    S3 --> ENCODE
    S4 --> ENCODE
    S5 --> ENCODE
    S6 --> ENCODE

    ENCODE["BGE-M3 向量编码<br/>Dense 1024维 + Sparse 词汇权重"]

    ENCODE --> MILVUS["Milvus 混合检索<br/>IVF_FLAT + 稀疏倒排<br/>+ 标量过滤 ins≠r/product/doc_type..."]
```

---

## 6. Milvus 多维过滤检索

```mermaid
sequenceDiagram
    participant SS as Strategy Selector
    participant ENC as BGE-M3 编码
    participant MIL as Milvus

    Note over SS: query: "平安e生保住院能赔吗"<br/>intent: 条款解读<br/>entities: {保司:平安, 产品:e生保}

    SS->>ENC: query 文本
    ENC-->>SS: dense_vector(1024) + sparse_vector

    SS->>MIL: search()

    Note over MIL: Collection: insurance_clauses<br/>12 个标量字段

    rect rgb(240, 248, 255)
        Note over MIL: 过滤表达式构造

        MIL-->>MIL: filter = 'insurer=="平安健康"<br/>and product_name like "e生保%"<br/>and doc_type=="条款"<br/>and clause_type in ["保险责任","释义","责任免除"]<br/>and is_valid==true<br/>and chunk_type=="child"'
    end

    rect rgb(255, 248, 240)
        Note over MIL: 混合检索执行

        MIL-->>MIL: Dense ANN: IVF_FLAT (语义)<br/>Sparse: BM25 倒排索引 (关键词)<br/>RRF: k=60 加权融合<br/>+ 标量过滤剔除
    end

    MIL-->>SS: Top-30 子块<br/>+ 关联 parent_text

    Note over SS: 子块→补父块→去重→Top-5
```

---

## 7. PII 脱敏完整链路

```mermaid
flowchart LR
    subgraph 输入["用户输入"]
        RAW["我叫张三，身份证330102199001011234，<br/>我的平安e生保保单P2024XXXXX<br/>肺炎住院能赔吗？<br/>打我电话13812345678"]
    end

    subgraph 检测["检测层"]
        direction TB
        D1["正则匹配<br/>─────────<br/>身份证: \d{17}[\dXx]<br/>手机号: 1[3-9]\d{9}<br/>银行卡: \d{16,19}<br/>─────────<br/>命中率 ~90%"]
        D2["NER 模型<br/>─────────<br/>姓名识别<br/>地址识别<br/>─────────<br/>命中率 ~10%"]
        D3["规则兜底<br/>─────────<br/>保单号: P\d{4}XXXX<br/>自定义 pattern<br/>─────────<br/>命中率 ~5%"]
    end

    subgraph 脱敏["脱敏替换"]
        direction TB
        R1["张三 → [姓名]"]
        R2["330102... → [证件号]"]
        R3["P2024... → [POLICY_001]"]
        R4["1381234... → [手机号]"]
    end

    subgraph 输出["脱敏输出"]
        SAFE["我叫[姓名]，身份证[证件号]，<br/>我的平安e生保保单[POLICY_001]<br/>肺炎住院能赔吗？<br/>打我电话[手机号]"]
    end

    subgraph 映射表["PII 映射表 Redis"]
        MT["姓名: [姓名] → 张三<br/>证件号: [证件号] → 330102...<br/>POLICY_001: → P2024XXXXX<br/>手机号: [手机号] → 13812345678<br/>─────────<br/>TTL: 会话结束后 5min 清除"]
    end

    RAW --> D1
    RAW --> D2
    RAW --> D3
    D1 --> R1
    D1 --> R2
    D2 --> R1
    D3 --> R3
    D1 --> R4
    R1 --> SAFE
    R2 --> SAFE
    R3 --> SAFE
    R4 --> SAFE

    R1 -.->|写入映射| MT
    R2 -.->|写入映射| MT
    R3 -.->|写入映射| MT
    R4 -.->|写入映射| MT

    subgraph 还原["PII 还原"]
        REV["LLM 返回: '[姓名]先生，您的[POLICY_001]...'<br/>─────────<br/>查映射表还原 →<br/>'张三先生，您的P2024XXXXX...'"]
    end

    SAFE -->|发 LLM API| LLM_API["DeepSeek API<br/>收到脱敏 query"]
    LLM_API -->|返回答案| REV
    MT -.->|查表还原| REV
    REV -->|最终答案| 用户

    style 检测 fill:#fff3e0
    style 脱敏 fill:#e8f5e9
    style 映射表 fill:#e3f2fd
```

---

## 8. 数据摄取管道 — PDF → Milvus

```mermaid
flowchart TD
    subgraph 渠道["文档来源 3渠道"]
        direction LR
        CH1["API 拉取<br/>保司文档下发接口"]
        CH2["SFTP 同步<br/>定时扫描目录"]
        CH3["人工上传<br/>管理后台"]
    end

    CH1 --> INGEST["DocumentIngestion<br/>Orchestrator"]
    CH2 --> INGEST
    CH3 --> INGEST

    INGEST --> DEDUP{"MD5 去重"}
    DEDUP -->|已存在| SKIP["跳过"]
    DEDUP -->|新文档| PARSE

    subgraph 解析["PDF 解析"]
        direction TB
        PARSE["PyMuPDF / marker-pdf"]
        EXTRACT["InsuranceClauseExtractor<br/>章节拆分 + 产品信息提取"]
    end

    PARSE --> EXTRACT
    EXTRACT --> CHUNK

    subgraph 分块["智能分块"]
        direction TB
        CHUNK["DocumentChunker<br/>chunk_size=512, overlap=64"]
        PARENT["Parent-Child 策略<br/>4个子块 → 1个父块"]
    end

    CHUNK --> PARENT

    PARENT --> ENCODE["BGE-M3 向量编码<br/>Dense 1024维 + Sparse"]

    ENCODE --> SAVE

    subgraph 入库["双写入"]
        direction LR
        MILVUS_SAVE[("Milvus<br/>向量 + 12 标量字段")]
        MYSQL_SAVE[("MySQL<br/>chunk 元数据 + 全文")]
    end

    SAVE --> MILVUS_SAVE
    SAVE --> MYSQL_SAVE

    SKIP --> LOG["日志记录"]
    MILVUS_SAVE --> LOG
    MYSQL_SAVE --> LOG
```

---

## 9. Parent-Child 分块与检索策略

```mermaid
flowchart TD
    subgraph 分块["写入时: Parent-Child 分块"]
        direction TB
        RAW_TEXT["原始条款全文<br/>约 5000 字符"]
        SPLIT["按句号/换行切句"]
        CHILD["子块<br/>每个 ~512 字符<br/>overlap 64 字符"]
        PARENT["父块<br/>每 4 个子块合并<br/>~2000 字符"]
    end

    RAW_TEXT --> SPLIT --> CHILD --> PARENT

    subgraph 检索["检索时: small2big 策略"]
        direction TB
        QUERY["用户 query"]
        VECTOR["向量编码"]
        SEARCH["Milvus 检索<br/>filter: chunk_type=='child'"]
        TOP_K["Top-30 子块"]
        RERANK["Reranker 精排 → Top-5"]
        FETCH_PARENT["补父块<br/>查 parent_id → 完整上下文"]
        DEDUP["去重"]
        ASSEMBLE["组装 Prompt<br/>子块精确 + 父块完整"]
    end

    QUERY --> VECTOR --> SEARCH --> TOP_K --> RERANK --> FETCH_PARENT --> DEDUP --> ASSEMBLE

    CHILD -.->|chunk_type='child'| SEARCH
    PARENT -.->|parent_id 关联| FETCH_PARENT
```

---

## 10. 合规守卫规则

```mermaid
flowchart TD
    ANSWER["LLM 生成答案"] --> CHECK

    subgraph CHECK["合规守卫 Compliance Guard"]
        direction TB
        C1{"置信度 < 阈值?"}
        C2{"涉及赔付金额?"}
        C3{"医疗建议类?"}
        C4{"监管敏感词?"}
        C5{"贬低竞品?"}
    end

    C1 -->|是| REJECT["拒绝回答<br/>→ 转人工"]
    C1 -->|否| C2
    C2 -->|是| FORCE_CITE["强制引用条款原文"]
    C2 -->|否| C3
    C3 -->|是| BLOCK["自动拦截<br/>→ '请咨询专业医生'"]
    C3 -->|否| C4
    C4 -->|是| BLOCK
    C4 -->|否| C5
    C5 -->|是| FILTER["过滤贬低描述"]
    C5 -->|否| PASS["通过 ✅<br/>返回用户"]

    FORCE_CITE --> PASS
```

---

## 11. LLM API 混合部署 — 主备切换

```mermaid
flowchart TD
    REQ["LLM 请求"] --> PRIMARY

    PRIMARY{"DeepSeek-V3<br/>主 API"}
    PRIMARY -->|200 OK| RESP1["返回答案"]
    PRIMARY -->|超时/429/5xx| FALLBACK

    FALLBACK{"通义千问<br/>降级兜底"}
    FALLBACK -->|200 OK| RESP2["返回答案<br/>标记 fallback=true"]
    FALLBACK -->|超时/429/5xx| RETRY

    RETRY{"重试策略"}
    RETRY -->|第1次: DeepSeek| PRIMARY
    RETRY -->|第2次: 千问| FALLBACK
    RETRY -->|3次全失败| DEGRADE["降级回复<br/>'系统繁忙，请稍后再试'<br/>或 本地 BERT 规则回复"]

    RESP1 --> CIRCUIT["熔断器<br/>连续 5 次失败 → 熔断 30s"]
    RESP2 --> CIRCUIT
    CIRCUIT -.->|熔断期间| FALLBACK
```

---

## 12. 端到端延迟分解

```mermaid
gantt
    title 单次查询端到端延迟分解 ~350ms
    dateFormat X
    axisFormat %s

    section 安全
    PII脱敏          :pii, 0, 1
    section 路由
    关键词规则       :rule, 0, 1
    BERT意图分类     :bert, 1, 6
    section 检索
    LLM策略选择      :strategy, 1, 201
    BGE-M3编码       :encode, 6, 26
    Milvus混合检索   :milvus, 26, 56
    Reranker精排     :rerank, 56, 66
    section 生成
    上下文组装       :context, 66, 67
    LLM生成答案      :gen, 67, 317
    PII还原          :restore, 317, 318
    section 返回
    响应返回         :done, 318, 320
```

---

## 使用说明

在支持 Mermaid 的编辑器（VS Code + Mermaid 插件、Typora、Obsidian、GitHub）中可直接渲染。

或在浏览器中打开：https://mermaid.live 粘贴代码预览。
