# 保险智能问答系统 - 完整项目流程图 (v3.2 记忆管理)

> PII 脱敏/还原由 Spring 网关完成。Python 侧新增长短记忆: RedisSaver Checkpoint + MySQL 用户画像。

## 图1: 系统总览 (Top-Level)

```mermaid
flowchart TB
    subgraph spring["☕ Spring 网关层"]
        U[用户] --> PIIS["PII 脱敏 + 认证<br/>提取 X-User-Id"]
        PIIS --> FW["转发脱敏query + X-User-Id"]
    end

    subgraph 入口["🚪 Python 入口层"]
        FW --> B["FastAPI Gateway<br/>POST /chat | /chat/stream"]
    end

    subgraph 记忆["🧠 记忆管理层"]
        B --> MM["MemoryManager<br/>短记忆: RedisSaver<br/>长记忆: MySQL 用户画像"]
    end

    subgraph 路由["🔀 意图路由"]
        B --> C["QueryClassifier<br/>3层意图分类"]
        C --> D{Agent初始化<br/>成功?}
    end

    subgraph 多智能体["🤖 纯多Agent系统"]
        D -->|是| E["Orchestrator<br/>+ user_profile + session_id"]
        E --> F{路由模式}
        F -->|multi_agent| G["多Agent协作<br/>Primary + Secondary<br/>RedisSaver 恢复历史"]
        F -->|rag_fallback| I["RAGSystem<br/>12阶段RAG管道"]
        G --> J["LLM结果聚合 + 合规审查"]
    end

    subgraph RAG["📚 RAG系统 (降级路径)"]
        D -->|否| I
        I --> J
    end

    subgraph 响应["📤 响应 → Spring还原"]
        J --> K["JSON/SSE 响应"]
        K --> PIIR["Spring PII还原"]
        PIIR --> L[用户]
    end

    style spring fill:#fff3e0
    style 入口 fill:#e1f5fe
    style 记忆 fill:#e8eaf6
    style 路由 fill:#f3e5f5
    style 多智能体 fill:#e8f5e9
    style RAG fill:#fce4ec
    style 响应 fill:#fff3e0
```

## 图2: 完整请求处理流程 (端到端)

```mermaid
flowchart TD
    START(("Spring 转发脱敏query")) --> API["POST /chat<br/>GET /chat/stream (SSE)"]

    API --> INIT["init_multi_agent()<br/>懒加载4个Agent + Orchestrator"]

    INIT -->|成功| QC["QueryClassifier.classify()<br/>3层意图路由"]
    INIT -->|失败| RAG_INIT_FAIL["RAGSystem.query()<br/>RAG降级处理"]

    QC --> L1["Layer1: 关键词规则<br/>~60%命中率, 0ms"]
    L1 -->|命中| INTENT
    L1 -->|未命中| L2["Layer2: BERT模型<br/>~35%命中率, 5ms"]
    L2 -->|命中| INTENT
    L2 -->|未命中| L3["Layer3: LLM零样本<br/>~5%命中率, 200ms"]
    L3 --> INTENT["IntentResult<br/>{intent, confidence, entities}"]

    INTENT --> ORCH["Orchestrator.process()<br/>纯多Agent调度中心"]

    ORCH --> ROUTE["_route_intent()"]
    ROUTE --> SHORT{极短查询?<br/>≤8字符}
    SHORT -->|是| RAG_FB
    SHORT -->|否| MODE{路由模式}
    MODE -->|multi_agent| BUILD["_build_tasks(2个任务)<br/>Primary + Secondary"]
    MODE -->|rag_fallback| RAG_FB["_rag_fallback()<br/>→ RAGSystem"]

    BUILD --> EXEC["_execute_tasks()<br/>按依赖顺序串行"]

    EXEC --> AGENT_CALL["SubAgent.invoke(task)"]

    subgraph AGENT_LOOP["🔄 SubAgent Plan-Execute 循环"]
        AGENT_CALL --> PLAN["plan_node: LLM→工具调用计划"]
        PLAN --> EXEC_N["exec_node: 执行领域工具"]
        EXEC_N --> CHECK["check_node: 检查结果"]
        CHECK -->|不足| PLAN
        CHECK -->|充足| SYNTH["synthesize_node: 生成答案"]
        SYNTH --> AGENT_RESULT["SubAgentResult"]
    end

    AGENT_RESULT --> AGG["_aggregate_results()<br/>LLM合成多Agent结果"]
    AGG --> COMPLY["_compliance_review()<br/>替换禁用词"]
    COMPLY --> RESP["JSON响应 → Spring还原PII"]

    RESP --> END((("返回用户")))

    RAG_INIT_FAIL --> RAG_PIPE

    subgraph RAG_PIPELINE["📚 RAGSystem 12阶段管道"]
        RAG_FB --> R1["1. FAQ缓存检查"]
        R1 --> R2["2. 领域边界检查"]
        R2 --> R3["3. 意图分类"]
        R3 --> R4["4. 查询结果缓存"]
        R4 --> R5["5. 闲聊→LLM直接回复"]
        R5 --> R6["6. 投诉→转人工"]
        R6 --> R7["7. 策略选择 (6种策略)"]
        R7 --> R8["8. Milvus混合检索+BGE-M3+Reranker"]
        R8 --> R9["9. 检索质量检查"]
        R9 --> R10["10. LLM答案生成"]
        R10 --> R11["11. 合规检查 (5项规则)"]
        R11 --> R12["12. 缓存回写"]
    end

    RAG_PIPE --> COMPLY

    style START fill:#e65100,color:#fff
    style END fill:#4caf50,color:#fff
    style AGENT_LOOP fill:#e3f2fd
    style RAG_PIPELINE fill:#fce4ec
```

## 图3: 多Agent编排流程 (Orchestrator 核心)

```mermaid
flowchart TD
    INPUT["Orchestrator.process(query, intent, entities)<br/>query 已是脱敏文本"]

    INPUT --> ROUTE["_route_intent()<br/>查 INTENT_ROUTING 表"]

    ROUTE --> RTABLE["11种意图 → 多Agent路由:<br/>• 产品咨询 → insurance + service<br/>• 产品对比 → insurance + service<br/>• 保费试算 → insurance + service<br/>• 投保流程 → insurance + underwriting<br/>• 核保咨询 → underwriting + insurance<br/>• 理赔咨询 → claim + underwriting<br/>• 理赔进度 → claim + service<br/>• 条款解读 → claim + insurance<br/>• 保单查询 → service + insurance<br/>• 投诉建议 → RAG降级<br/>• 闲聊 → RAG降级"]

    RTABLE --> MODE{"路由模式判定"}
    MODE -->|"multi_agent"| BUILD["_build_tasks()<br/>Primary + Secondary<br/>Secondary 依赖 Primary"]
    MODE -->|"rag_fallback"| RAG["_rag_fallback()<br/>RAGSystem.query()"]

    BUILD --> EXEC["_execute_tasks()<br/>Task 0 → Task 1 串行"]

    EXEC --> TASK0["Task 0: Primary Agent<br/>注入 user_profile"]
    TASK0 --> INVOKE0["PrimaryAgent.invoke(task_0)"]
    INVOKE0 --> DEP{"注入上游结果<br/>到 Task 1 context"}

    DEP --> TASK1["Task 1: Secondary Agent<br/>context 含 upstream_result"]
    TASK1 --> INVOKE1["SecondaryAgent.invoke(task_1)"]

    INVOKE1 --> AGGRE["_aggregate_results()<br/>LLM合成结果 + 去重 + 统一格式"]

    AGGRE --> COMPLY["_compliance_review()<br/>替换确定性承诺<br/>'一定能赔'→'以审核为准'"]

    COMPLY --> DONE["返回最终答案<br/>{answer, route_mode, agents_used, sources}<br/>→ Spring 还原 PII"]

    style INPUT fill:#e65100,color:#fff
    style DONE fill:#4caf50,color:#fff
    style RTABLE fill:#e3f2fd
```

## 图4: SubAgent Plan-Execute 循环

```mermaid
flowchart TD
    INVOKE["SubAgent.invoke(task)"]

    INVOKE --> INIT_S["初始化 AgentState<br/>max_iterations=3"]

    INIT_S --> PLAN_N["plan_node<br/>领域LLM→工具调用计划"]

    subgraph PLAN_DETAIL["计划生成"]
        PLAN_N --> P1["分析 task + 上游 context"]
        P1 --> P2["选择领域工具 (3-5个)"]
        P2 --> P3["生成JSON计划<br/>{tool_calls: [{name, args, depends_on}]}"]
    end

    P3 --> EXEC_N["exec_node<br/>按计划执行工具"]

    subgraph TOOLS["领域工具集"]
        subgraph INS["InsuranceAgent"]
            I1["product_compare"] & I2["premium_calc"] & I3["clause_search"] & I4["human_handoff"]
        end
        subgraph CLM["ClaimAgent"]
            C1["policy_query"] & C2["clause_search"] & C3["claim_eligibility"] & C4["claim_tracking"] & C5["premium_calc"] & C6["human_handoff"]
        end
        subgraph UW["UnderwritingAgent"]
            U1["clause_search"] & U2["human_handoff"]
        end
        subgraph SVC["ServiceAgent"]
            S1["policy_query"] & S2["human_handoff"]
        end
    end

    EXEC_N --> CHECK_N["check_node<br/>结果充足?"]
    CHECK_N -->|"不足 & iter<3"| PLAN_N
    CHECK_N -->|"充足"| SYN_N["synthesize_node<br/>领域SystemPrompt+合规约束→答案"]

    SYN_N --> COMP_N["合规检查 (Agent级)"]
    COMP_N --> RETURN["SubAgentResult<br/>{content, confidence, status, disclaimer}"]

    style PLAN_DETAIL fill:#e3f2fd
    style TOOLS fill:#fff3e0
    style RETURN fill:#4caf50,color:#fff
```

## 图5: RAG管道详细流程 (12个阶段)

```mermaid
flowchart TD
    Q["脱敏query (Spring转发)"] --> S1["阶段1: FAQ缓存检查<br/>L1 Redis → L2 MySQL<br/>精确匹配, <1ms"]

    S1 -->|命中| CACHE_HIT["直接返回缓存答案"]
    S1 -->|未命中| S2["阶段2: 领域边界检查<br/>白名单(60+保险词) vs 黑名单"]

    S2 -->|越界| OOD["Out-of-Domain响应"]
    S2 -->|通过| S3["阶段3: 3层意图分类<br/>关键词→BERT→LLM<br/>9类意图 + 实体提取"]

    S3 --> S4["阶段4: 查询结果缓存<br/>Key: query+intent, TTL: 10min"]

    S4 -->|命中| S4_HIT["返回缓存答案"]
    S4 -->|未命中| S5{意图类型}

    S5 -->|闲聊| S6["阶段5: LLM直接回复"]
    S5 -->|投诉建议| S7["阶段6: 转人工处理"]
    S5 -->|业务咨询| S8["阶段7: 策略选择<br/>DIRECT/HYDE/SUB_QUERY/COMPARE/CONDITIONAL/FALLBACK"]

    S8 --> S9["阶段8: Milvus混合检索"]

    subgraph MILVUS["Milvus混合检索流程"]
        S9_1["Dense检索<br/>BGE-M3 1024维向量<br/>IVF_FLAT索引"] --> S9_3["RRF融合排序"]
        S9_2["Sparse检索<br/>BM25关键词<br/>SPARSE_INVERTED_INDEX"] --> S9_3
        S9_3 --> S9_4["Parent-Child扩充"]
        S9_4 --> S9_5["Top-30粗排结果"]
    end

    S9_5 --> S10["Reranker: bge-reranker-large<br/>Cross-Encoder精排<br/>Top-30 → Top-5"]

    S10 --> S11["阶段9: 检索质量检查<br/>Top-1分数/平均分/有效chunk数<br/>按意图差异化阈值"]

    S11 -->|不合格| FALLBACK["低质量降级回复"]
    S11 -->|合格| S12["阶段10: LLM答案生成<br/>DeepSeek(主) + Qwen(备)<br/>CircuitBreaker保护"]

    S12 --> S13["阶段11: 合规检查<br/>5项: 医疗建议|禁用词|贬低竞品|金额引用|免责声明"]

    S13 --> S14["阶段12: 缓存回写<br/>结果写入Redis, TTL: 10min"]

    S6 --> S14
    S14 --> RESULT["RAGResponse<br/>{answer, intent, sources, strategy}<br/>→ Spring 还原 PII"]

    style MILVUS fill:#fff3e0
    style S9_5 fill:#ffe0b2
    style S12 fill:#e3f2fd
```

## 图6: 文档摄入管道

```mermaid
flowchart TD
    START_ING(("📄 文档摄入")) --> SOURCE{"文档来源"}

    SOURCE -->|"API上传"| API_U["POST /admin/ingest"]
    SOURCE -->|"SFTP同步"| SFTP["定时SFTP拉取"]
    SOURCE -->|"本地文件"| LOCAL["ingest_local_pdfs()"]

    API_U & SFTP & LOCAL --> ORCH["IngestionOrchestrator"]

    ORCH --> DEDUP["1. MD5去重检查"]

    DEDUP -->|已存在| SKIP["跳过"]
    DEDUP -->|新文档| PARSE["2. PDF解析<br/>PyMuPDF(文本) / marker-pdf(OCR)"]

    PARSE --> SECT["3. 章节提取 + 产品信息提取"]

    SECT --> SEG["4. BERT语义分段<br/>Damo文档分割器"]

    SEG --> CHUNK["5. Parent-Child分块<br/>chunk_size: 512, overlap: 64"]

    CHUNK --> ANNOTATE["6. 条款类型标注"]

    ANNOTATE --> ENCODE["7. BGE-M3批量编码<br/>Dense(1024) + Sparse"]

    ENCODE --> MILVUS_W["8. 写入Milvus<br/>Dense+Sparse双路索引"]

    ENCODE --> MYSQL_W["9. 写入MySQL"]

    MILVUS_W & MYSQL_W --> INVALID["10. 失效旧版本"]

    INVALID --> DONE_ING(("✅ 摄入完成"))

    style START_ING fill:#4caf50,color:#fff
    style DONE_ING fill:#4caf50,color:#fff
```

## 图7a: 长记忆 — MySQL 用户画像构建

```mermaid
flowchart TD
    START(("请求进入")) --> GW["FastAPI Gateway<br/>POST /chat<br/>Header: X-User-Id"]

    GW --> EXTRACT{"X-User-Id<br/>是否存在?"}

    EXTRACT -->|"不存在"| EMPTY["user_profile = {}<br/>不阻塞请求"]

    EXTRACT -->|"存在"| MM["MemoryManager<br/>.get_user_profile(user_id)"]

    MM --> POL_SQL["查询 policy_cache<br/>SELECT insurer, product_name,<br/>  status, sum_insured, premium,<br/>  effective_date, expire_date<br/>FROM policy_cache<br/>WHERE user_id = :uid<br/>  AND is_valid = TRUE<br/>ORDER BY effective_date DESC<br/>LIMIT 10"]

    POL_SQL --> POL{"查询成功?"}
    POL -->|"失败"| POL_ERR["日志 warn, policies=[]"]
    POL -->|"成功"| POL_OK["解析保单列表<br/>policies: [{insurer, product_name,<br/>  status, sum_insured, premium,<br/>  effective_date, expire_date}]"]

    POL_OK --> CLM_SQL["查询 claim_records<br/>SELECT report_no, status, stage,<br/>  submitted_at, estimated_days<br/>FROM claim_records<br/>WHERE user_id = :uid<br/>  AND is_valid = TRUE<br/>ORDER BY submitted_at DESC<br/>LIMIT 10"]

    POL_ERR --> CLM_SQL

    CLM_SQL --> CLM{"查询成功?"}
    CLM -->|"失败"| CLM_ERR["日志 warn, claims=[]"]
    CLM -->|"成功"| CLM_OK["解析理赔列表<br/>claims: [{report_no, status,<br/>  stage, submitted_at,<br/>  estimated_days}]"]

    CLM_ERR --> BUILD
    CLM_OK --> BUILD["构建 user_profile dict:<br/>{<br/>  user_id: str,<br/>  policies: [...],<br/>  claims: [...],<br/>  preferences: {},<br/>  has_data: bool,<br/>  queried_at: float<br/>}"]

    BUILD --> INJECT["注入 Orchestrator<br/>.process(user_profile=...)"]

    INJECT --> CONTEXT["注入 SubAgentTask<br/>.context = {<br/>  user_profile: {...}<br/>}"]

    CONTEXT --> PLAN_N["plan_node 读取:<br/>context.get('user_profile')<br/>格式化用户画像 Prompt"]

    EMPTY --> PLAN_N

    style START fill:#4caf50,color:#fff
    style POL_SQL fill:#e3f2fd
    style CLM_SQL fill:#e3f2fd
    style BUILD fill:#e8f5e9
    style PLAN_N fill:#e8eaf6
```

## 图7b: 短记忆 — RedisSaver Checkpoint 状态持久化

```mermaid
flowchart TD
    START_S(("SubAgent.invoke()")) --> INIT["构造 initial_state:<br/>{<br/>  user_query, intent, entities,<br/>  context, session_id,<br/>  messages: [],<br/>  iteration: 0<br/>}"]

    INIT --> CKPT_CHECK{"checkpointer<br/>是否可用?"}

    CKPT_CHECK -->|"是 (RedisSaver)" | CONFIG["构建 invoke config:<br/>config = {<br/>  configurable: {<br/>    thread_id: session_id<br/>  }<br/>}"]

    CONFIG --> INVOKE["graph.invoke(<br/>  input=initial_state,<br/>  config=config<br/>)"]

    INVOKE --> LANGGRAPH["LangGraph 内部流程:<br/>① 查 Redis checkpoint<br/>   key: thread_id={session_id}<br/>② 恢复 messages 历史对话<br/>   (add_messages annotator<br/>    自动合并新旧消息)<br/>③ 执行节点: plan→exec→<br/>   check→synthesize<br/>④ 每步自动写入 checkpoint<br/>   TTL: 30min"]

    CKPT_CHECK -->|"否 (降级)"| INVOKE_NOOP["graph.invoke(<br/>  input=initial_state<br/>)<br/>无状态模式"]

    INVOKE_NOOP --> NOOP_RESULT["每次全新会话<br/>messages 始终为空"]

    LANGGRAPH --> NODE_FLOW

    subgraph NODE_FLOW["节点状态流转"]
        P["plan_node<br/>① 读取 messages 历史<br/>② 读取 user_profile 画像<br/>③ 注入 Prompt 上下文<br/>④ 生成工具调用计划"]
        P --> E["exec_node<br/>按计划执行工具<br/>写入 tool_results"]
        E --> C["check_node<br/>检查结果充足性<br/>不足则 iteration+1"]
        C -->|"继续"| P
        C -->|"完成"| S["synthesize_node<br/>① 读取 messages 历史<br/>② 整合 tool_results<br/>③ 生成最终答案<br/>④ 写入 final_answer"]
    end

    S --> SAVE["LangGraph 自动持久化到 Redis:<br/>state channel → Redis hash<br/>  messages → 追加本轮 user+assistant<br/>  final_answer → 最新答案<br/>TTL 刷新为 30min"]

    SAVE --> RESULT(("返回 SubAgentResult"))

    NOOP_RESULT --> RESULT

    style START_S fill:#4caf50,color:#fff
    style RESULT fill:#4caf50,color:#fff
    style CKPT_CHECK fill:#fff3e0
    style LANGGRAPH fill:#e8eaf6
    style NODE_FLOW fill:#e3f2fd
    style SAVE fill:#e8f5e9
```

## 图7c: 记忆时序 — 多轮对话端到端流转

```mermaid
sequenceDiagram
    actor U as 用户
    participant SG as Spring网关
    participant GW as FastAPI Gateway
    participant MM as MemoryManager
    participant OR as Orchestrator
    participant SA as SubAgent
    participant RS as RedisSaver
    participant DB as MySQL
    participant LLM as DeepSeek API

    Note over U,LLM: === 第1轮: 用户首次咨询 ===

    U->>SG: "我的保单有哪些"
    SG->>SG: PII脱敏 + 提取 X-User-Id: U12345
    SG->>GW: POST /chat (query, session_id=s1, X-User-Id)

    GW->>MM: get_user_profile("U12345")
    MM->>DB: SELECT policy_cache WHERE user_id='U12345'
    DB-->>MM: [{尊享e生2024, 平安, 有效}, {重疾保, 众安, 有效}]
    MM->>DB: SELECT claim_records WHERE user_id='U12345'
    DB-->>MM: []
    MM-->>GW: user_profile = {policies: [...], claims: []}

    GW->>OR: process(query, user_profile, session_id=s1)
    OR->>SA: invoke(task, config={thread_id: s1})
    SA->>RS: 查 checkpoint(thread_id=s1)
    RS-->>SA: null (首轮无历史)
    SA->>LLM: plan + synthesize
    LLM-->>SA: "您有2份有效保单: 尊享e生2024(平安)、重疾保(众安)"
    SA->>RS: 写入 checkpoint(thread_id=s1, messages=[user, assistant])
    SA-->>OR: SubAgentResult
    OR-->>GW: answer
    GW-->>SG: JSON
    SG-->>U: 还原PII → 用户看到答案

    Note over U,LLM: === 第2轮: 上下文追问 (间隔5分钟) ===

    U->>SG: "那个医疗险免赔额多少"
    SG->>GW: POST /chat (query, session_id=s1, X-User-Id)

    GW->>MM: get_user_profile("U12345")
    MM->>DB: SELECT policy_cache WHERE user_id='U12345'
    DB-->>MM: [{尊享e生2024, 平安, 有效}, {重疾保, 众安, 有效}]
    MM-->>GW: user_profile (同上轮)

    GW->>OR: process(query, user_profile, session_id=s1)
    OR->>SA: invoke(task, config={thread_id: s1})
    SA->>RS: 查 checkpoint(thread_id=s1)
    RS-->>SA: messages = [{user: "我的保单有哪些"}, {assistant: "您有2份..."}]

    Note over SA: plan_node 注入历史:
    Note over SA: "用户有: 尊享e生2024(平安)+重疾保(众安)"
    Note over SA: "上轮问保单列表, 本轮追问免赔额"
    Note over SA: → 自动匹配到尊享e生2024

    SA->>LLM: plan + synthesize (含历史上下文)
    LLM-->>SA: "尊享e生2024的免赔额为1万元/年..."
    SA->>RS: 追加写入 checkpoint(thread_id=s1)
    SA-->>OR: SubAgentResult
    OR-->>GW: answer
    GW-->>SG: JSON
    SG-->>U: "尊享e生2024的免赔额为1万元/年"

    Note over U,LLM: === 第3轮: 省略主语的延续追问 ===

    U->>SG: "能赔吗"
    SG->>GW: POST /chat (query, session_id=s1, X-User-Id)

    GW->>MM: get_user_profile("U12345")
    MM-->>GW: user_profile

    GW->>OR: process(query, session_id=s1, user_profile)
    OR->>SA: invoke(task, config={thread_id: s1})
    SA->>RS: 查 checkpoint(thread_id=s1)
    RS-->>SA: messages = [{保单查询对话}, {免赔额对话}]

    Note over SA: 结合2轮历史 + 用户画像:
    Note over SA: "用户在问尊享e生2024的理赔"
    Note over SA: "免赔额1万 → 需判断花费是否超免赔"

    SA->>LLM: plan → clause_search + claim_eligibility
    LLM-->>SA: "尊享e生2024住院可赔, 但需超过1万免赔额..."
    SA->>RS: 追加写入 checkpoint(thread_id=s1, TTL 续期)
    SA-->>OR: SubAgentResult
    OR-->>GW: answer
    GW-->>SG: JSON
    SG-->>U: "住院可赔, 但需超过1万免赔额..."

    Note over RS: 30min 内无新请求 → checkpoint 自动过期清除
```

## 图8: 系统架构与外部依赖

```mermaid
flowchart TB
    subgraph 网关层["☕ Spring 网关层"]
        SG["Spring Gateway<br/>PII 脱敏 + PII 还原<br/>X-User-Id 注入"]
    end

    subgraph 应用层["🖥️ Python 应用层"]
        GW["FastAPI Gateway<br/>POST /chat | GET /chat/stream"]
        MM["MemoryManager<br/>RedisSaver + MySQL 画像"]
        ORCH_G["Orchestrator<br/>纯多Agent调度中心"]
        RAG_G["RAGSystem<br/>12阶段RAG降级管道"]
        INGEST["IngestionOrchestrator"]
    end

    subgraph 智能体层["🤖 多Agent层 (4个领域Agent)"]
        INS["InsuranceAgent<br/>4工具 (DeepSeek-V3)"]
        CLAIM["ClaimAgent<br/>6工具 (DeepSeek-R1)"]
        UW["UnderwritingAgent<br/>2工具 (DeepSeek-R1)"]
        SVC["ServiceAgent<br/>2工具 (DeepSeek-V3)"]
        BASE_A["SubAgent ABC<br/>Plan-Execute 自包含"]
    end

    subgraph RAG核心["📚 RAG核心层"]
        QC2["QueryClassifier<br/>3层意图路由"]
        VS["VectorStore<br/>Milvus混合检索"]
        STRATEGY["StrategySelector<br/>6种检索策略"]
        GUARDS["安全守卫<br/>Domain/Quality/Compliance"]
    end

    subgraph 基础层["⚙️ 基础层"]
        LLM["LLMClient<br/>DeepSeek+Qwen<br/>CircuitBreaker"]
        ENC["BGEM3Encoder<br/>BGE-M3 1024维"]
        RERANK["Reranker<br/>bge-reranker-large"]
        DB["MySQL连接池"]
    end

    subgraph 缓存层["💾 缓存层"]
        REDIS["Redis<br/>FAQ/Query/Embedding/Product"]
        CG["CacheGuard<br/>防穿透/击穿/雪崩"]
    end

    subgraph 外部服务["🌐 外部服务"]
        DS["DeepSeek API"]
        QWEN["Qwen DashScope"]
        MILVUS_S["Milvus :19530"]
        MYSQL_S["MySQL :3306"]
        REDIS_S["Redis :6379"]
    end

    SG --> GW
    GW --> MM
    MM --> DB
    MM --> REDIS
    GW --> ORCH_G
    GW --> RAG_G
    GW --> INGEST
    ORCH_G --> INS & CLAIM & UW & SVC
    INS & CLAIM & UW & SVC --> BASE_A
    INS & CLAIM & UW & SVC --> LLM
    RAG_G --> VS
    RAG_G --> QC2
    RAG_G --> STRATEGY
    RAG_G --> GUARDS
    VS --> ENC
    VS --> RERANK
    BASE_A --> DB
    BASE_A --> REDIS
    ENC & RERANK & LLM --> DS & QWEN
    VS --> MILVUS_S
    DB --> MYSQL_S
    REDIS --> REDIS_S

    style 网关层 fill:#fff3e0
    style 应用层 fill:#e1f5fe
    style 智能体层 fill:#e8f5e9
    style RAG核心 fill:#fce4ec
    style 基础层 fill:#fff3e0
    style 缓存层 fill:#f3e5f5
    style 外部服务 fill:#e0e0e0
```

## 图9: 安全与合规管道 (Python侧)

```mermaid
flowchart TD
    INPUT_Q["脱敏query (Spring转发)"] --> DG_IN["领域边界守卫<br/>白名单(60+保险词)<br/>黑名单过滤"]

    DG_IN -->|通过| PROCESS["多Agent处理 或 RAG处理"]
    DG_IN -->|越界| REJECT["返回越界提示"]

    PROCESS --> ANSWER["生成答案"]

    ANSWER --> COMPLIANCE["合规检查"]

    subgraph 合规规则["5项合规规则"]
        COMPLIANCE --> R1["① 医疗建议检测 → 追加免责声明"]
        COMPLIANCE --> R2["② 监管禁用词 → 自动替换"]
        COMPLIANCE --> R3["③ 贬低竞品检测 → 自动修正"]
        COMPLIANCE --> R4["④ 金额引用校验 → 追加数据来源"]
        COMPLIANCE --> R5["⑤ 确定性承诺替换 → '以审核为准'"]
    end

    R1 & R2 & R3 & R4 & R5 --> OUTPUT_PY["Python 输出 → Spring PII 还原"]

    OUTPUT_PY --> SPRING_R["Spring 还原层<br/>[NAME_001]→张三<br/>[PHONE_001]→138xxxx"]

    SPRING_R --> FINAL["最终输出给用户"]

    style 合规规则 fill:#fce4ec
    style REJECT fill:#ffcdd2
    style FINAL fill:#4caf50,color:#fff
```

## 图10: 意图→多Agent路由对照表

```mermaid
flowchart LR
    subgraph 投保域["投保相关"]
        I1["产品咨询"] --> A1["InsuranceAgent + ServiceAgent"]
        I2["产品对比"] --> A2["InsuranceAgent + ServiceAgent"]
        I3["保费试算"] --> A3["InsuranceAgent + ServiceAgent"]
        I4["投保流程"] --> A4["InsuranceAgent + UnderwritingAgent"]
    end

    subgraph 核保域["核保相关"]
        I5["核保咨询"] --> A5["UnderwritingAgent + InsuranceAgent"]
    end

    subgraph 理赔域["理赔相关"]
        I6["理赔咨询"] --> A6["ClaimAgent + UnderwritingAgent"]
        I7["理赔进度"] --> A7["ClaimAgent + ServiceAgent"]
        I8["条款解读"] --> A8["ClaimAgent + InsuranceAgent"]
    end

    subgraph 客服域["客服相关"]
        I9["保单查询"] --> A9["ServiceAgent + InsuranceAgent"]
    end

    subgraph 降级["RAG降级"]
        I10["投诉建议"] --> R1["RAGSystem"]
        I11["闲聊"] --> R2["RAGSystem"]
    end

    style 投保域 fill:#e3f2fd
    style 核保域 fill:#e8f5e9
    style 理赔域 fill:#fff3e0
    style 客服域 fill:#f3e5f5
    style 降级 fill:#fce4ec
```

---

## 变更记录

| 版本 | 变更 |
|------|------|
| v3.0 | 删除 `agent/graph.py`，纯多Agent架构，统一 multi_agent 路由 |
| v3.1 | PII 脱敏/还原移至 Spring 网关，RAG 管道 14→12 阶段 |
| v3.2 | 新增长短记忆: RedisSaver Checkpoint + MySQL 用户画像，支持多轮对话 |

## 关键文件索引

| 模块 | 核心文件 | 职责 |
|------|---------|------|
| 入口 | `gateway/app.py` | FastAPI, X-User-Id 提取, user_profile 构建 |
| 记忆 | `agent/memory.py` | MemoryManager: RedisSaver checkpointer + MySQL 用户画像 |
| 编排 | `agent/orchestrator.py` | 纯多Agent调度中心, user_profile 注入 |
| 智能体 | `agent/sub_agents/*.py` | 4个领域Agent, Plan-Execute + RedisSaver 持久化 |
| 基类 | `agent/sub_agents/base.py` | SubAgent ABC, checkpoint 集成, 对话历史注入 |
| 工具 | `agent/tools/all_tools.py` | 7个LangChain工具, 按领域分配 |
| 状态 | `agent/state.py` | AgentState/SubAgentTask/OrchestratorState |
| RAG | `rag_qa/core/rag_system.py` | **12阶段** RAG降级管道 |
| 向量库 | `rag_qa/core/vector_store.py` | Milvus混合检索 |
| 分类器 | `rag_qa/core/query_classifier.py` | 3层意图路由 |
| 编码器 | `base/encoder.py` | BGE-M3, Dense(1024)+Sparse |
| 排序器 | `base/reranker.py` | bge-reranker-large |
| LLM | `base/llm_client.py` | DeepSeek+Qwen, CircuitBreaker |
| 摄入 | `rag_qa/ingestion/ingestion_orchestrator.py` | PDF→分块→编码→入库 |
| 缓存 | `cache/*.py` | Redis三级缓存 |
| 配置 | `config/settings.py` | 全局配置单例 |
