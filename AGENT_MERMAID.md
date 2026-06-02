# 保险 Agent 智能体系统 — Mermaid 架构图集

---

## 1. Agent 整体流程框架

```mermaid
flowchart TD
    U["用户输入<br/>'我的平安e生保上个月肺炎住院能赔吗？<br/>帮我算一下如果加保门诊险要多少钱'"] --> PII

    PII["PII 脱敏层<br/>正则 + NER 双层"] --> ROUTER

    subgraph ROUTER["意图路由"]
        direction LR
        R1["规则匹配"]
        R2["BERT 分类<br/>9分类 + OOD"]
        R3["LLM 兜底"]
    end

    ROUTER --> DECISION{"意图类型?"}

    DECISION -->|闲聊| GREET["LLM 直答<br/>不进入 Agent"]
    DECISION -->|单步查询| RAG["RAG Pipeline<br/>检索 → 生成"]
    DECISION -->|复杂任务| AGENT

    subgraph AGENT["Agent 引擎 LangGraph"]
        direction TB

        PLAN["🧠 Planner<br/>任务分解 + 工具选择"]
        EXEC["🔧 Executor<br/>工具调用执行"]
        MEM["💾 Checkpoint<br/>会话状态持久化"]
        REFLECT["🪞 Reflector<br/>结果校验 + 修正"]

        PLAN -->|"工具调用计划"| EXEC
        EXEC -->|"工具返回结果"| REFLECT
        REFLECT -->|"需要修正"| PLAN
        REFLECT -->|"通过"| SYNTHESIZE["📝 Synthesizer<br/>多工具结果整合"]
        SYNTHESIZE -->|"中间状态"| MEM
        MEM -.->|"恢复状态"| PLAN
    end

    AGENT --> COMPLY["合规守卫<br/>5道检查"] --> RESPONSE["SSE 流式返回用户"]
```

---

## 2. Planner-Executor 状态图

```mermaid
stateDiagram-v2
    [*] --> Idle

    Idle --> Planning: 用户 query 到达

    state Planning {
        [*] --> AnalyzeIntent: LLM 分析 query
        AnalyzeIntent --> DecomposeTasks: 拆解子任务
        DecomposeTasks --> SelectTools: 为每个子任务选工具
        SelectTools --> BuildPlan: 生成执行计划
    }

    Planning --> Executing: 计划就绪

    state Executing {
        [*] --> NextTool: 取下一个工具调用
        NextTool --> CallTool: 执行工具
        CallTool --> ValidateResult: 校验返回
        ValidateResult --> NextTool: 还有未执行工具
        ValidateResult --> Done: 全部完成
    }

    Executing --> Reflecting: 所有工具执行完毕

    state Reflecting {
        [*] --> CheckCompleteness: 检查是否回答完整
        CheckCompleteness --> Planning: 信息不足 → 补调工具
        CheckCompleteness --> Synthesize: 通过
        Synthesize --> [*]
    }

    Reflecting --> Responding: 合成最终答案

    Responding --> Idle: 流式返回用户
```

---

## 3. LangGraph 节点与边

```mermaid
graph LR
    START["__start__"] --> PLAN_NODE

    subgraph PLAN_NODE["plan 节点"]
        P1["LLM 推理<br/>DeepSeek-R1"]
        P2["输出: 工具列表 + 参数"]
        P3["状态写入<br/>plan_step"]
    end

    PLAN_NODE --> EXEC_NODE

    subgraph EXEC_NODE["exec 节点"]
        E1["ToolNode<br/>并行/串行调用"]
        E2["保单查询工具"]
        E3["理赔资格预检工具"]
        E4["条款检索工具"]
        E5["保费试算工具"]
        E6["结果收集"]
    end

    EXEC_NODE --> REFLECT_NODE

    subgraph REFLECT_NODE["reflect 节点"]
        R1["LLM 判断<br/>是否还需要更多信息?"]
        R2["需要 → 回 plan"]
        R3["不需要 → 过"]
    end

    REFLECT_NODE -->|"继续"| PLAN_NODE
    REFLECT_NODE -->|"完成"| SYNTH_NODE

    subgraph SYNTH_NODE["synthesize 节点"]
        S1["整合所有工具结果"]
        S2["格式化输出"]
        S3["合规检查 + 溯源"]
    end

    SYNTH_NODE --> END_NODE["__end__"]

    style PLAN_NODE fill:#e3f2fd
    style EXEC_NODE fill:#e8f5e9
    style REFLECT_NODE fill:#fff3e0
    style SYNTH_NODE fill:#fce4ec
```

---

## 4. 7个工具定义

```mermaid
flowchart TD
    TOOLS["7 个 Agent 工具"] --> T1
    TOOLS --> T2
    TOOLS --> T3
    TOOLS --> T4
    TOOLS --> T5
    TOOLS --> T6
    TOOLS --> T7

    T1["🔍 保单查询<br/>─────────<br/>输入: 保司 + 产品 + 证件号后4位<br/>输出: 保单状态/保额/保障期限<br/>依赖: MySQL 保单缓存"]
    T2["🏥 理赔资格预检<br/>─────────<br/>输入: 保单号 + 疾病/治疗<br/>输出: 是否在保障范围内 + 条款引用<br/>依赖: RAG 检索 + 规则引擎"]
    T3["📋 条款检索<br/>─────────<br/>输入: 产品名 + 关键词<br/>输出: 条款原文 + 释义<br/>依赖: Milvus 向量检索"]
    T4["💰 保费试算<br/>─────────<br/>输入: 产品 + 年龄 + 保额 + 附加险<br/>输出: 年保费 + 缴费方式<br/>依赖: MySQL 费率表 + 计算引擎"]
    T5["⚖️ 多产品对比<br/>─────────<br/>输入: 产品A + 产品B + 对比维度<br/>输出: 表格对比 + 推荐<br/>依赖: RAG 对比检索 + LLM"]
    T6["📊 理赔追踪<br/>─────────<br/>输入: 报案号<br/>输出: 理赔阶段/预计时效/补充材料<br/>依赖: MySQL 理赔记录"]
    T7["👤 人工转接<br/>─────────<br/>输入: 会话上下文<br/>输出: 转接工单 + 对话摘要<br/>依赖: 客服系统 API"]
```

---

## 5. 复杂多工具编排示例

```mermaid
sequenceDiagram
    actor U as 用户
    participant A as Agent 引擎
    participant P as Planner
    participant E as Executor
    participant T1 as 保单查询
    participant T3 as 条款检索
    participant T2 as 理赔资格预检
    participant T4 as 保费试算
    participant S as Synthesizer

    U->>A: "我的平安e生保肺炎住院能赔吗？<br/>顺便帮我看看加保门诊险要多少钱"

    rect rgb(227, 242, 253)
        Note over P: 阶段1: Planner 拆解
        P->>P: LLM 分析<br/>→ 子任务1: 查保单(确定是否有e生保)<br/>→ 子任务2: 查条款(肺炎赔付条件)<br/>→ 子任务3: 理赔资格预检<br/>→ 子任务4: 保费试算(门诊险)
    end

    rect rgb(232, 245, 233)
        Note over E: 阶段2: Executor 串行调用

        E->>T1: call 保单查询(平安e生保)
        T1-->>E: 保单状态: 有效, 保额50万

        E->>T3: call 条款检索(e生保, 肺炎+住院)
        T3-->>E: 条款2.3: 住院医疗含肺炎, 免赔额1万

        E->>T2: call 理赔资格预检(保单+肺炎住院)
        T2-->>E: ✅ 符合, 预计赔付=(费用-1万)×比例

        Note over E: 子任务4 不依赖前三者<br/>可以并行
        E->>T4: call 保费试算(门诊险, 30岁, 附加)
        T4-->>E: 年保费 580元
    end

    rect rgb(252, 228, 236)
        Note over S: 阶段3: Synthesizer 整合

        S->>S: 整合结果<br/>1. 保单确认 ✅<br/>2. 理赔条件清晰<br/>3. 本次住院可赔<br/>4. 门诊险报价

        S->>S: 合规检查<br/>✅ 有条款引用<br/>✅ 无医疗建议<br/>✅ 无贬低竞品
    end

    S-->>U: "您的平安e生保保障有效。<br/>肺炎住院属于保障范围，<br/>超过1万免赔额部分可按比例赔付。<br/>加保门诊险年保费580元。"
```

---

## 6. Checkpoint 双层记忆系统

```mermaid
flowchart TD
    subgraph SHORT["短期记忆 当前会话"]
        direction TB
        S1["当前 query"]
        S2["上轮意图"]
        S3["上轮实体 保司/产品"]
        S4["已执行工具 + 结果"]
        S5["已生成的部分回答"]
    end

    subgraph LONG["长期记忆 LangGraph Checkpoint"]
        direction TB
        L1["用户画像 年龄/性别/偏好"]
        L2["历史保单列表"]
        L3["最近 N 次咨询摘要"]
        L4["理赔历史"]
        L5["用户行为模式 高频问题/时段"]
    end

    subgraph CHECKPOINT["Checkpoint 机制"]
        direction TB
        C1["每个 Agent 节点执行后<br/>自动保存状态快照"]
        C2["状态 key: thread_id + step"]
        C3["中断恢复<br/>用户追加信息 → 从断点继续"]
        C4["回溯<br/>plan 修正 → 回到上一个快照"]
    end

    SHORT -.->|"会话结束<br/>摘要沉淀"| LONG
    SHORT --> CHECKPOINT
    LONG -->|"查询用户画像<br/>补充实体信息"| SHORT
    LONG --> CHECKPOINT
```

---

## 7. 多轮对话状态流转

```mermaid
sequenceDiagram
    actor U as 用户
    participant CK as Checkpoint
    participant PT as Planner
    participant EX as Executor

    Note over U,EX: === 第1轮 ===
    U->>PT: "平安e生保怎么样？"
    PT->>CK: 保存状态 thread_id=abc, step=1

    rect rgb(227, 242, 253)
        Note over PT: planner输出<br/>→ 工具: 条款检索(e生保, 保障责任)
    end

    EX->>CK: 保存状态 step=2<br/>工具结果已缓存
    EX-->>U: "平安e生保是百万医疗险，<br/>保额最高400万，涵盖住院/门诊..."

    Note over U,EX: === 第2轮 指代消解 ===
    U->>PT: "那它免赔额多少？"
    PT->>CK: 加载状态 thread_id=abc

    rect rgb(232, 245, 233)
        Note over PT: 从 Checkpoint 恢复<br/>"它" → 平安e生保<br/>实体自动补全
    end
    PT->>EX: 工具: 条款检索(e生保, 免赔额)
    EX-->>U: "平安e生保一般医疗免赔额为1万元..."

    Note over U,EX: === 第3轮 用户打断/追加 ===
    U->>PT: "等等，帮我也看看众安尊享e生"
    PT->>CK: 当前状态已保存

    rect rgb(255, 243, 224)
        Note over PT: 用户追加新需求<br/>不丢失之前 e生保 对话上下文<br/>追加对比任务
    end
    PT->>EX: 工具: 条款检索(尊享e生, 保障责任)
    EX-->>U: "众安尊享e生也是百万医疗险，<br/>与平安e生保相比..."
```

---

## 8. 工具调用并行/串行决策

```mermaid
flowchart TD
    PLAN["Planner 生成工具调用列表"] --> ANALYZE

    ANALYZE["依赖分析"]

    ANALYZE -->|"工具B依赖工具A的输出"| SERIAL
    ANALYZE -->|"工具之间无依赖"| PARALLEL

    subgraph SERIAL["串行执行"]
        direction TB
        S1["Tool A 执行"]
        S2["等待 A 返回"]
        S3["Tool B 使用 A 的结果"]
        S4["Tool C 使用 B 的结果"]
        S1 --> S2 --> S3 --> S4
    end

    subgraph PARALLEL["并行执行"]
        direction LR
        P1["Tool A"]
        P2["Tool B"]
        P3["Tool C"]
    end

    SERIAL --> MERGE["结果合并<br/>按计划顺序组装"]
    PARALLEL --> MERGE

    MERGE --> CHECK{"所有工具<br/>执行成功?"}
    CHECK -->|是| NEXT["进入 Reflector"]
    CHECK -->|部分失败| RETRY["最多重试 2 次<br/>失败工具降级处理"]

    style SERIAL fill:#fff3e0
    style PARALLEL fill:#e8f5e9
```

---

## 9. Agent 故障降级链路

```mermaid
flowchart TD
    QUERY["复杂 query 到达"] --> AGENT_TRY{"Agent 引擎<br/>正常?"}

    AGENT_TRY -->|正常| FULL_AGENT["完整 Planner-Executor 流程"]
    AGENT_TRY -->|超时 10s| DEGRADE

    FULL_AGENT --> RESULT{"结果质量?"}
    RESULT -->|通过| ANSWER["返回答案"]
    RESULT -->|工具调用全部失败| DEGRADE

    subgraph DEGRADE["降级策略"]
        direction TB
        D1["🟡 降级1: 跳过 Agent<br/>直接 RAG 检索 + LLM 生成"]
        D2["🟠 降级2: 跳过 RAG<br/>仅 LLM 基于知识回答"]
        D3["🔴 降级3: 规则回复<br/>'抱歉系统繁忙，请稍后再试'"]
    end

    D1 --> D1_CHECK{"检索结果?"}
    D1_CHECK -->|有结果| ANSWER
    D1_CHECK -->|无结果| D2

    D2 --> D2_CHECK{"LLM 可用?"}
    D2_CHECK -->|可用| ANSWER
    D2_CHECK -->|不可用| D3

    ANSWER --> LOG["记录降级路径<br/>用于监控告警"]

    style DEGRADE fill:#ffebee
```

---

## 10. Agent 与 RAG 的协作关系

```mermaid
flowchart LR
    subgraph RAG["RAG 系统 Phase 1"]
        direction TB
        R1["Milvus 向量检索"]
        R2["MySQL 元数据"]
        R3["6种检索策略"]
        R4["BGE-M3 编码"]
    end

    subgraph BRIDGE["桥接层"]
        direction TB
        B1["Tool 抽象封装<br/>LangChain Tool 接口"]
        B2["结果转换<br/>Milvus chunk → ToolOutput"]
        B3["共享 Schema<br/>同一套 Milvus Collection"]
    end

    subgraph AGENT["Agent 系统 Phase 2"]
        direction TB
        A1["Planner-Executor"]
        A2["7 个工具"]
        A3["Checkpoint 记忆"]
        A4["合规守卫"]
    end

    RAG -->|"条款检索工具"| BRIDGE
    BRIDGE -->|"标准 Tool 接口"| AGENT

    Note1["RAG 是基础设施<br/>Agent 是上层应用"] -.-> RAG
    Note2["Agent 通过 Tool 抽象<br/>复用 RAG 全部能力"] -.-> BRIDGE
```

---

## 11. 工具注册与 LangChain 集成

```mermaid
classDiagram
    class BaseTool {
        +name: str
        +description: str
        +args_schema: BaseModel
        +_run(**kwargs) str
        +_arun(**kwargs) str
    }

    class PolicyQueryTool {
        +name = "policy_query"
        +description = "查询用户保单信息"
        +_run(insurer, product, id_last4) PolicyResult
    }

    class ClaimEligibilityTool {
        +name = "claim_eligibility"
        +description = "预检理赔资格"
        +_run(policy_no, disease, treatment) EligibilityResult
    }

    class ClauseSearchTool {
        +name = "clause_search"
        +description = "检索保险条款"
        +_run(product, keywords) List~Clause~
    }

    class PremiumCalcTool {
        +name = "premium_calc"
        +description = "保费试算"
        +_run(product, age, sum_insured, riders) PremiumResult
    }

    class ProductCompareTool {
        +name = "product_compare"
        +description = "多产品对比"
        +_run(products, dimensions) CompareResult
    }

    class ClaimTrackingTool {
        +name = "claim_tracking"
        +description = "理赔进度追踪"
        +_run(report_no) TrackingResult
    }

    class HumanHandoffTool {
        +name = "human_handoff"
        +description = "转接人工客服"
        +_run(context) HandoffResult
    }

    BaseTool <|-- PolicyQueryTool
    BaseTool <|-- ClaimEligibilityTool
    BaseTool <|-- ClauseSearchTool
    BaseTool <|-- PremiumCalcTool
    BaseTool <|-- ProductCompareTool
    BaseTool <|-- ClaimTrackingTool
    BaseTool <|-- HumanHandoffTool

    ClauseSearchTool --> "复用" RagPipeline: 底层调用 Milvus + Reranker
    PolicyQueryTool --> "查询" MySQL: 保单缓存表
    PremiumCalcTool --> "查询" MySQL: 费率表
    ClaimTrackingTool --> "查询" MySQL: 理赔记录表
    HumanHandoffTool --> "调用" CSAPI: 客服系统 API
```

---

## 12. 端到端延迟分解 Agent 场景

```mermaid
gantt
    title Agent 复杂任务端到端延迟 ~2500ms (5个工具)
    dateFormat X
    axisFormat %s

    section 前置
    PII 脱敏             :pii, 0, 1
    意图路由              :router, 1, 6

    section Planner
    LLM 任务分解 DeepSeek-R1 :plan, 6, 506

    section Executor 工具调用
    保单查询 MySQL        :t1, 506, 556
    条款检索 Milvus       :t2, 556, 606
    理赔资格预检           :t3, 606, 706
    保费试算               :t4, 706, 756
    产品对比检索           :t5, 756, 856

    section 后置
    Reflector 校验         :reflect, 856, 956
    Synthesizer 整合       :synth, 956, 1756
    合规检查               :comply, 1756, 1758
    流式返回               :stream, 1758, 2500
```

---

## 使用说明

在支持 Mermaid 的编辑器中直接渲染，或粘贴到 https://mermaid.live 预览。

文件位置: `/home/newnew/code/code/pythonCode/python_learning/integrated_qa_system/AGENT_MERMAID.md`
