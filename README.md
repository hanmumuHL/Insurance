# 保险聚合平台 — RAG 智能客服 + Agent 智能体系统

## 项目定位

面向保险聚合场景（对接多家保险公司、200+ 产品）的智能客服 + Agent 系统。

- **Phase 1: RAG 智能客服** — 基于 Milvus + BGE-M3 的条款检索问答
- **Phase 2: Agent 智能体** — LangGraph Planner-Executor 多工具编排

## 技术栈

| 层级 | 技术 |
|------|------|
| LLM API | DeepSeek-V3 (主) + 通义千问 (降级) + DeepSeek-R1 (Agent 规划) |
| Embedding | BGE-M3 (本地, Dense+Sparse 双向量) |
| Reranker | bge-reranker-large (本地) |
| 意图分类 | BERT-base-Chinese (本地, 9分类含 OOD) |
| 向量库 | Milvus (IVF_FLAT + 稀疏倒排) |
| 元数据 | MySQL (产品/FAQ/保单缓存/费率) |
| 缓存 | Redis (FAQ/Embedding/产品/会话) |
| Agent | LangGraph + LangChain Tools |
| API | FastAPI + WebSocket SSE |

## 项目结构

```
Insurance/
├── config/            # 配置管理
├── base/              # 日志 + 公共组件
├── cache/             # 三级缓存层 (Redis + MySQL + Milvus 内置)
├── rag_qa/            # RAG 智能客服核心
│   ├── core/          # PII脱敏 / 领域守卫 / 意图路由 / 检索策略 / 向量库
│   └── ingestion/     # PDF 文档摄取管道
├── agent/             # Agent 智能体
│   └── tools/         # 7 个 LangChain Tool
├── gateway/           # FastAPI API 层
│   └── routes/        # 路由
└── tests/             # 单元测试
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 复制并修改配置
cp config/.env.example config/.env

# 启动
uvicorn gateway.app:app --host 0.0.0.0 --port 8000
```
