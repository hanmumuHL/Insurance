"""
全局配置管理 — 从 .env 加载，提供类型安全的访问
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

# 加载 .env
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int = 0) -> int:
    return int(os.getenv(key, str(default)))


def _env_bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).lower() in ("true", "1", "yes")


@dataclass
class LLMConfig:
    deepseek_api_key: str = field(default_factory=lambda: _env("DEEPSEEK_API_KEY"))
    deepseek_base_url: str = field(default_factory=lambda: _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    qwen_api_key: str = field(default_factory=lambda: _env("QWEN_API_KEY"))
    qwen_base_url: str = field(default_factory=lambda: _env("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    primary_model: str = "deepseek-chat"
    fallback_model: str = "qwen-max"
    planner_model: str = "deepseek-reasoner"   # DeepSeek-R1
    temperature: float = 0.3
    max_tokens: int = 2048


@dataclass
class MilvusConfig:
    host: str = field(default_factory=lambda: _env("MILVUS_HOST", "localhost"))
    port: int = field(default_factory=lambda: _env_int("MILVUS_PORT", 19530))
    collection: str = field(default_factory=lambda: _env("MILVUS_COLLECTION", "insurance_clauses"))
    uri: str = ""

    def __post_init__(self):
        self.uri = f"http://{self.host}:{self.port}"


@dataclass
class MySQLConfig:
    host: str = field(default_factory=lambda: _env("MYSQL_HOST", "localhost"))
    port: int = field(default_factory=lambda: _env_int("MYSQL_PORT", 3306))
    user: str = field(default_factory=lambda: _env("MYSQL_USER", "root"))
    password: str = field(default_factory=lambda: _env("MYSQL_PASSWORD"))
    database: str = field(default_factory=lambda: _env("MYSQL_DB", "insurance_platform"))
    pool_size: int = 10

    @property
    def url(self) -> str:
        return f"mysql+mysqlconnector://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class RedisConfig:
    host: str = field(default_factory=lambda: _env("REDIS_HOST", "localhost"))
    port: int = field(default_factory=lambda: _env_int("REDIS_PORT", 6379))
    password: str = field(default_factory=lambda: _env("REDIS_PASSWORD"))
    db: int = field(default_factory=lambda: _env_int("REDIS_DB", 0))

    @property
    def url(self) -> str:
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


@dataclass
class Neo4jConfig:
    uri: str = field(default_factory=lambda: _env("NEO4J_URI", "bolt://localhost:7687"))
    user: str = field(default_factory=lambda: _env("NEO4J_USER", "neo4j"))
    password: str = field(default_factory=lambda: _env("NEO4J_PASSWORD", "neo4j123"))


@dataclass
class ModelPaths:
    bge_m3: str = field(default_factory=lambda: _env("BGE_M3_MODEL_PATH", "models/bge-m3"))
    reranker: str = field(default_factory=lambda: _env("RERANKER_MODEL_PATH", "models/bge-reranker-large"))
    bert_classifier: str = field(default_factory=lambda: _env("BERT_CLASSIFIER_PATH", "models/bert_intent"))
    bert_model_dir: str = field(default_factory=lambda: _env("BERT_MODEL_DIR", "models/bert_intent"))
    bert_segmenter: str = field(default_factory=lambda: _env("BERT_SEGMENTER_PATH", "models/nlp_bert_document-segmentation_chinese-base"))


@dataclass
class AuthConfig:
    """双通道认证配置"""
    auth_enabled: bool = field(default_factory=lambda: _env_bool("AUTH_ENABLED", True))
    jwt_secret: str = field(default_factory=lambda: _env("JWT_SECRET", ""))
    default_role: str = "agent"  # 未认证时的默认角色（向后兼容）


@dataclass
class Settings:
    llm: LLMConfig = field(default_factory=LLMConfig)
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    mysql: MySQLConfig = field(default_factory=MySQLConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    models: ModelPaths = field(default_factory=ModelPaths)
    auth: AuthConfig = field(default_factory=AuthConfig)

    # RAG 参数
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_retrieve: int = 30
    top_k_rerank: int = 5
    min_similarity: float = 0.55

    # Agent 参数
    agent_max_iterations: int = 5
    agent_timeout_seconds: int = 30

    # 缓存参数
    faq_cache_ttl: int = 3600
    embedding_cache_ttl: int = 86400
    product_cache_ttl: int = 3600
    query_result_cache_ttl: int = 600


# 单例
settings = Settings()
