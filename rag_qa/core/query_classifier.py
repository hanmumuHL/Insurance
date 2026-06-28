"""
Query Router — 三层意图路由
  第1层: 关键词规则 (0延迟, 命中~60%)
  第2层: BERT 分类 (5ms, 命中~35%)
  第3层: LLM 兜底 (200ms, 命中~5%)

  9 分类: 8 个保险意图 + 1 个 out_of_domain
"""
import re
from dataclasses import dataclass, field
from base.logger import logger


@dataclass
class IntentResult:
    intent: str
    confidence: float
    entities: dict = field(default_factory=dict)
    reject: bool = False
    reject_reason: str = ""
    fallback_response: str = ""
    source: str = ""  # "rule" / "bert" / "llm"


class QueryClassifier:
    """三层意图路由"""

    INTENT_LABELS = [
        "闲聊寒暄", "条款解读", "保单查询", "理赔咨询",
        "产品对比", "保费试算", "退保咨询", "投诉建议",
        "out_of_domain",
    ]

    # ── 第1层: 关键词规则 ──
    RULE_MAP = {
        "条款解读": [
            "条款", "保障范围", "保什么", "不保什么", "免责",
            "等待期", "犹豫期", "免赔额", "赔付比例", "报销范围",
            "续保条件", "健康告知",
        ],
        "保单查询": [
            "我的保单", "保单号", "保单状态", "查看保单",
            "查保单", "保单信息",
        ],
        "理赔咨询": [
            "理赔", "报销", "赔付", "报案", "理赔进度",
            "理赔材料", "怎么赔", "能赔吗", "赔多少",
        ],
        "产品对比": [
            "对比", "比较", "哪个好", "哪个划算", "区别",
            "vs", "PK",
        ],
        "保费试算": [
            "多少钱", "保费", "费用", "价格", "试算",
            "一年多少", "每月多少",
        ],
        "退保咨询": [
            "退保", "退保险", "不想保了", "取消",
            "退多少钱", "现金价值",
        ],
        "投诉建议": [
            "投诉", "举报", "不满意", "态度差",
            "建议", "反馈",
        ],
    }

    # 各意图的最低置信度阈值
    CONFIDENCE_THRESHOLDS = {
        "条款解读": 0.50,
        "保单查询": 0.60,
        "理赔咨询": 0.50,
        "产品对比": 0.55,
        "保费试算": 0.60,
        "退保咨询": 0.55,
        "投诉建议": 0.70,
        "闲聊寒暄": 0.70,
        "out_of_domain": 0.40,
    }

    OOD_RESPONSE = (
        "抱歉，我是保险智能客服，暂时无法回答这个问题。\n"
        "您可以咨询：保单查询、理赔流程、产品对比、保费试算等保险相关问题。"
    )

    LOW_CONF_RESPONSE = (
        "我不太确定您的问题类型。\n"
        "请问您是想咨询：1) 保单条款  2) 理赔流程  3) 产品对比  4) 保费试算？"
    )

    def __init__(self, bert_model=None, llm_client=None):
        """
        bert_model: 可选，BERT 分类器实例
        llm_client: 可选，LLM API 客户端
        """
        self.bert_model = bert_model
        self.llm_client = llm_client

    def classify(self, query: str) -> IntentResult:
        """三层路由分类"""

        # ── 第1层: 关键词规则 ──
        result = self._rule_classify(query)
        if result is not None:
            return result

        # ── 第2层: BERT 分类 ──
        if self.bert_model is not None:
            result = self._bert_classify(query)
            if result is not None:
                return result

        # ── 第3层: LLM 兜底 ──
        if self.llm_client is not None:
            result = self._llm_classify(query)
            if result is not None:
                return result

        # ── 全部未命中 → 默认闲聊 ──
        return IntentResult(
            intent="闲聊寒暄",
            confidence=0.3,
            source="fallback",
        )

    def _rule_classify(self, query: str) -> IntentResult | None:
        """第1层: 关键词规则匹配"""
        for intent, keywords in self.RULE_MAP.items():
            for kw in keywords:
                if kw in query:
                    logger.info(f"规则命中: {intent} (关键词: {kw})")
                    return IntentResult(
                        intent=intent,
                        confidence=0.95,
                        entities=self._extract_entities(query),
                        source="rule",
                    )
        return None

    def _bert_classify(self, query: str) -> IntentResult | None:
        """第2层: BERT 分类"""
        try:
            # 调用 BERTIntentClassifier.predict() → {"intent": "...", "confidence": 0.xx}
            pred = self.bert_model.predict(query)
            intent = pred["intent"]
            confidence = pred["confidence"]

            threshold = self.CONFIDENCE_THRESHOLDS.get(intent, 0.5)

            if intent == "out_of_domain":
                return IntentResult(
                    intent="out_of_domain",
                    confidence=confidence,
                    reject=True,
                    reject_reason=f"BERT 判定 OOD conf={confidence:.2f}",
                    fallback_response=self.OOD_RESPONSE,
                    source="bert",
                )

            if confidence < threshold:
                max_prob = confidence
                if max_prob < 0.35:
                    return IntentResult(
                        intent="out_of_domain",
                        confidence=max_prob,
                        reject=True,
                        reject_reason=f"所有意图置信度均低 max={max_prob:.2f}",
                        fallback_response=self.OOD_RESPONSE,
                        source="bert",
                    )
                return IntentResult(
                    intent=intent,
                    confidence=confidence,
                    reject=True,
                    reject_reason=f"置信度不足 conf={confidence:.2f}<threshold={threshold}",
                    fallback_response=self.LOW_CONF_RESPONSE,
                    source="bert",
                )

            return IntentResult(
                intent=intent,
                confidence=confidence,
                entities=self._extract_entities(query),
                source="bert",
            )
        except Exception as e:
            logger.warning(f"BERT 分类失败: {e}")
            return None

    def _llm_classify(self, query: str) -> IntentResult | None:
        """第3层: LLM 兜底分类"""
        try:
            prompt = f"""你是一个保险客服系统的意图分类器。
请将用户问题分类为以下意图之一:
{', '.join(self.INTENT_LABELS)}

用户问题: {query}

请只返回一个 JSON: {{"intent": "...", "confidence": 0.xx}}
"""
            response = self.llm_client.chat_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=100,
            )
            return IntentResult(
                intent=response.get("intent", "闲聊"),
                confidence=response.get("confidence", 0.6),
                entities=self._extract_entities(query),
                source="llm",
            )
        except Exception as e:
            logger.warning(f"LLM 分类失败: {e}")
            return None

    def _extract_entities(self, query: str) -> dict:
        """
        从 query 中提取实体（保司/产品名/疾病/事件）

        优先使用 KG 实体链接器（支持疾病、事件等更多实体类型），
        降级时回退到正则匹配。
        """
        # ── 尝试 KG 实体链接 ──
        try:
            from rag_qa.core.kg_entity_linker import get_entity_linker
            linker = get_entity_linker()
            kg_result = linker.link(query)

            entities = {}
            if kg_result.get("insurer"):
                entities["insurer"] = kg_result["insurer"]
            if kg_result.get("product"):
                entities["product_name"] = kg_result["product"]
            if kg_result.get("products") and len(kg_result["products"]) > 1:
                entities["products"] = kg_result["products"]
            if kg_result.get("disease"):
                entities["disease"] = kg_result["disease"]
            if kg_result.get("disease_category"):
                entities["disease_category"] = kg_result["disease_category"]
            if kg_result.get("event"):
                entities["event"] = kg_result["event"]
            if kg_result.get("dimensions"):
                entities["dimensions"] = kg_result["dimensions"]

            if entities:
                return entities
        except Exception as e:
            logger.debug(f"KG 实体链接不可用，回退到正则: {e}")

        # ── 回退: 正则匹配 ──
        entities = {}
        insurers = ["平安", "众安", "太平洋", "人保", "太平", "阳光", "泰康", "新华", "友邦"]
        for ins in insurers:
            if ins in query:
                entities["insurer"] = ins
                break
        product_patterns = [
            r"([\u4e00-\u9fa5a-zA-Z]{2,10}(?:保|生|e生|尊享|守护))",
        ]
        for pat in product_patterns:
            m = re.search(pat, query)
            if m:
                entities["product_name"] = m.group(1)
                break
        return entities

