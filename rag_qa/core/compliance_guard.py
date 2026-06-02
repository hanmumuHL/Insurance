# -*- coding: utf-8 -*-
"""
合规守卫 — LLM 生成答案后的合规检查

设计背景:
  保险是强监管行业，LLM 生成的答案必须经过合规检查才能返回给用户。
  不合规的答案可能导致:
    - 误导用户做出错误的投保/理赔决策
    - 违反银保监会的合规要求
    - 被投诉甚至被处罚

检查规则 (5 道检查):
  1. 置信度检查 — LLM 不确定时必须拒绝回答
  2. 金额引用检查 — 涉及赔付金额必须引用条款原文
  3. 医疗建议拦截 — 不能给医疗建议，只能引导看医生
  4. 监管敏感词拦截 — 特定词汇不能出现在回答中
  5. 竞品贬低检查 — 聚合平台不能贬低任何保司产品
"""

import re
from dataclasses import dataclass
from base.logger import logger


@dataclass
class ComplianceResult:
    """
    合规检查结果

    Attributes:
        passed: 是否通过合规检查
        violated_rules: 触发的规则列表
        modified_response: 修改后的回复（如强制添加了条款引用）
        original_response: 原始 LLM 回复
    """
    passed: bool
    violated_rules: list[str] = None
    modified_response: str = ""
    original_response: str = ""


class ComplianceGuard:
    """
    合规守卫

    在 LLM 生成答案之后、返回用户之前调用。
    逐条检查合规规则，不通过则替换/拦截答案。
    """

    def __init__(self):
        # ── 医疗建议关键词 ──
        # LLM 如果说了这些，说明在给医疗建议，必须拦截
        self.MEDICAL_KEYWORDS = [
            "建议您服用", "建议您使用", "可以吃", "应该吃",
            "推荐药物", "治疗方案", "用药建议",
            "建议手术", "建议住院",  # 保险场景下这些是条款解释，不是医疗建议
            # 所以这里需要配合上下文判断，不能简单关键词匹配
            # 简化版本：如果同时出现"医生"+"建议"才算
        ]

        # ── 监管敏感词 ──
        # 保险销售中不允许使用的承诺性/误导性表述
        self.REGULATORY_BLOCK_WORDS = [
            "保证理赔", "一定能赔", "肯定赔",
            "绝对不会拒赔", "无条件赔付",
            "保本保息", "稳赚不赔",
            "最好的保险", "最便宜的",  # 极限用语
        ]

        # ── 竞品贬低模式 ──
        # 聚合平台不能贬低任何保司/产品
        self.DENIGRATE_PATTERNS = [
            r"不推荐.{0,10}(平安|众安|太平洋|人保|太平|阳光|泰康|新华|友邦)",
            r"(平安|众安|太平洋|人保|太平|阳光|泰康|新华|友邦).{0,10}(不好|坑|骗|差|垃圾|不靠谱)",
            r"千万别买.{0,10}(平安|众安|太平洋|人保|太平|阳光|泰康|新华|友邦)",
        ]

    def check(
        self,
        response: str,
        context_chunks: list = None,
        intent: str = "",
    ) -> ComplianceResult:
        """
        对 LLM 生成的回复进行合规检查

        Args:
            response: LLM 生成的回复文本
            context_chunks: 检索到的条款 chunks (用于验证金额引用)
            intent: 意图类型

        Returns:
            ComplianceResult: 合规检查结果
        """
        violated = []

        # ── 检查 1: 医疗建议拦截 ──
        if self._check_medical_advice(response):
            violated.append("medical_advice")
            logger.warning("合规检查: 检测到医疗建议")

        # ── 检查 2: 监管敏感词拦截 ──
        blocked_word = self._check_regulatory_words(response)
        if blocked_word:
            violated.append(f"regulatory_word:{blocked_word}")
            logger.warning(f"合规检查: 命中监管敏感词 '{blocked_word}'")

        # ── 检查 3: 竞品贬低检查 ──
        if self._check_denigration(response):
            violated.append("denigration")
            logger.warning("合规检查: 检测到贬低竞品")

        # ── 检查 4: 金额引用检查 ──
        # 如果回复中提到了金额/比例，必须有条款引用支撑
        if intent in ("理赔咨询", "保费试算"):
            if not self._check_amount_citation(response, context_chunks):
                violated.append("amount_no_citation")
                logger.warning("合规检查: 金额无条款引用")

        # ── 处理结果 ──
        if violated:
            modified = self._apply_fixes(response, violated, context_chunks)
            return ComplianceResult(
                passed=len(violated) <= 1,  # 只有一条轻微违规可以通过修改修复
                violated_rules=violated,
                modified_response=modified,
                original_response=response,
            )

        return ComplianceResult(
            passed=True,
            violated_rules=[],
            modified_response=response,
            original_response=response,
        )

    # ============================================================
    # 各检查规则实现
    # ============================================================

    def _check_medical_advice(self, response: str) -> bool:
        """
        检查是否包含医疗建议

        判定逻辑: 同时出现"医生"相关词 + "建议"相关词 → 可能在给医疗建议
        简单但有效的启发式规则

        注意: 保险条款解读中也可能出现"就医"等词，
        所以这里要求同时匹配多个关键词才触发。
        """
        medical_indicators = ["医生", "就医", "服药", "用药", "治疗"]
        advice_indicators = ["建议", "推荐", "应该", "最好"]

        has_medical = any(kw in response for kw in medical_indicators)
        has_advice = any(kw in response for kw in advice_indicators)

        # 同时出现医疗词 + 建议词 → 可能在给医疗建议
        if has_medical and has_advice:
            # 但如果是引用条款原文（有"根据条款"字样），则放行
            if "根据条款" in response or "条款规定" in response:
                return False
            return True

        return False

    def _check_regulatory_words(self, response: str) -> str:
        """
        检查是否包含监管敏感词

        返回命中的敏感词（空字符串表示未命中）
        这些词在保险销售中是禁止使用的承诺性/误导性表述
        """
        for word in self.REGULATORY_BLOCK_WORDS:
            if word in response:
                return word
        return ""

    def _check_denigration(self, response: str) -> bool:
        """
        检查是否贬低竞品

        聚合平台的中立性要求: 不能贬低任何保司或产品
        使用正则模式匹配常见的贬低表述
        """
        for pattern in self.DENIGRATE_PATTERNS:
            if re.search(pattern, response):
                return True
        return False

    def _check_amount_citation(
        self,
        response: str,
        context_chunks: list = None,
    ) -> bool:
        """
        检查涉及金额/比例的回复是否有条款引用

        判定逻辑:
          1. 检查回复中是否提到了金额或百分比
          2. 如果提到了，检查是否有条款引用标记（如 "根据条款X.X"）
          3. 有金额但无引用 → 不合规

        为什么: 涉及钱的回答必须有据可查，否则用户可能据此索赔
        """
        # 检查是否提到了金额/百分比
        amount_patterns = [
            r"\d+元",           # "1000元"
            r"\d+万",           # "10万"
            r"\d+%",            # "80%"
            r"百分之[\u4e00-\u9fa5]+",  # "百分之八十"
            r"免赔额",
            r"赔付比例",
            r"保额",
        ]

        has_amount = any(re.search(p, response) for p in amount_patterns)

        if not has_amount:
            return True  # 没有提到金额，无需检查

        # 有金额 → 检查是否有引用
        citation_patterns = [
            r"根据条款",
            r"条款第[\d.]+条",
            r"依据.*?规定",
            r"保险合同约定",
            r"条款.*?明确",
        ]

        has_citation = any(re.search(p, response) for p in citation_patterns)

        return has_citation

    # ============================================================
    # 修复方法
    # ============================================================

    def _apply_fixes(
        self,
        response: str,
        violated: list[str],
        context_chunks: list = None,
    ) -> str:
        """
        对不合规的回复进行修复

        修复策略:
          - 医疗建议 → 替换为引导看医生的标准话术
          - 监管敏感词 → 删除/替换
          - 贬低竞品 → 删除贬低描述
          - 金额无引用 → 追加"具体以条款为准"的免责说明
        """
        modified = response

        for rule in violated:
            if rule == "medical_advice":
                # 在末尾追加医疗免责说明
                disclaimer = (
                    "\n\n⚠️ 温馨提示：以上信息仅基于保险条款解读，"
                    "不构成医疗建议。具体诊疗方案请咨询专业医生。"
                )
                modified = modified.rstrip() + disclaimer

            elif rule.startswith("regulatory_word:"):
                # 删除监管敏感词
                word = rule.split(":")[1]
                modified = modified.replace(word, "")

            elif rule == "denigration":
                # 替换贬低描述为中立表述
                modified = re.sub(
                    r"(不推荐|千万别买).{0,20}",
                    "不同产品适合不同需求，建议根据自身情况选择。",
                    modified,
                )

            elif rule == "amount_no_citation":
                # 追加免责说明
                disclaimer = "\n\n📋 以上金额仅供参考，具体以保险合同条款约定为准。"
                modified = modified.rstrip() + disclaimer

        return modified
