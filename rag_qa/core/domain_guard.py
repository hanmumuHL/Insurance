"""
领域边界守卫 — 判断 query 是否属于保险领域
  在 Query Router 之前执行，不匹配的直接拒绝
  0 延迟，拦截 ~70% 不相关问题
"""
from dataclasses import dataclass
from base.logger import logger


@dataclass
class GuardResult:
    passed: bool
    reason: str = ""
    fallback_response: str = ""


class DomainBoundaryGuard:
    """领域边界守卫"""

    # 保险领域关键词白名单
    INSURANCE_KEYWORDS = {
        # 险种
        "医疗险", "重疾险", "意外险", "寿险", "车险", "年金险",
        "百万医疗", "防癌险", "门诊险", "旅行险", "家财险",
        "定期寿险", "终身寿险", "两全险", "教育金", "养老险",
        # 产品动作
        "投保", "续保", "退保", "理赔", "报案", "核保", "加保",
        "保费", "保额", "免赔额", "等待期", "犹豫期", "现金价值",
        "健康告知", "除外责任", "保险责任", "条款", "保障",
        "赔付", "报销", "给付", "津贴", "补偿",
        # 场景
        "住院", "门诊", "手术", "确诊", "意外", "身故",
        "伤残", "疾病", "治疗", "药品", "检查",
        # 保司 (常见)
        "平安", "众安", "太平洋", "人保", "太平", "阳光",
        "泰康", "新华", "友邦", "招商信诺",
    }

    # 明确非保险的拦截词
    BLOCK_KEYWORDS = {
        "天气", "股票", "基金", "游戏", "外卖", "快递",
        "电影", "音乐", "编程", "代码", "服务器",
    }

    # 标准拒绝话术
    OOD_RESPONSE = (
        "抱歉，我是保险智能客服，暂时无法回答这个问题。\n"
        "您可以咨询：保单查询、理赔流程、产品对比、保费试算等保险相关问题。"
    )

    GREETING_RESPONSE = "您好！我是保险智能客服，请问有什么保险相关问题可以帮您？"

    def check(self, query: str) -> GuardResult:
        """检查 query 是否属于保险领域"""
        query_lower = query.strip()

        # ── 太短 → 可能是闲聊 ──
        if len(query_lower) <= 2:
            return GuardResult(
                passed=False,
                reason="query 过短",
                fallback_response=self.GREETING_RESPONSE,
            )

        # ── 白名单检查 ──
        for kw in self.INSURANCE_KEYWORDS:
            if kw in query_lower:
                return GuardResult(passed=True)

        # ── 黑名单检查 ──
        for kw in self.BLOCK_KEYWORDS:
            if kw in query_lower:
                return GuardResult(
                    passed=False,
                    reason=f"命中拦截词: {kw}",
                    fallback_response=self.OOD_RESPONSE,
                )

        # ── 未命中任何规则 → 放行给下一层 ──
        return GuardResult(passed=True)
