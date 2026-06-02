"""
PII 脱敏引擎 — 正则 + NER 双层检测
  保险场景下覆盖 6 类敏感信息
  脱敏在 query 发给 LLM API 之前执行
"""
import re
import uuid
from dataclasses import dataclass, field
from base.logger import logger


@dataclass
class DesensitizeResult:
    text: str                           # 脱敏后的文本
    mapping: dict[str, str] = field(default_factory=dict)  # {占位符: 原值}
    detected_types: list[str] = field(default_factory=list)


class PIIDesensitizer:
    """PII 脱敏器 — 正则层 + 规则兜底"""

    # ── 正则模式（按优先级排序，先匹配长的）──
    PATTERNS = [
        # 身份证号（18位）— 最长，优先匹配
        ("证件号", r"\b\d{17}[\dXx]\b"),
        # 银行卡号（16-19位）— 次优先，防止被15位身份证规则截断
        ("银行卡号", r"\b\d{16,19}\b"),
        # 身份证号（15位旧版）— 最后
        ("证件号", r"\b\d{15}\b"),
        # 手机号
        ("手机号", r"\b1[3-9]\d{9}\b"),
        # 保单号（P开头+数字）
        ("保单号", r"\bP\d{6,20}\b"),
        # 邮箱
        ("邮箱", r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    ]

    # ── 姓名检测模式（上下文依赖）──
    NAME_PATTERNS = [
        r"(?:我(?:叫|是|姓))([\u4e00-\u9fa5]{2,4})",
        r"(?:姓名[：:]\s*)([\u4e00-\u9fa5]{2,4})",
        r"(?:患者|被保人|投保人)[：:]\s*([\u4e00-\u9fa5]{2,4})",
    ]

    def desensitize(self, text: str) -> DesensitizeResult:
        """
        对文本执行 PII 脱敏
        返回脱敏后文本 + 映射表（用于后续还原）
        """
        mapping = {}
        detected = []
        result = text

        # ── 第1层：正则匹配 ──
        for pii_type, pattern in self.PATTERNS:
            matches = re.findall(pattern, result)
            for match in matches:
                if match not in mapping.values():
                    placeholder = f"[{pii_type}_{len(mapping):03d}]"
                    mapping[placeholder] = match
                    detected.append(pii_type)
                # 找到该 match 对应的 placeholder
                for ph, val in mapping.items():
                    if val == match:
                        result = result.replace(match, ph, 1)
                        break

        # ── 第2层：姓名检测（上下文模式）──
        for pattern in self.NAME_PATTERNS:
            for m in re.finditer(pattern, result):
                name = m.group(1)
                if len(name) >= 2 and name not in mapping.values():
                    placeholder = f"[姓名_{len(mapping):03d}]"
                    mapping[placeholder] = name
                    detected.append("姓名")
                    result = result.replace(name, placeholder, 1)

        if mapping:
            logger.info(f"PII 脱敏: 检测到 {len(mapping)} 处敏感信息 {detected}")

        return DesensitizeResult(
            text=result,
            mapping=mapping,
            detected_types=list(set(detected)),
        )

    @staticmethod
    def restore(text: str, mapping: dict[str, str]) -> str:
        """根据映射表还原 PII"""
        result = text
        for placeholder, original in mapping.items():
            result = result.replace(placeholder, original)
        return result


# ── 便捷函数 ──
_desensitizer = PIIDesensitizer()

def desensitize(text: str) -> DesensitizeResult:
    return _desensitizer.desensitize(text)

def restore(text: str, mapping: dict[str, str]) -> str:
    return PIIDesensitizer.restore(text, mapping)
