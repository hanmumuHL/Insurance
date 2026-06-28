# -*- coding: utf-8 -*-
"""
SQL LIKE 通配符转义工具

参数化查询可防 SQL 注入但不会转义 LIKE 通配符。
用户输入中的 % 会被 SQL LIKE 解释为"零到多个任意字符"，
_ 会被解释为"单个任意字符"，导致匹配到无关数据。

用法:
    pattern = f"%{escape_like_pattern(user_input)}%"
    cursor.execute("SELECT ... WHERE col LIKE %s", (pattern,))
"""

import re


def escape_like_pattern(value: str) -> str:
    """转义 LIKE 模式中的 SQL 通配符 % 和 _"""
    return re.sub(r'([%_\\])', r'\\\1', value)
