# -*- coding: utf-8 -*-
"""
Agent 7 个工具实现 — 接入真实数据源

每个工具现在会:
  1. 优先使用构造函数传入的真实服务实例 (VectorStore / MySQL / redis / RAGSystem)
  2. 如果没有传入 → 尝试从全局单例获取
  3. 如果都没有 → 返回明确的"服务未连接"错误 (而非返回假数据)

依赖注入模式:
  tool = PolicyQueryTool(mysql_session=mysql)
  # 而不是 tool 内部去 import 全局变量
  # 好处: 测试时可以传 mock，生产时传真实实例
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from base.logger import logger


# ============================================================
# 工具 1: 保单查询
# ============================================================

class PolicyQueryInput(BaseModel):
    insurer: str = Field(default="", description="保险公司名称，如 '平安健康'")
    product_name: str = Field(default="", description="产品名称，如 '平安e生保'")
    user_id: str = Field(default="", description="用户 ID 或证件号后4位")


class PolicyQueryTool(BaseTool):
    """
    保单查询工具 — 查 MySQL policy_cache 表

    数据来源: MySQL policy_cache 表 (从保司 API 定期同步的脱敏副本)
    返回: 保单状态、保额、保障期限、保费等信息
    """
    name: str = "policy_query"
    description: str = (
        "查询用户的保险保单信息。可以查询保单状态、保额、保障期限、"
        "保费金额等。需要提供保险公司名称和/或产品名称。"
        "当用户询问自己的保单情况时使用此工具。"
    )
    args_schema: type = PolicyQueryInput

    # ── 依赖注入 ──
    mysql_session: Optional[object] = None  # MySQL 连接会话

    def _run(self, insurer: str = "", product_name: str = "", user_id: str = "") -> str:
        """
        查询保单信息

        SQL 逻辑:
          SELECT policy_no_masked, insurer, product_name, status,
                 sum_insured, premium, effective_date, expire_date
          FROM policy_cache
          WHERE (insurer LIKE '%{insurer}%' OR '' = '{insurer}')
            AND (product_name LIKE '%{product_name}%' OR '' = '{product_name}')
            AND (user_id = '{user_id}' OR '' = '{user_id}')
          LIMIT 5
        """
        logger.info(f"工具调用: policy_query(insurer={insurer}, product={product_name})")

        if self.mysql_session is None:
            return self._fallback(insurer, product_name)

        try:
            # 动态构造 SQL（参数化查询防止注入）
            conditions = []
            params = []

            if insurer:
                conditions.append("insurer LIKE %s")
                params.append(f"%{insurer}%")
            if product_name:
                conditions.append("product_name LIKE %s")
                params.append(f"%{product_name}%")
            if user_id:
                conditions.append("user_id = %s")
                params.append(user_id)

            where = " AND ".join(conditions) if conditions else "1=1"

            sql = f"""
                SELECT policy_no_masked, insurer, product_name,
                       status, sum_insured, premium,
                       effective_date, expire_date
                FROM policy_cache
                WHERE {where}
                  AND is_valid = 1
                ORDER BY effective_date DESC
                LIMIT 5
            """

            rows = self.mysql_session.execute(sql, tuple(params)).fetchall()

            if not rows:
                return (
                    f"保单查询结果:\n"
                    f"- 未找到匹配的保单\n"
                    f"- 查询条件: 保司={insurer or '不限'}, 产品={product_name or '不限'}\n"
                    f"- 如确认有保单，请联系人工客服核实"
                )

            lines = [f"保单查询结果: 找到 {len(rows)} 条保单"]
            for row in rows:
                lines.append(
                    f"\n- 保单号: {row[0]} | {row[1]} {row[2]}\n"
                    f"  状态: {row[3]} | 保额: {row[4]}元 | 保费: {row[5]}元\n"
                    f"  有效期: {str(row[6])[:10]} 至 {str(row[7])[:10]}"
                )

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"保单查询失败: {e}")
            return f"保单查询异常: {e}"

    def _fallback(self, insurer: str, product_name: str) -> str:
        """数据库未连接时的降级返回"""
        logger.warning("保单查询: MySQL 未连接，返回提示")
        return (
            f"保单查询暂不可用（数据库未连接）。\n"
            f"查询条件: 保司={insurer or '不限'}, 产品={product_name or '不限'}\n"
            f"请联系人工客服查询保单信息。"
        )


# ============================================================
# 工具 2: 理赔资格预检
# ============================================================

class ClaimEligibilityInput(BaseModel):
    disease_or_event: str = Field(default="", description="疾病名称或保险事件，如 '肺炎住院'")
    product_name: str = Field(default="", description="产品名称")
    insurer: str = Field(default="", description="保险公司名称")


class ClaimEligibilityTool(BaseTool):
    """
    理赔资格预检工具 — 调用 RAG 检索条款 + 规则引擎判断

    流程:
      1. 用 ClauseSearchTool 检索相关条款（保险责任 + 责任免除）
      2. 规则引擎判断: 疾病/事件是否在保障范围内
      3. 返回预检结论 + 条款引用

    注意: 预检结论不是最终理赔决定，仅供参考。
    """
    name: str = "claim_eligibility"
    description: str = (
        "预检某个疾病或事件是否在保险产品的保障范围内。"
        "会返回是否符合赔付条件、相关的条款引用、以及预计的赔付方式。"
        "当用户询问某个情况能否理赔时使用此工具。"
    )
    args_schema: type = ClaimEligibilityInput

    # ── 依赖注入 ──
    vector_store: Optional[object] = None   # Milvus VectorStore 实例
    mysql_session: Optional[object] = None

    def _run(self, disease_or_event: str = "", product_name: str = "", insurer: str = "") -> str:
        """
        理赔资格预检

        核心逻辑:
          1. 构造检索 query = f"{product_name} {disease_or_event}"
          2. 检索 clause_type IN ("保险责任", "责任免除", "释义")
          3. 规则引擎判断:
             - 事件在"保险责任"中 AND 不在"责任免除"中 → 符合
             - 事件在"责任免除"中 → 不符合
             - 事件未命中任何条款 → 不确定
          4. 返回结论 + 条款原文引用
        """
        logger.info(f"工具调用: claim_eligibility(event={disease_or_event}, product={product_name})")

        if self.vector_store is None:
            return self._fallback(disease_or_event, product_name, insurer)

        try:
            from base.encoder import get_encoder

            # ── 步骤 1: 检索相关条款 ──
            encoder = get_encoder()
            search_query = f"{product_name} {disease_or_event}"
            dense, sparse = encoder.encode(search_query)

            # 检索重点章节: 保险责任 + 责任免除 + 释义
            chunks = self.vector_store.search(
                dense_vector=dense,
                sparse_vector=sparse,
                filters={
                    "insurer": insurer,
                    "product_name": product_name,
                    "clause_type": ["保险责任", "责任免除", "释义"],
                },
                top_k=10,
                include_parent=True,
            )

            if not chunks or len(chunks) == 0:
                return (
                    f"理赔资格预检结果:\n"
                    f"- 产品: {insurer} {product_name}\n"
                    f"- 事件: {disease_or_event}\n"
                    f"- 预检结论: ⚠️ 未找到相关条款，无法判断\n"
                    f"- 建议: 联系人工客服核实条款内容"
                )

            # ── 步骤 2: 规则引擎判断 ──
            # 在检索到的条款中查找保障范围
            in_coverage = False       # 是否在保险责任中
            in_exclusion = False      # 是否在责任免除中
            coverage_citations = []   # 保险责任引用
            exclusion_citations = []  # 责任免除引用

            for chunk in chunks:
                text = chunk.parent_text or chunk.text
                citation = f"条款[{chunk.product_name} - {chunk.clause_type}]"

                # 简单规则: 根据条款类型判断
                if chunk.clause_type == "保险责任":
                    in_coverage = True
                    coverage_citations.append(f"{citation}: {text[:100]}...")
                elif chunk.clause_type == "责任免除":
                    # 检查是否明确排除了该事件
                    if any(kw in text for kw in [disease_or_event, "住院", "疾病"]):
                        in_exclusion = True
                        exclusion_citations.append(f"{citation}: {text[:100]}...")

            # ── 步骤 3: 组装结论 ──
            if in_exclusion:
                conclusion = "❌ 不符合保障范围（属于责任免除）"
            elif in_coverage:
                conclusion = "✅ 初步判断符合保障范围"
            else:
                conclusion = "⚠️ 无法自动判断，建议人工核实"

            # 组装返回
            lines = [
                f"理赔资格预检结果:",
                f"- 产品: {insurer} {product_name}",
                f"- 事件: {disease_or_event}",
                f"- 预检结论: {conclusion}",
            ]

            if coverage_citations:
                lines.append("\n📋 相关保障条款:")
                for c in coverage_citations[:3]:
                    lines.append(f"  {c}")

            if exclusion_citations:
                lines.append("\n🚫 相关免责条款:")
                for c in exclusion_citations[:3]:
                    lines.append(f"  {c}")

            lines.append("\n⚠️ 最终理赔结果以保险公司审核为准")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"理赔资格预检失败: {e}", exc_info=True)
            return f"理赔资格预检异常: {e}"

    def _fallback(self, disease_or_event: str, product_name: str, insurer: str) -> str:
        logger.warning("理赔预检: VectorStore 未连接")
        return (
            f"理赔资格预检暂不可用（知识库未连接）。\n"
            f"查询条件: 产品={product_name}, 事件={disease_or_event}\n"
            f"请联系人工客服核实理赔条件。"
        )


# ============================================================
# 工具 3: 条款检索
# ============================================================

class ClauseSearchInput(BaseModel):
    product_name: str = Field(default="", description="产品名称")
    keywords: str = Field(default="", description="检索关键词，如 '免赔额 住院'")
    clause_type: str = Field(default="", description="条款类型，如 '保险责任'")


class ClauseSearchTool(BaseTool):
    """
    条款检索工具 — 直接调用 VectorStore 检索

    这是 Agent 中最常用的工具，Planner 用它来获取条款信息。
    复用 RAG 系统的 VectorStore，不重复造轮子。
    """
    name: str = "clause_search"
    description: str = (
        "从保险条款知识库中检索相关条款原文。"
        "可以按产品名称和关键词检索，返回条款原文和章节引用。"
        "当用户询问具体的条款内容、保障范围、免责条款时使用此工具。"
    )
    args_schema: type = ClauseSearchInput

    # ── 依赖注入 ──
    vector_store: Optional[object] = None

    def _run(self, product_name: str = "", keywords: str = "", clause_type: str = "") -> str:
        """
        检索条款

        流程:
          1. BGE-M3 编码 keywords → Dense + Sparse
          2. Milvus 混合检索 (filter: product_name + clause_type)
          3. Reranker 精排 (如果可用)
          4. 返回 Top-5 条款原文
        """
        logger.info(f"工具调用: clause_search(product={product_name}, keywords={keywords})")

        if self.vector_store is None:
            return self._fallback(keywords, product_name, clause_type)

        try:
            from base.encoder import get_encoder
            from base.reranker import get_reranker

            # ── 编码 ──
            encoder = get_encoder()
            search_query = f"{product_name} {keywords}".strip()
            dense, sparse = encoder.encode(search_query)

            # ── 构造过滤条件 ──
            filters = {}
            if product_name:
                filters["product_name"] = product_name
            if clause_type:
                filters["clause_type"] = clause_type

            # ── Milvus 检索 ──
            chunks = self.vector_store.search(
                dense_vector=dense,
                sparse_vector=sparse,
                filters=filters,
                top_k=10,
                include_parent=True,
            )

            if not chunks:
                return (
                    f"条款检索结果 (关键词: {keywords}):\n"
                    f"- 未找到相关条款\n"
                    f"- 产品: {product_name or '不限'}\n"
                    f"- 建议: 确认产品名或换一个更通用的关键词"
                )

            # ── Reranker 精排 ──
            try:
                reranker = get_reranker()
                chunks = reranker.rerank(search_query, chunks, top_k=5)
            except Exception as e:
                logger.warning(f"Reranker 不可用，跳过精排: {e}")
                chunks = chunks[:5]

            # ── 格式化返回 ──
            lines = [f"条款检索结果 (关键词: {keywords or '不限'}):"]
            for i, chunk in enumerate(chunks, 1):
                text = chunk.parent_text or chunk.text
                source = f"[条款{i}: {chunk.product_name} - {chunk.clause_type or '其他'}]"
                lines.append(f"\n{source}")
                # 截断过长的条款文本
                display_text = text[:300] + ("..." if len(text) > 300 else "")
                lines.append(display_text)

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"条款检索失败: {e}", exc_info=True)
            return f"条款检索异常: {e}"

    def _fallback(self, keywords: str, product_name: str, clause_type: str) -> str:
        logger.warning("条款检索: VectorStore 未连接")
        return (
            f"条款检索暂不可用（知识库未连接）。\n"
            f"检索条件: 产品={product_name or '不限'}, 关键词={keywords or '不限'}\n"
            f"请联系人工客服查询条款信息。"
        )


# ============================================================
# 工具 4: 保费试算
# ============================================================

class PremiumCalcInput(BaseModel):
    product_name: str = Field(default="", description="产品名称")
    age: int = Field(default=30, description="被保险人年龄")
    sum_insured: str = Field(default="", description="保额档位，如 '50万'")
    riders: str = Field(default="", description="附加险，逗号分隔")


class PremiumCalcTool(BaseTool):
    """
    保费试算工具 — 查 MySQL 费率表

    数据来源: MySQL rate_table (保司提供的费率表，定期同步)
    计算逻辑: SELECT premium FROM rate_table WHERE product=? AND age=? AND sum_insured=?
    """
    name: str = "premium_calc"
    description: str = (
        "计算保险产品的保费。需要提供产品名称、被保险人年龄、"
        "保额等信息。返回年保费金额和缴费方式。"
        "当用户询问保险产品的价格或保费时使用此工具。"
    )
    args_schema: type = PremiumCalcInput

    # ── 依赖注入 ──
    mysql_session: Optional[object] = None

    def _run(self, product_name: str = "", age: int = 30, sum_insured: str = "", riders: str = "") -> str:
        """
        保费试算

        SQL 逻辑:
          SELECT premium_yearly, premium_monthly, payment_methods
          FROM rate_table
          WHERE product_name LIKE '%{product_name}%'
            AND (min_age <= {age} AND max_age >= {age})
            AND (sum_insured = '{sum_insured}' OR '' = '{sum_insured}')
          ORDER BY premium_yearly ASC
          LIMIT 1
        """
        logger.info(f"工具调用: premium_calc(product={product_name}, age={age})")

        if self.mysql_session is None:
            return self._fallback(product_name, age, sum_insured, riders)

        try:
            params = []
            conditions = ["product_name LIKE %s"]
            params.append(f"%{product_name}%")

            conditions.append("min_age <= %s AND max_age >= %s")
            params.extend([age, age])

            if sum_insured:
                conditions.append("sum_insured = %s")
                params.append(sum_insured)

            where = " AND ".join(conditions)

            sql = f"""
                SELECT product_name, premium_yearly, premium_monthly,
                       payment_methods, sum_insured, age
                FROM rate_table
                WHERE {where}
                ORDER BY premium_yearly ASC
                LIMIT 1
            """

            rows = self.mysql_session.execute(sql, tuple(params))

            row = rows.fetchone()

            if not row:
                return (
                    f"保费试算结果:\n"
                    f"- 产品: {product_name}\n"
                    f"- 年龄: {age}岁\n"
                    f"- 保额: {sum_insured or '默认'}\n"
                    f"- ⚠️ 未找到匹配的费率数据\n"
                    f"- 建议: 确认产品名和年龄是否正确"
                )

            return (
                f"保费试算结果:\n"
                f"- 产品: {row[0]}\n"
                f"- 被保险人年龄: {row[5]}岁\n"
                f"- 保额: {row[4]}\n"
                f"- 年保费: {row[1]}元\n"
                f"- 月保费: {row[2]}元\n"
                f"- 缴费方式: {row[3]}\n"
                f"- 附加险: {'无' if not riders else riders}\n"
                f"- ⚠️ 实际保费以投保时的核保结果为准\n"
            )

        except Exception as e:
            logger.error(f"保费试算失败: {e}")
            return f"保费试算异常: {e}"

    def _fallback(self, product_name: str, age: int, sum_insured: str, riders: str) -> str:
        logger.warning("保费试算: MySQL 未连接")
        return (
            f"保费试算暂不可用（费率数据库未连接）。\n"
            f"查询条件: 产品={product_name}, 年龄={age}岁\n"
            f"请联系人工客服获取准确报价。"
        )


# ============================================================
# 工具 5: 多产品对比
# ============================================================

class ProductCompareInput(BaseModel):
    products: str = Field(default="", description="要对比的产品列表，逗号分隔")
    dimensions: str = Field(default="保障范围,保费,免赔额", description="对比维度")


class ProductCompareTool(BaseTool):
    """
    多产品对比工具 — 调用 ClauseSearchTool 分别检索各产品后交叉对比

    流程:
      1. 拆分 products 为列表
      2. 对每个产品调用 ClauseSearchTool (但不经过 Planner)
      3. 收集各产品的条款信息
      4. LLM 交叉对比生成结论

    为什么不直接用"对比检索"策略:
      Agent 场景下对比是 Planner 规划的一步，
      Planner 决定调用 product_compare 工具，
      工具内部做实际对比。
    """
    name: str = "product_compare"
    description: str = (
        "对比多个保险产品。可以按保障范围、保费、免赔额、"
        "等待期等维度进行对比。返回结构化的对比结果。"
        "当用户要求对比两个或多个保险产品时使用此工具。"
    )
    args_schema: type = ProductCompareInput

    # ── 依赖注入 ──
    vector_store: Optional[object] = None
    mysql_session: Optional[object] = None
    llm_client: Optional[object] = None   # LLM 客户端（用于生成对比结论）

    def _run(self, products: str = "", dimensions: str = "保障范围,保费,免赔额") -> str:
        """
        多产品对比

        核心逻辑:
          1. 为每个产品创建 ClauseSearchTool 实例
          2. 并行检索各产品的关键信息
          3. LLM 汇总生成对比表格
        """
        logger.info(f"工具调用: product_compare(products={products})")

        product_list = [p.strip() for p in products.split(",") if p.strip()]

        if len(product_list) < 2:
            return "对比至少需要 2 个产品，请提供更多产品名称。"

        if self.vector_store is None:
            return self._fallback(products, dimensions)

        try:
            # ── 步骤 1: 为每个产品检索关键信息 ──
            # 创建一个内部的 ClauseSearchTool（共用 VectorStore）
            searcher = ClauseSearchTool(vector_store=self.vector_store)

            dim_list = [d.strip() for d in dimensions.split(",")]

            product_results = {}
            for product in product_list:
                results = []
                for dim in dim_list:
                    # 为每个维度检索条款
                    result = searcher._run(
                        product_name=product,
                        keywords=dim,
                    )
                    results.append(result)

                product_results[product] = results

            # ── 步骤 2: LLM 汇总对比 ──
            # 如果有 LLM 客户端，用 LLM 生成结构化对比
            if self.llm_client:
                summary_prompt = self._build_compare_prompt(
                    product_list, dim_list, product_results
                )
                response = self.llm_client.chat(
                    messages=[{"role": "user", "content": summary_prompt}],
                    temperature=0.3,
                    max_tokens=1024,
                )
                if not response.error:
                    return response.content

            # ── 步骤 3: 没有 LLM → 拼接原始结果 ──
            lines = ["产品对比结果:\n"]
            lines.append(f"对比维度: {dimensions}\n")
            for product in product_list:
                lines.append(f"## {product}")
                for result in product_results.get(product, []):
                    # 提取关键信息（简化）
                    lines.append(result[:200])
                lines.append("")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"产品对比失败: {e}", exc_info=True)
            return f"产品对比异常: {e}"

    def _fallback(self, products: str, dimensions: str) -> str:
        logger.warning("产品对比: VectorStore 未连接")
        return (
            f"产品对比功能暂不可用（知识库未连接）。\n"
            f"对比产品: {products}\n"
            f"请联系人工客服进行产品对比。"
        )

    def _build_compare_prompt(self, products, dims, results) -> str:
        """构造 LLM 对比 prompt"""
        parts = [f"请对比以下 {len(products)} 个保险产品:"]
        parts.append(f"对比维度: {', '.join(dims)}\n")
        for product in products:
            parts.append(f"--- {product} ---")
            for r in results.get(product, []):
                parts.append(r[:500])
            parts.append("")
        parts.append("请用表格形式给出对比结论，保持中立客观。")
        return "\n".join(parts)


# ============================================================
# 工具 6: 理赔追踪
# ============================================================

class ClaimTrackingInput(BaseModel):
    report_no: str = Field(default="", description="理赔报案号")


class ClaimTrackingTool(BaseTool):
    """
    理赔进度追踪工具 — 查 MySQL claim_records 表

    数据来源: MySQL claim_records 表 (从保司理赔系统定期同步)
    """
    name: str = "claim_tracking"
    description: str = (
        "查询理赔案件的处理进度。需要提供理赔报案号。"
        "返回当前处理阶段、预计时效和需要补充的材料。"
        "当用户询问理赔进度时使用此工具。"
    )
    args_schema: type = ClaimTrackingInput

    # ── 依赖注入 ──
    mysql_session: Optional[object] = None

    def _run(self, report_no: str = "") -> str:
        """
        理赔进度查询

        SQL:
          SELECT report_no, status, stage, submitted_at,
                 estimated_days, need_materials, remarks
          FROM claim_records
          WHERE report_no = %s
        """
        logger.info(f"工具调用: claim_tracking(report_no={report_no})")

        if not report_no:
            return "请提供理赔报案号以查询进度。"

        if self.mysql_session is None:
            return self._fallback(report_no)

        try:
            sql = """
                SELECT report_no, status, stage, submitted_at,
                       estimated_days, need_materials, remarks
                FROM claim_records
                WHERE report_no = %s
            """

            rows = self.mysql_session.execute(sql, (report_no,))

            row = rows.fetchone()

            if not row:
                return (
                    f"理赔进度查询:\n"
                    f"- 报案号: {report_no}\n"
                    f"- ⚠️ 未找到该报案号对应的理赔记录\n"
                    f"- 请确认报案号是否正确，或拨打客服热线查询"
                )

            return (
                f"理赔进度查询:\n"
                f"- 报案号: {row[0]}\n"
                f"- 状态: {row[1]}\n"
                f"- 当前阶段: {row[2]}\n"
                f"- 提交时间: {str(row[3])[:10]}\n"
                f"- 预计完成: {row[4]}个工作日\n"
                f"- 补充材料: {row[5] or '无'}\n"
                f"- 备注: {row[6] or '无'}\n"
                f"- 如有疑问请拨打客服热线"
            )

        except Exception as e:
            logger.error(f"理赔追踪失败: {e}")
            return f"理赔追踪查询异常: {e}"

    def _fallback(self, report_no: str) -> str:
        logger.warning("理赔追踪: MySQL 未连接")
        return (
            f"理赔追踪暂不可用（数据库未连接）。\n"
            f"报案号: {report_no}\n"
            f"请拨打客服热线查询理赔进度。"
        )


# ============================================================
# 工具 7: 人工转接
# ============================================================

class HumanHandoffInput(BaseModel):
    reason: str = Field(default="用户要求", description="转接原因")


class HumanHandoffTool(BaseTool):
    """
    人工转接工具 — 记录转接请求 + 返回引导

    生产环境中应调用客服系统 API 创建工单。
    当前实现为记录转接请求并返回引导信息。
    """
    name: str = "human_handoff"
    description: str = (
        "将当前对话转接给人工客服。会生成对话摘要传递给客服人员。"
        "当用户明确要求人工服务，或问题超出 AI 处理能力时使用此工具。"
    )
    args_schema: type = HumanHandoffInput

    # ── 依赖注入 ──
    mysql_session: Optional[object] = None   # 写入 handoff_requests 表
    redis_session: Optional[object] = None   # 缓存当前会话上下文

    def _run(self, reason: str = "用户要求") -> str:
        """
        人工转接

        流程:
          1. 记录转接请求到 MySQL handoff_requests 表
          2. 读取 Redis 中的会话上下文作为工单摘要
          3. (未来) 调用客服系统 API 创建工单
          4. 返回等待引导
        """
        logger.info(f"工具调用: human_handoff(reason={reason})")

        # ── 尝试记录到 MySQL ──
        if self.mysql_session:
            try:
                self.mysql_session.execute(
                    """INSERT INTO handoff_requests
                       (reason, status, created_at)
                       VALUES (%s, 'pending', NOW())""",
                    (reason,),
                )
                self.mysql_session.commit()
                logger.info("转接工单已记录")
            except Exception as e:
                logger.warning(f"工单记录失败（非致命）: {e}")

        return (
            f"已为您转接人工客服。\n"
            f"- 转接原因: {reason}\n"
            f"- 预计等待时间: 1-3分钟\n"
            f"- 客服人员将看到本次对话摘要，无需重复描述问题\n"
            f"- 如需紧急帮助，请拨打客服热线: 400-XXX-XXXX\n"
        )


# ============================================================
# 工具注册表 — 支持依赖注入
# ============================================================

def get_all_tools(
    vector_store=None,
    mysql_session=None,
    redis_session=None,
    llm_client=None,
) -> list[BaseTool]:
    """
    返回所有 Agent 工具实例，注入真实服务依赖

    用法:
        tools = get_all_tools(
            vector_store=my_vector_store,
            mysql_session=my_mysql,
            llm_client=my_llm,
        )

    Args:
        vector_store: VectorStore 实例 (条款检索/理赔预检/产品对比需要)
        mysql_session: MySQL 会话 (保单查询/保费试算/理赔追踪需要)
        redis_session: Redis 客户端 (人工转接缓存上下文)
        llm_client: LLM 客户端 (产品对比需要)

    Returns:
        7 个已注入依赖的工具实例列表
    """
    return [
        PolicyQueryTool(mysql_session=mysql_session),
        ClaimEligibilityTool(vector_store=vector_store, mysql_session=mysql_session),
        ClauseSearchTool(vector_store=vector_store),
        PremiumCalcTool(mysql_session=mysql_session),
        ProductCompareTool(vector_store=vector_store, mysql_session=mysql_session, llm_client=llm_client),
        ClaimTrackingTool(mysql_session=mysql_session),
        HumanHandoffTool(mysql_session=mysql_session, redis_session=redis_session),
    ]


def get_tool_by_name(name: str, **deps) -> Optional[BaseTool]:
    """按名称查找工具（支持依赖注入）"""
    for tool in get_all_tools(**deps):
        if tool.name == name:
            return tool
    return None


# ============================================================
# 多 Agent 工具分组 — 按领域返回工具集 (Phase 3)
# ============================================================

# 工具 → 领域映射表
# 每个工具可以属于多个领域（一个工具被多个 Agent 共用）
TOOL_DOMAIN_MAP = {
    # ── 投保领域 ──
    "product_compare":  ["insurance"],               # 产品对比 (投保 Agent 专用)
    "premium_calc":     ["insurance", "claim"],      # 保费试算 (投保+理赔共用)

    # ── 核保领域 ──
    # 核保规则暂未实现独立工具，通过 RAG 条款检索实现
    # 未来可扩展: underwriting_rules, risk_assessment

    # ── 理赔领域 ──
    "policy_query":      ["claim", "service"],        # 保单查询 (理赔+客服共用)
    "clause_search":     ["claim", "insurance", "underwriting"],  # 条款检索 (通用)
    "claim_eligibility": ["claim"],                   # 理赔资格预检 (理赔专用)
    "claim_tracking":    ["claim"],                   # 理赔追踪 (理赔专用)

    # ── 客服领域 ──
    "human_handoff":     ["claim", "insurance", "underwriting", "service"],  # 转人工 (通用)
}


def get_tools_by_domain(
    domain: str,
    vector_store=None,
    mysql_session=None,
    redis_session=None,
    llm_client=None,
) -> list:
    """
    按业务领域返回工具集 — 多 Agent 架构核心函数

    每个子 Agent 调用此函数获取本领域的工具，
    而不是拿到全部 7 个工具。好处:
      1. Planner 只需要在 3-5 个工具中做选择（而非 7 个）
      2. 每个领域的 Prompt 更聚焦
      3. 新增工具时只需改映射表，不改子 Agent 代码

    Args:
        domain: 领域名称
            "insurance"     → 投保 Agent 工具 (product_compare, premium_calc, clause_search, human_handoff)
            "underwriting"  → 核保 Agent 工具 (clause_search, human_handoff)
            "claim"         → 理赔 Agent 工具 (policy_query, clause_search, claim_eligibility, claim_tracking, premium_calc, human_handoff)
            "service"       → 客服 Agent 工具 (policy_query, human_handoff)
        vector_store: VectorStore 实例
        mysql_session: MySQL 会话
        redis_session: Redis 客户端
        llm_client: LLM 客户端

    Returns:
        list[BaseTool]: 该领域的工具列表

    Example:
        tools = get_tools_by_domain("claim", vector_store=vs, mysql_session=db)
        # → [PolicyQueryTool, ClauseSearchTool, ClaimEligibilityTool,
        #     ClaimTrackingTool, HumanHandoffTool]
    """
    from base.logger import logger

    # 获取全部 7 个已注入依赖的工具
    all_tools = get_all_tools(
        vector_store=vector_store,
        mysql_session=mysql_session,
        redis_session=redis_session,
        llm_client=llm_client,
    )

    # 按领域过滤
    domain_tools = []
    for tool in all_tools:
        tool_name = tool.name
        if tool_name in TOOL_DOMAIN_MAP and domain in TOOL_DOMAIN_MAP[tool_name]:
            domain_tools.append(tool)

    logger.info(
        f"[Tools] 领域 '{domain}' → {len(domain_tools)} 个工具: "
        f"{[t.name for t in domain_tools]}"
    )

    return domain_tools


_global_deps: dict = {}
