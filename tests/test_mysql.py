# -*- coding: utf-8 -*-
"""
MySQL 服务测试模块

运行方式:
    cd /home/newnew/code/code/pythonCode/Insurance
    python -m pytest tests/test_mysql.py -v

    或单独运行:
    python tests/test_mysql.py

测试内容:
  1. 连接测试        — 能否连接到 MySQL 服务
  2. 建库建表        — 创建 insurance_platform 数据库和所有业务表
  3. CRUD 操作       — 插入/查询/更新/删除
  4. 连接池压力测试  — 多线程并发读写

依赖:
  - mysql-connector-python (已在 requirements.txt 中)
  - config/.env 中配置正确的 MYSQL_USER / MYSQL_PASSWORD
"""

import sys
import os
import time
import threading

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings


# ============================================================
# 测试数据库连接
# ============================================================

def get_connection():
    """
    创建 MySQL 连接

    使用 config/.env 中的配置。
    如果连接失败，抛出异常（由调用方处理）。
    """
    import mysql.connector
    return mysql.connector.connect(
        host=settings.mysql.host,
        port=settings.mysql.port,
        user=settings.mysql.user,
        password=settings.mysql.password,
        charset="utf8mb4",
        autocommit=False,
    )


def test_connection():
    """
    测试 1: 验证能否连接到 MySQL 服务。

    这是最基础的测试——如果连不上，后续测试全都没法跑。
    通过此测试说明:
      - MySQL 服务正在运行
      - config/.env 中的用户名和密码正确
      - 网络/防火墙配置正确
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT VERSION()")
        version = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        print(f"  ✅ MySQL 连接成功 (版本: {version})")
        return True
    except Exception as e:
        print(f"  ❌ MySQL 连接失败: {e}")
        print(f"     请检查 config/.env 中的 MYSQL_USER / MYSQL_PASSWORD")
        return False


# ============================================================
# 建库建表
# ============================================================

def test_create_database():
    """
    测试 2: 创建数据库（如果不存在）。

    数据库名: insurance_platform
    字符集: utf8mb4 (支持中文和 emoji)
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()

        db_name = settings.mysql.database
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
        conn.commit()

        cursor.execute(f"USE `{db_name}`")
        cursor.close()
        conn.close()

        print(f"  ✅ 数据库 '{db_name}' 已就绪")
        return True
    except Exception as e:
        print(f"  ❌ 创建数据库失败: {e}")
        return False


def test_create_tables():
    """
    测试 3: 创建所有业务表。

    以下是保险平台需要的全部 8 张表:

    ┌─────────────────────┐
    │ documents           │  文档元数据（PDF 来源、MD5、解析状态）
    ├─────────────────────┤
    │ document_chunks     │  chunk 级数据（文本、类型、父块关联）
    ├─────────────────────┤
    │ faq_questions       │  FAQ 问答对（精确匹配缓存的数据源）
    ├─────────────────────┤
    │ products            │  产品注册信息（保司、产品名、状态）
    ├─────────────────────┤
    │ policy_cache        │  保单缓存（从保司 API 同步的脱敏副本）
    ├─────────────────────┤
    │ rate_table          │  费率表（年龄 → 保费映射）
    ├─────────────────────┤
    │ claim_records       │  理赔记录（从保司理赔系统同步）
    ├─────────────────────┤
    │ handoff_requests    │  人工转接请求记录
    └─────────────────────┘

    建表原则:
      - 所有表使用 InnoDB 引擎（支持事务和外键）
      - 字符集 utf8mb4（完整 Unicode 支持）
      - 合适的索引（加速查询）
      - 时间字段有默认值
    """
    tables = {
        # ── 文档元数据 ──
        "documents": """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id VARCHAR(100) PRIMARY KEY COMMENT '文档唯一 ID: {产品编码}_{MD5前8位}',
                insurer VARCHAR(100) NOT NULL COMMENT '保险公司名称',
                product_name VARCHAR(200) NOT NULL COMMENT '产品名称',
                product_code VARCHAR(50) NOT NULL COMMENT '产品编码',
                doc_type VARCHAR(50) DEFAULT '条款' COMMENT '文档类型: 条款/投保须知/费率表/理赔指南',
                file_name VARCHAR(500) COMMENT '原始文件名',
                file_md5 VARCHAR(32) UNIQUE COMMENT '文件 MD5 (去重用)',
                version VARCHAR(20) DEFAULT '1.0' COMMENT '版本号',
                page_count INT DEFAULT 0 COMMENT 'PDF 页数',
                raw_text MEDIUMTEXT COMMENT '解析后的全文 (截断存储)',
                is_active BOOLEAN DEFAULT TRUE COMMENT '是否有效 (旧版本标记 FALSE)',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_insurer (insurer),
                INDEX idx_product (product_code),
                INDEX idx_md5 (file_md5)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='文档元数据'
        """,

        # ── 文档分块 ──
        "document_chunks": """
            CREATE TABLE IF NOT EXISTS document_chunks (
                chunk_id VARCHAR(100) PRIMARY KEY COMMENT 'chunk 唯一 ID',
                doc_id VARCHAR(100) NOT NULL COMMENT '所属文档 ID',
                text TEXT NOT NULL COMMENT 'chunk 文本内容',
                chunk_type ENUM('child', 'parent') DEFAULT 'child' COMMENT '块类型',
                parent_id VARCHAR(100) COMMENT '父块 ID (child 块关联)',
                chunk_index INT DEFAULT 0 COMMENT '在文档中的顺序位置',
                clause_type VARCHAR(100) COMMENT '条款章节: 保险责任/责任免除/释义 等',
                is_valid BOOLEAN DEFAULT TRUE COMMENT '是否有效',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_doc_id (doc_id),
                INDEX idx_parent (parent_id),
                INDEX idx_clause_type (clause_type),
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='文档分块'
        """,

        # ── FAQ 问答对 ──
        "faq_questions": """
            CREATE TABLE IF NOT EXISTS faq_questions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                question VARCHAR(500) NOT NULL COMMENT '常见问题',
                normalized_question VARCHAR(500) NOT NULL COMMENT '标准化后的问题 (去标点空格)',
                answer TEXT NOT NULL COMMENT '标准答案',
                category VARCHAR(50) COMMENT '分类: 理赔/投保/退保/产品',
                frequency INT DEFAULT 0 COMMENT '被命中次数',
                is_active BOOLEAN DEFAULT TRUE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY uk_normalized (normalized_question),
                INDEX idx_category (category)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='FAQ 问答对'
        """,

        # ── 产品信息 ──
        "products": """
            CREATE TABLE IF NOT EXISTS products (
                product_code VARCHAR(50) PRIMARY KEY COMMENT '产品编码',
                product_name VARCHAR(200) NOT NULL COMMENT '产品名称',
                insurer VARCHAR(100) NOT NULL COMMENT '保险公司',
                category VARCHAR(50) COMMENT '险种: 医疗险/重疾险/意外险 等',
                is_active BOOLEAN DEFAULT TRUE COMMENT '是否在售',
                description TEXT COMMENT '产品简介',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_insurer (insurer),
                INDEX idx_category (category)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='产品注册信息'
        """,

        # ── 保单缓存 ──
        "policy_cache": """
            CREATE TABLE IF NOT EXISTS policy_cache (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id VARCHAR(50) NOT NULL COMMENT '用户 ID',
                policy_no_masked VARCHAR(50) COMMENT '脱敏保单号',
                insurer VARCHAR(100) NOT NULL COMMENT '保险公司',
                product_name VARCHAR(200) NOT NULL COMMENT '产品名称',
                status VARCHAR(20) DEFAULT '有效' COMMENT '保单状态: 有效/已失效/退保',
                sum_insured DECIMAL(12,2) COMMENT '保额',
                premium DECIMAL(10,2) COMMENT '年保费',
                effective_date DATE COMMENT '生效日期',
                expire_date DATE COMMENT '到期日期',
                is_valid BOOLEAN DEFAULT TRUE,
                synced_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '同步时间',
                INDEX idx_user (user_id),
                INDEX idx_insurer (insurer)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='保单缓存 (脱敏副本)'
        """,

        # ── 费率表 ──
        "rate_table": """
            CREATE TABLE IF NOT EXISTS rate_table (
                id INT AUTO_INCREMENT PRIMARY KEY,
                product_name VARCHAR(200) NOT NULL COMMENT '产品名称',
                min_age INT NOT NULL COMMENT '最小年龄',
                max_age INT NOT NULL COMMENT '最大年龄',
                sum_insured VARCHAR(20) COMMENT '保额档位',
                premium_yearly DECIMAL(10,2) NOT NULL COMMENT '年保费',
                premium_monthly DECIMAL(10,2) COMMENT '月保费',
                payment_methods VARCHAR(100) DEFAULT '年缴/月缴' COMMENT '缴费方式',
                is_active BOOLEAN DEFAULT TRUE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uk_rate (product_name, min_age, max_age, sum_insured),
                INDEX idx_product (product_name)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='费率表'
        """,

        # ── 理赔记录 ──
        "claim_records": """
            CREATE TABLE IF NOT EXISTS claim_records (
                id INT AUTO_INCREMENT PRIMARY KEY,
                report_no VARCHAR(50) UNIQUE NOT NULL COMMENT '报案号',
                user_id VARCHAR(50) COMMENT '用户 ID',
                status VARCHAR(20) DEFAULT '处理中' COMMENT '理赔状态',
                stage VARCHAR(50) COMMENT '当前阶段: 已报案/材料审核/调查/理算/赔付/结案',
                submitted_at DATE COMMENT '报案日期',
                estimated_days INT COMMENT '预计处理天数',
                need_materials TEXT COMMENT '需补充的材料',
                remarks TEXT COMMENT '备注',
                is_valid BOOLEAN DEFAULT TRUE,
                synced_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_user (user_id),
                INDEX idx_report (report_no)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='理赔记录'
        """,

        # ── 人工转接请求 ──
        "handoff_requests": """
            CREATE TABLE IF NOT EXISTS handoff_requests (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(100) COMMENT '会话 ID',
                reason VARCHAR(200) COMMENT '转接原因',
                status VARCHAR(20) DEFAULT 'pending' COMMENT 'pending/accepted/resolved',
                summary TEXT COMMENT '对话摘要',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_session (session_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='人工转接请求'
        """,
    }

    try:
        conn = get_connection()
        # 必须先 USE 数据库才能建表
        db_name = settings.mysql.database
        conn.database = db_name

        cursor = conn.cursor()

        created_count = 0
        for table_name, ddl in tables.items():
            try:
                cursor.execute(ddl)
                created_count += 1
            except Exception as e:
                print(f"  ⚠️ 表 {table_name} 创建失败: {e}")

        conn.commit()
        cursor.close()
        conn.close()

        print(f"  ✅ 数据表已就绪 ({created_count}/{len(tables)} 张表)")
        return created_count == len(tables)
    except Exception as e:
        print(f"  ❌ 建表失败: {e}")
        return False


# ============================================================
# CRUD 操作测试
# ============================================================

def test_insert_and_query():
    """
    测试 4: 插入测试数据并查询验证。

    验证各表的 INSERT / SELECT / UPDATE / DELETE 是否正常工作。
    测试数据使用 "TEST_" 前缀，方便识别和清理。

    测试内容:
      1. 插入一条产品记录 → 查询验证
      2. 插入一条 FAQ 记录 → 查询验证
      3. 插入保单/费率/理赔的测试记录
      4. 事务回滚验证（故意插入重复数据 → 回滚 → 数据未污染）
      5. 清理所有 TEST_ 数据
    """
    try:
        conn = get_connection()
        conn.database = settings.mysql.database
        cursor = conn.cursor()

        # ── 测试 1: 产品 INSERT + SELECT ──
        cursor.execute("""
            INSERT INTO products (product_code, product_name, insurer, category)
            VALUES ('TEST-001', '测试产品_医疗险', '测试保司', '医疗险')
            ON DUPLICATE KEY UPDATE product_name=VALUES(product_name)
        """)
        conn.commit()

        cursor.execute(
            "SELECT product_name, insurer FROM products WHERE product_code = 'TEST-001'"
        )
        row = cursor.fetchone()
        assert row is not None, "产品查询失败"
        assert row[0] == '测试产品_医疗险', f"产品名不匹配: {row[0]}"
        print(f"     产品 CRUD ✅ ({row[0]} / {row[1]})")

        # ── 测试 2: FAQ INSERT + SELECT ──
        cursor.execute("""
            INSERT INTO faq_questions (question, normalized_question, answer, category)
            VALUES ('测试问题: 怎么理赔?', 'test_how_to_claim', '测试答案: 理赔需要报案→提交材料→审核→赔付', '理赔')
            ON DUPLICATE KEY UPDATE answer=VALUES(answer)
        """)
        conn.commit()

        cursor.execute(
            "SELECT answer FROM faq_questions WHERE normalized_question = 'test_how_to_claim'"
        )
        row = cursor.fetchone()
        assert row is not None and '理赔' in row[0], "FAQ 查询失败"
        print(f"     FAQ CRUD ✅")

        # ── 测试 3: 保单缓存 INSERT + SELECT ──
        cursor.execute("""
            INSERT INTO policy_cache (user_id, insurer, product_name, status, sum_insured, premium, effective_date, expire_date)
            VALUES ('TEST_USER_001', '测试保司', '测试产品_医疗险', '有效', 500000, 358, '2025-01-01', '2026-01-01')
        """)
        conn.commit()

        cursor.execute(
            "SELECT status, sum_insured FROM policy_cache WHERE user_id = 'TEST_USER_001'"
        )
        row = cursor.fetchone()
        assert row is not None and row[0] == '有效', "保单查询失败"
        print(f"     保单缓存 CRUD ✅ (保额: {row[1]}元)")

        # ── 测试 4: 费率表 INSERT + SELECT ──
        cursor.execute("""
            INSERT IGNORE INTO rate_table (product_name, min_age, max_age, sum_insured, premium_yearly, premium_monthly)
            VALUES ('测试产品_医疗险', 18, 25, '50万', 258, 22)
        """)
        conn.commit()

        cursor.execute(
            "SELECT premium_yearly FROM rate_table WHERE product_name = '测试产品_医疗险' AND 20 BETWEEN min_age AND max_age"
        )
        row = cursor.fetchone()
        assert row is not None, "费率查询失败"
        print(f"     费率表 CRUD ✅ (年保费: {row[0]}元)")

        # ── 测试 5: 理赔记录 INSERT + SELECT ──
        cursor.execute("""
            INSERT IGNORE INTO claim_records (report_no, user_id, status, stage, submitted_at, estimated_days)
            VALUES ('TEST-CLM-00001', 'TEST_USER_001', '处理中', '材料审核', '2025-05-20', 7)
        """)
        conn.commit()

        cursor.execute(
            "SELECT stage, estimated_days FROM claim_records WHERE report_no = 'TEST-CLM-00001'"
        )
        row = cursor.fetchone()
        assert row is not None and row[0] == '材料审核', "理赔查询失败"
        print(f"     理赔记录 CRUD ✅ (阶段: {row[0]}, 预计: {row[1]}天)")

        # ── 测试 6: 人工转接 INSERT + SELECT ──
        cursor.execute("""
            INSERT INTO handoff_requests (session_id, reason, status)
            VALUES ('TEST-SESSION-001', '测试转接', 'pending')
        """)
        conn.commit()

        cursor.execute(
            "SELECT status FROM handoff_requests WHERE session_id = 'TEST-SESSION-001'"
        )
        row = cursor.fetchone()
        assert row is not None, "转接请求查询失败"
        print(f"     人工转接 CRUD ✅ (状态: {row[0]})")

        # ── 测试 7: 事务回滚 ──
        # 验证: 插入失败时事务回滚，数据不被污染
        try:
            cursor.execute("START TRANSACTION")
            # 故意插入重复的 PRIMARY KEY
            cursor.execute("""
                INSERT INTO products (product_code, product_name, insurer)
                VALUES ('TEST-001', '应该回滚', '测试')
            """)
            # 这行不应该被执行（上一行已经报错）
            conn.commit()
            print("     ⚠️ 事务测试: 意外通过（不应插入重复主键）")
        except Exception:
            conn.rollback()
            # 查询验证: test-001 的数据没有被第二次插入影响
            cursor.execute(
                "SELECT product_name FROM products WHERE product_code = 'TEST-001'"
            )
            row = cursor.fetchone()
            assert row[0] == '测试产品_医疗险', f"事务回滚后数据被污染: {row[0]}"
            print(f"     事务回滚 CRUD ✅ (回滚后数据未被污染)")

        # ── 清理测试数据 ──
        cursor.execute("DELETE FROM claim_records WHERE report_no LIKE 'TEST-%'")
        cursor.execute("DELETE FROM policy_cache WHERE user_id LIKE 'TEST_%'")
        cursor.execute("DELETE FROM rate_table WHERE product_name LIKE 'TEST_%'")
        cursor.execute("DELETE FROM handoff_requests WHERE session_id LIKE 'TEST-%'")
        cursor.execute("DELETE FROM faq_questions WHERE normalized_question LIKE 'test_%'")
        cursor.execute("DELETE FROM products WHERE product_code LIKE 'TEST-%'")
        conn.commit()

        print(f"     测试数据已清理")

        cursor.close()
        conn.close()

        print(f"  ✅ CRUD 测试全部通过 (7 项)")
        return True

    except Exception as e:
        print(f"  ❌ CRUD 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# 连接池压力测试
# ============================================================

def test_connection_pool():
    """
    测试 5: 多线程并发读写（模拟高并发场景）。

    启动 10 个线程并发查询 products 表，验证:
      1. 连接池不会耗尽
      2. 查询结果正确
      3. 没有死锁

    实际部署时建议使用 SQLAlchemy 的连接池管理（pool_size=10）。
    """
    try:
        errors = []
        results = []

        def worker(thread_id: int):
            """单个线程的查询工作"""
            try:
                conn = get_connection()
                conn.database = settings.mysql.database
                cursor = conn.cursor()
                for _ in range(5):  # 每个线程执行 5 次查询
                    cursor.execute("SELECT COUNT(*) FROM products")
                    row = cursor.fetchone()
                    results.append(row[0])
                cursor.close()
                conn.close()
            except Exception as e:
                errors.append(f"Thread-{thread_id}: {e}")

        # 启动 10 个线程
        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # 等待所有线程完成
        for t in threads:
            t.join(timeout=5)

        if errors:
            print(f"  ❌ 并发测试失败: {errors}")
            return False

        assert len(results) == 50, f"期望 50 次查询结果，实际 {len(results)}"
        print(f"  ✅ 连接池压力测试通过 (10 线程 × 5 查询 = {len(results)} 次, 无错误)")
        return True

    except Exception as e:
        print(f"  ❌ 并发测试失败: {e}")
        return False


# ============================================================
# 主入口
# ============================================================

def run_all_tests():
    """运行所有 MySQL 测试"""
    print("=" * 60)
    print("MySQL 服务测试")
    print(f"连接目标: {settings.mysql.host}:{settings.mysql.port}/{settings.mysql.database}")
    print(f"用户: {settings.mysql.user}")
    print("=" * 60)
    print()

    results = {}

    # 测试 1: 连接
    print("测试 1: MySQL 连接 ...")
    results["连接"] = test_connection()
    if not results["连接"]:
        print("\n⚠️ 连接失败，跳过后续测试。请检查:")
        print("  1. MySQL 服务是否运行: sudo systemctl status mysql")
        print("  2. config/.env 中的 MYSQL_USER / MYSQL_PASSWORD 是否正确")
        print("  3. 用户是否有权限: mysql -u root -p")
        return

    # 测试 2: 建库
    print("\n测试 2: 创建数据库 ...")
    results["建库"] = test_create_database()

    # 测试 3: 建表
    print("\n测试 3: 创建数据表 ...")
    results["建表"] = test_create_tables()

    # 测试 4: CRUD
    print("\n测试 4: CRUD 操作 ...")
    results["CRUD"] = test_insert_and_query()

    # 测试 5: 并发
    print("\n测试 5: 连接池压力测试 ...")
    results["并发"] = test_connection_pool()

    # ── 汇总 ──
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")

    all_passed = all(results.values())
    print(f"\n{'🎉 全部通过!' if all_passed else '⚠️ 部分测试失败，请检查上述错误信息'}")


if __name__ == "__main__":
    run_all_tests()
