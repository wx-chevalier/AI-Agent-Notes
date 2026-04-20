import pymysql
from datetime import datetime

database_schema_string = """
CREATE TABLE IF NOT EXISTS Classes (
    class_id INT PRIMARY KEY COMMENT '班级的ID编号',
    class_name VARCHAR(100) NOT NULL COMMENT '班级的名称'
) ENGINE=InnoDB COMMENT = '班级表';

CREATE TABLE IF NOT EXISTS Students (
    student_id INT PRIMARY KEY COMMENT '学生的唯一性ID编号',
    name VARCHAR(100) NOT NULL COMMENT '学生姓名',
    class_id INT COMMENT '学生所在班级的ID编号，和班级表中的班级ID编号对应'
) ENGINE=InnoDB COMMENT = '学生表';

CREATE TABLE IF NOT EXISTS Scores (
    score_id INT PRIMARY KEY COMMENT '学生成绩表的唯一性ID编号',
    student_id INT COMMENT '学生个人的ID编号，和学生的唯一性ID编号对应',
    subject VARCHAR(100) NOT NULL COMMENT '考试科目，中文名称标识',
    score FLOAT NOT NULL COMMENT '考试科目的分数'
) ENGINE=InnoDB COMMENT = '学生科目成绩表';
"""

# 连接数据库
# 端口、用户名、密码、数据库IP地址请根据自己的实际情况进行修改
# database请在程序运行前确保已在MySQL中已经创建
connection = pymysql.connect(
    host='127.0.0.1',
    port=13306,
    user='root',
    password='123456',
    database='ALLLM',
    charset='utf8mb4'
)
cursor = connection.cursor()

# 处理多语句执行
try:
    sql_statements = [stmt.strip() for stmt in database_schema_string.split(';') if stmt.strip()]
    for stmt in sql_statements:
        cursor.execute(stmt)
    connection.commit()
    print("表创建成功！")
except pymysql.Error as e:
    print("建表失败:", e)
    connection.rollback()
finally:
    cursor.close()
    connection.close()



"""数据插入"""
# 时间戳记录
print(f"数据插入开始：{datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")

# 重构的插入语句（分语句存储）
insert_commands = {
    "classes": """INSERT INTO Classes (class_id, class_name) VALUES (%s, %s)""",
    "students": """INSERT INTO Students (student_id, name, class_id) VALUES (%s, %s, %s)""",
    "scores": """INSERT INTO Scores (score_id, student_id, subject, score) VALUES (%s, %s, %s, %s)"""
}

# 参数化数据（避免SQL注入）
data_pool = {
    "classes": [(1, '一班'), (2, '二班'), (3, '三班'), (4, '四班'), (5, '五班')],
    "students": [
        (1, '张三', 1), (2, '李四', 1), (3, '王五', 2),
        (4, '赵六', 3), (5, '钱七', 4)
    ],
    "scores": [
        (1, 1, '数学', 85.5), (2, 1, '英语', 90.0),
        (3, 2, '数学', 78.0), (4, 3, '英语', 88.5),
        (5, 4, '数学', 92.0)
    ]
}

# 事务处理流程
# 端口、用户名、密码、数据库IP地址请根据自己的实际情况进行修改
# 程序运行前请确保数据库和表已存在
try:
    with pymysql.connect(
            host='127.0.0.1', user='root', password='123456',
            database='ALLLM', charset='utf8mb4',port=13306
    ) as conn:
        with conn.cursor() as cursor:
            # 按依赖顺序插入（班级→学生→成绩）
            for table in ['classes', 'students', 'scores']:
                result = cursor.executemany(
                    insert_commands[table],
                    data_pool[table]
                )
                print(f"[{table.upper()}]  插入成功 {result} 条记录")

            conn.commit()
except pymysql.Error as e:
    print(f"事务异常回滚：{e}")
    conn.rollback()
finally:
    print(f"数据插入结束：{datetime.now().strftime('%H:%M:%S')}")