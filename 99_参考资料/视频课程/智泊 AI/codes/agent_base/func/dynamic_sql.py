import pymysql
import json

from models import get_normal_client, ALI_TONGYI_MAX_MODEL, ALI_TONGYI_PLUS_MODEL

# 创建一个LLM客户端
client = get_normal_client()

#表的描述
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
# 程序运行前请运行 db_init.py，并确保数据库和表以及表中数据已存在
# 端口、用户名、密码、数据库IP地址请根据自己的实际情况进行修改
connection = pymysql.connect(
    host='127.0.0.1',
    port=13306,
    user='root',
    password='123456',
    database='ALLLM',
    charset='utf8mb4'  # 添加推荐的字符集参数
)

cursor = connection.cursor()

def get_sql_completion(messages, model=ALI_TONGYI_PLUS_MODEL):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        tools=[{
            "type": "function",
            "function": {
                "name": "ask_database",
                "description": "查询数据库的函数。输出是数据库中表的记录",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": f"""
                            一个SQL查询，这个SQL查询提取信息以回答用户的问题。
                            SQL应该使用这个数据库架构来编写:
                            {database_schema_string}
                            SQL查询查询应以纯文本形式返回，而不是JSON格式。
                            查询应仅包含MySQL支持的语法.
                            """,
                        }
                    },
                    "required": ["query"],
                }
            }
        }],
    )
    return response.choices[0].message

def ask_database(query):
    cursor.execute(query)
    records = cursor.fetchall()
    return records


prompt = "查询一班的学生数学成绩是多少？"
messages = [
    {"role": "system", "content": "基于表回答用户问题"},
    {"role": "user", "content": prompt}
]

#response的内容应该是什么？
response = get_sql_completion(messages)
messages.append(response)

if response.tool_calls is not None:
    tool_call = response.tool_calls[0]
    if tool_call.function.name == "ask_database":
        arguments = tool_call.function.arguments
        args = json.loads(arguments)
        print("====SQL====")
        print(args["query"])
        result = ask_database(args["query"])
        print("====MySQL数据库查询结果====")
        print(result)

        messages.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": "ask_database",
            "content": str(result)
        })
        response = get_sql_completion(messages)
        print("====最终回复====")
        print(response.content)