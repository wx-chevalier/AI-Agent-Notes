"""
传给大模型的函数名称必须和函数的本名一模一样吗？
"""
import pandas as pd

from models import get_normal_client, ALI_TONGYI_MAX_MODEL, ALI_TONGYI_PLUS_MODEL

client = get_normal_client()
import json
from io import StringIO

# 加载样例数据
df_complex = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Salary': [50000.0, 100000.5, 150000.75],
    'IsMarried': [True, False, True]
})
print(df_complex)

# 将样例数据中的DataFrame转换为JSON格式
df_complex_json = df_complex.to_json(orient='split')

# 编写函数功能
def calculate_total_age_from_split_json(input_json):
    """
    从给定的JSON格式字符串（按'split'方向排列）中解析出DataFrame，计算所有人的年龄总和，并以JSON格式返回结果。
    参数:
    input_json (str): 包含个体数据的JSON格式字符串。
    返回:
    str: 所有人的年龄总和，以JSON格式返回。
    """
    print("函数calculate_total_age_from_split_json被调用")
    # 将JSON字符串转换为DataFrame
    df = pd.read_json(StringIO(input_json), orient='split')
    # 计算所有人的年龄总和
    total_age = df['Age'].sum()
    # 将结果转换为字符串形式，然后使用json.dumps()转换为JSON格式
    return json.dumps({"total_age": str(total_age)})

#  测试函数功能  使用函数计算年龄总和，并以JSON格式输出
result = calculate_total_age_from_split_json(df_complex_json)
print("The JSON output is:", result)

# 定义函数库
function_repository = {
    "计算年龄总和的函数": calculate_total_age_from_split_json,
}

# 构建messages
messages = [
    {"role": "system","content": "你是一位优秀的数据分析师, 现在有这样一个数据集input_json：%s，数据集以JSON形式呈现" % df_complex_json},
    {"role": "user", "content": "请在数据集input_json上执行计算所有人年龄总和函数"}
]
response = client.chat.completions.create(
    model=ALI_TONGYI_PLUS_MODEL,
    messages=messages,
    # 加载函数调用的参数
    tools=[{  # 用 JSON 描述函数。可以定义多个。由大模型决定调用谁。也可能都不调用
        "type": "function",
        "function": {
            "name": "计算年龄总和的函数",
            "description": "计算年龄总和的函数，会从给定的JSON格式字符串中解析出DataFrame，计算所有人的年龄总和，并以JSON格式返回结果。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_json": {
                        "type": "string",
                        "description": "包含待计算年龄总和的数据集",
                    },
                },
                "required": ["input_json"],
            }
        }
    }],  # 编写JSON Schema描述
    tool_choice="auto"
)
print('发送给大模型的消息: ',messages)
print('大模型的应答:  ',response.choices[0].message)

# 保存交互过程中的函数名称
function_name = response.choices[0].message.tool_calls[0].function.name
print('大模型的认为应该执行的函数: ', function_name)

# 加载交互过程中的参数
function_args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
print('应该执行的函数所需要的参数:',function_args)

# 保存交互过程中的函数ID
tool_calls_id = response.choices[0].message.tool_calls[0].id
print('应该执行的函数的ID: ',tool_calls_id)


# 得到外部函数的响应结果
local_fuction_call = function_repository[function_name]
print('本地函数调用: ',local_fuction_call)
function_response = local_fuction_call(**function_args)
print('本地函数调用结果:',function_response)

# 拼接模型返回结果messages
messages.append(response.choices[0].message)
print('发送给大模型的消息拼接 - 1、应答: ',messages)

# 拼接函数执行结果messages
messages.append({
    "role": "tool",
    "name": function_name,
    "tool_call_id": tool_calls_id,
    "content": function_response
    }
)

print('发送给大模型的消息拼接- 2、函数执行结果: ',messages)

# 再次向ChatCompletion模型提问
final_response = client.chat.completions.create(
    model=ALI_TONGYI_PLUS_MODEL,
    messages=messages,
)

print('answer: ',final_response.choices[0].message.content)
