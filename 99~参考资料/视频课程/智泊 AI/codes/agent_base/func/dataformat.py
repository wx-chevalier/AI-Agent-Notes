import pandas as pd

from models import get_normal_client, ALI_TONGYI_MAX_MODEL, ALI_TONGYI_PLUS_MODEL

client = get_normal_client()

# 加载样例数据
df_complex = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Salary': [50000.0, 100000.5, 150000.75],
    'IsMarried': [True, False, True]
})
print("df_complex：",df_complex)

# # 传入大模型，看下模型的是否能够解读数据。
# response = client.chat.completions.create(
#     model=ALI_TONGYI_PLUS_MODEL,
#     messages=[
#         {"role": "system", "content": "你是一位优秀的数据分析师，现在有这样一份数据集：'%s'" % df_complex},
#         {"role": "user", "content": "请解释一下这个数据集的分布情况"}
#     ],
#
# )
# print(response.choices[0].message.content)
#
# print("==============以JSON方式处理数据集==============")
# # 将DataFrame转换为JSON格式,orient='split'参数将数据、索引和列分开存储
# df_complex_json = df_complex.to_json(orient='split')
# print("df_complex_json：",df_complex_json)
# # 传入大模型，看下模型的是否能够解读数据。
# response = client.chat.completions.create(
#     model=ALI_TONGYI_PLUS_MODEL,
#     messages=[
#         {"role": "system", "content": "你是一位优秀的数据分析师，现在有这样一份数据集：'%s'" % df_complex_json},
#         {"role": "user", "content": "请解释一下这个数据集的分布情况"}
#     ],
#
# )
# print(response.choices[0].message.content)

response = client.chat.completions.create(
    model=ALI_TONGYI_PLUS_MODEL,
    messages=[
        {"role": "system", "content": "你是AI助手" },
        {"role": "user", "content": "今天时间是多少"}
    ],

)
print(response.choices[0].message.content)

