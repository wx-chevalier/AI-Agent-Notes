import gradio as gr
import os
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
from http import HTTPStatus
from dashscope import Generation
import random

'''本代码无需深入学习，也不提供任何疑问解答，
只为展示大模型中Function Call，为何要定义函数库
本代码使用了阿里云百炼的专用SDK ，安装方法 pip install dashscope
实现时，参考了阿里官方Demo'''

INIT_HISTORY = {"role": "system",
     "content": "你是由通义千问提供的人工智能助手，你是百科全书。"}

#初始化上下文
query_history = [INIT_HISTORY]

#定义外部工具列表，模型在选择使用哪个工具时会参考工具的name和description
tools = [
    # 工具1 获取当前时刻的时间
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "当你想知道现在的时间时非常有用。",#
            "parameters": {}  # 因为获取当前时间无需输入参数，因此parameters为空字典
        }
    },
    # 工具2 饮品推荐
    {
        "type": "function",
        "function": {
            "name": "recom_drink",
            "description": "为您推荐附近的饮品店",  #
            "parameters": {}
        }
    },
    # 工具2 打开本地计算器
    {
        "type": "function",
        "function": {
            "name": "open_calc",
            "description": "打开本地计算机上的计算器。",
            "parameters": {}
        }
    },
    # 工具3 打开某个网站
    {
        "type": "function",
        "function": {
            "name": "open_browser",
            "description": "打开本地计算机上的网页浏览器，并接受网站的url作为参数。",
            "parameters": {
                "type": "object",
                "properties": {
                    #定义的第一个参数的详情
                    "first": {
                        "type": "str",
                        "description": "网站的url"
                    }
                },
                "required": ["first"]
            }
        }
    }
]
#和大模型通话过程的一个包装，
# 会把用户的查询请求和大模型的应答内容做额外处理，这也是多轮对话和单次对话最大的不同
def chat(query):
    #将本轮对话的请求内容放入的历史记录
    query_history.append({
        "role": "user",
        "content": query
    })
    print("-" * 5,query, "-" * 5)
    response = Generation.call(
        api_key=DASHSCOPE_API_KEY,
        model="qwen-max",
        messages=query_history,
        # 告诉大模型，你有哪些工具可以用，怎么用
        tools=tools,
        seed=random.randint(1, 10000),
        result_format='message'
    )

    if response.status_code == HTTPStatus.OK:
        result = response.output.choices[0].message
        print(result,"="*5)
        query_history.append(result)
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
        messages = query_history[:-1]
    return result

# 查询当前时间的工具。返回结果示例：“当前时间：2024-04-15 17:15:18。“
from datetime import datetime
import json
def get_current_time():
    # 获取当前日期和时间
    current_datetime = datetime.now()
    # 格式化当前日期和时间
    formatted_time = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    result = f"当前时间：{formatted_time}。"
    print(result)
    # 返回格式化后的当前时间
    return result

def recom_drink():
    result = '''距离您500米内有如下饮料店：\n
    1、蜜雪冰城\n
    2、茶颜悦色\n
    另外距离您200米内有惠民便利店，里面应该有矿泉水或其他饮品'''
    return result

import subprocess
def open_calc():
    subprocess.Popen(['calc.exe'])

import webbrowser
def open_browser(url, browser_name=None):
    if browser_name:
        # 获取特定浏览器的控制器
        browser = webbrowser.get(browser_name)
    else:
        # 使用默认浏览器
        browser = webbrowser
    # 打开浏览器并导航到指定的URL
    browser.open(url)


#将输入与输出显示在Gradio前端展示界面
def process_llm_response(query,show_history):
    # 处理输入为空的情况
    if len(query) == 0:
        return show_history+[("","")]
    try:
        yield show_history+[(query,"正在查询大模型...")]
        response = chat(query)
        if 'tool_calls' not in response:
            yield show_history+[(query,response.content)]
        elif response.tool_calls[0]['function']['name'] == 'get_current_time':
            yield show_history + [(query, "大模型不支持实时查询，需要调用工具get_current_time")]
            tool_info = {"name": "get_current_time", "role": "tool"}
            tool_info['content'] = get_current_time()
            yield show_history+[(query, "工具get_current_time调用完成，大模型继续处理")]
            # 模型的第二轮调用，对工具的输出进行总结
            query_history.append(tool_info)
            second_response = chat("工具结果已经生成，请继续处理")
            print(second_response)
            yield show_history + [(query, second_response.content)]
        elif response.tool_calls[0]['function']['name'] == 'open_calc':
            yield show_history + [(query, "AI助手正在为您打开计算器")]
            open_calc()
            tool_info = {"name": "open_calc", "role": "tool"}
            tool_info['content'] = "打开计算器成功"
            query_history.append(tool_info)
            yield show_history+[(query, "AI助手打开计算器完成，等待你的新指令")]
        elif response.tool_calls[0]['function']['name'] == 'open_browser':
            yield show_history + [(query, "AI助手正在为您打开本地浏览器")]
            first = json.loads(response.tool_calls[0]['function']['arguments'])['first']
            open_browser(first)
            tool_info = {"name": "open_browser", "role": "tool"}
            tool_info['content'] = "打开本地浏览器成功"
            query_history.append(tool_info)
            yield show_history+[(query, "AI助手打开本地浏览器完成，等待你的新指令")]
        elif response.tool_calls[0]['function']['name'] == 'recom_drink':
            yield show_history + [(query, "根据您目前的需求，AI助手正在为您找寻合适的店铺")]
            result = recom_drink()
            tool_info = {"name": "recom_drink", "role": "tool"}
            tool_info['content'] = "找寻店铺完成"
            query_history.append(tool_info)
            result = result + "，需要AI助手为您下单或导航吗？"
            yield show_history+[(query, result)]
    except Exception as e:
        print(e)
        yield show_history+[(query,"AI助手出错，请重试或者检查")]

# 前端界面展示
with gr.Blocks(title="大模型中Function Call演示") as demo:
    # 在界面中央展示标题
    gr.HTML('<center><h1>大模型中Function Call演示 开发：云帆</h1></center>')
    with gr.Row():
        with gr.Column(scale=10):
            chatbot = gr.Chatbot(value=[["hello","很高兴见到您！我是云帆老师开发的AI应用"]],height=650)
    with gr.Row():
        msg = gr.Textbox(label="输入",placeholder="您想了解什么呢？")
    # 一些示例问题:比如帮我打开浏览器并访问淘宝官方网站
    with gr.Row():
        examples = gr.Examples(examples=[
            '请问如何做红烧牛肉？',
            '料酒可以换成白酒吗？',
            '帮我打开计算器',
            '现在几点了？',
            '帮我访问淘宝网',
            '我渴了'],inputs=[msg])
    clear = gr.ClearButton([chatbot,msg])
    msg.submit(process_llm_response, [msg, chatbot], [chatbot])

if __name__ == '__main__':
    demo.launch(server_port=7779)