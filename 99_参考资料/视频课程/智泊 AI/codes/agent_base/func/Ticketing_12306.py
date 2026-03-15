import json
import requests
import inspect
from models import get_normal_client, ALI_TONGYI_MAX_MODEL, ALI_TONGYI_PLUS_MODEL
import pandas as pd
from datetime import datetime

client = get_normal_client()

#根据实际的业务接口，做数据的获取和分析
def check_tick(date, start, end):
    print('开始访问12306接口:',date, start, end)
    url = 'https://kyfw.12306.cn/otn/leftTicket/queryG?leftTicketDTO.train_date={}&leftTicketDTO.from_station={}&leftTicketDTO.to_station={}&purpose_codes=ADULT'.format(
        date, start, end)
    headers = {
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "If-Modified-Since": "0",
        "Referer": "https://kyfw.12306.cn/otn/leftTicket/init?linktypeid=dc&fs=^%^E4^%^B8^%^8A^%^E6^%^B5^%^B7,SHH&ts=^%^E5^%^8C^%^97^%^E4^%^BA^%^AC,BJP&date=2025-07-03&flag=N,N,Y",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.97 Safari/537.36 SE 2.X MetaSr 1.0",
        "X-Requested-With": "XMLHttpRequest",
        "sec-ch-ua": "^\\^Not)A;Brand^^;v=^\\^24^^, ^\\^Chromium^^;v=^\\^116^^",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "^\\^Windows^^"
    }
    cookies = {
        "_uab_collina": "175144477944438320557093",
        "JSESSIONID": "3D658D5D08A9CA7B82498DD10544A58C",
        "route": "9036359bb8a8a461c164a04f8f50b252",
        "BIGipServerotn": "1473839370.24610.0000",
        "BIGipServerpassport": "1005060362.50215.0000",
        "guidesStatus": "off",
        "highContrastMode": "defaltMode",
        "cursorStatus": "off",
        "_jc_save_fromStation": "^%^u4E0A^%^u6D77^%^2CSHH",
        "_jc_save_toStation": "^%^u5317^%^u4EAC^%^2CBJP",
        "_jc_save_fromDate": "2025-07-03",
        "_jc_save_toDate": "2025-07-02",
        "_jc_save_wfdc_flag": "dc"
    }

    session = requests.session()
    res = session.get(url, headers=headers, cookies=cookies)

    data = res.json()
    print('12306接口返回，并准备后续处理:', data)

    # 这是一个列表
    result = data["data"]["result"]

    lis = []
    for index in result:
        index_list = index.replace('有', 'Yes').replace('无', 'No').split('|')
        # print(index_list)
        train_number = index_list[3]  # 车次

        if 'G' in train_number:
            time_1 = index_list[8]  # 出发时间
            time_2 = index_list[9]  # 到达时间
            prince_seat = index_list[25]  # 特等座
            first_class_seat = index_list[31]  # 一等座
            second_class = index_list[30]  # 二等座
            dit = {
                '车次': train_number,
                '出发时间': time_1,
                '到站时间': time_2,
                "是否可以预定": index_list[11],

            }
            lis.append(dit)
        else:
            # print(index_list)
            time_1 = index_list[8]  # 出发时间
            time_2 = index_list[9]  # 到达时间

            dit = {
                '车次': train_number,
                '出发时间': time_1,
                '到站时间': time_2,
                "是否可以预定": index_list[11],

            }
            lis.append(dit)
    # print(lis)
    content = pd.DataFrame(lis)
    # print(content)
    return content


def check_date():
    today = datetime.now().date()
    return today

# 定义函数映射字典
function_map = {
    "check_tick": check_tick,
    "check_date": check_date
}

def get_completion(messages, model=ALI_TONGYI_PLUS_MODEL):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=1024,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "check_tick",
                    "description": "给定日期查询有没有票",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "date": {
                                "type": "string",
                                "description": "日期",
                            },
                            "start": {
                                "type": "string",
                                "description": "出发站的地址编码",
                            },
                            "end": {
                                "type": "string",
                                "description": "终点站的地址编码",
                            }

                        },

                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_date",
                    "description": "返回当前的日期",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "str": {
                                "type": "string",
                                "description": "返回今天的日期",
                            }
                        }
                    }
                }
            }
        ]
    )
    return response.choices[0].message


prompt = "查询今天北京到上海的票"

messages = [
    {"role": "system", "content": "你是一个地图通，你可以找到任何地址，找到地址后可以参考的地址编码有<北京：BJP；上海：SHH；天津：TJP；长沙：CSQ；>"},
    {"role": "user", "content": prompt}
]
response = get_completion(messages)

messages.append(response)  # 把大模型的回复加入到对话中
print("=====大模型回复=====")
print(response)

# 用户的请求需要多次函数调用，如果返回的是函数调用结果，则打印出来
while (response.tool_calls is not None):
    for tool_call in response.tool_calls:
        args = json.loads(tool_call.function.arguments)
        print("参数：", args)

        # if (tool_call.function.name == "check_tick"):
        #     print("Call: check_tick")
        #     result = check_tick(**args)
        # elif (tool_call.function.name == "check_date"):
        #     print("Call: check_date")
        #     result = check_date()
        function_name = tool_call.function.name
        if function_name in function_map:
            print(f"Call: {function_name}")
            func = function_map[function_name]
            # 获取函数签名，python内置内省库inspect
            sig = inspect.signature(func)
            params = sig.parameters
            # 根据函数参数决定如何调用
            if params:  # 函数有参数
                if args:
                    result = func(**args)
                else:
                    # 可以提供默认值或抛出错误
                    result = func()
            else:  # 函数无参数
                result = func()
        print(f"=====函数{function_name}返回=====")
        print(result)

        messages.append({
            "tool_call_id": tool_call.id,  # 用于标识函数调用的 ID
            "role": "tool",
            "name": tool_call.function.name,
            "content": str(result)  # 数值result 必须转成字符串
        })

    response = get_completion(messages)
    print("=====大模型回复2=====")
    print(response)
    messages.append(response)  # 把大模型的回复加入到对话中

print("=====最终回复=====")
print(response.content)