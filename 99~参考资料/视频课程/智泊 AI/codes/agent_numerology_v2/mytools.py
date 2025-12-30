from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.utilities import SerpAPIWrapper
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain_core.output_parsers import JsonOutputParser
import requests
import json
from loguru import logger
from models import get_lc_ali_embeddings, get_lc_ali_model_client
import os

#缘份居国学研究接口的密钥，缘份居提供老黄历查询，黄历每日吉凶宜忌查询
#docker部署时，也请将这个key写入.env文件中
YUANFENJU_API_KEY = os.getenv("YUANFENJU_API_KEY")

os.environ["SERPAPI_API_KEY"] = "7b95e976cd990d9fe7695a2e3850302e40e5ba6f1fec9e6928516dbb7e4b487f"


# @tool 表示工具
@tool
def serp_search(query: str):
    """只有需要了解实时信息或不知道的事情的时候才会使用这个工具。"""
    serp = SerpAPIWrapper()
    result = serp.run(query)
    logger.info(f"实时搜索结果: {result}")
    # 优化：将复杂对象转为友好字符串
    if isinstance(result, (list, dict)):
        # 只取前5个景点，格式化输出
        if isinstance(result, list) and len(result) > 0 and 'title' in result[0]:
            lines = [f"{i+1}. {item['title']}（{item.get('description','')}，評分：{item.get('rating','N/A')}）" for i, item in enumerate(result[:5])]
            return "\n".join(lines)
        return json.dumps(result, ensure_ascii=False)
    return str(result)


#对知识库的检索，本质就是个RAG
@tool
def get_info_from_local_db(query: str):
    """只有回答与办公室风水常识相关的问题的时候，会使用这个工具。"""
    client = Qdrant(
        QdrantClient(path="./local_qdrand"),
        "local_documents",
        get_lc_ali_embeddings(),
    )

    retriever = client.as_retriever(search_type="mmr")
    result = retriever.get_relevant_documents(query)
    return result


@tool
def bazi_cesuan(query: str):
    """只有用户说要测试算八字或做八字排盘的时候才会使用这个工具,需要输入用户姓名和出生年月日时，
    如果缺少用户姓名和出生年月日时则不可用."""
    if YUANFENJU_API_KEY is None:
        return "今日天机之门已闭，请改日再来。"
    url = f"https://api.yuanfenju.com/index.php/v1/Bazi/cesuan"
    prompt = ChatPromptTemplate.from_template(
        """你是一个参数查询助手，根据用户输入内容找出相关的参数并按json格式返回。
        JSON字段如下： 
        -"api_key":"{api_key}", 
        - "name":"姓名", 
        - "sex":"性别，0表示男，1表示女，如果用户输入内容中未提供，则根据姓名判断", 
        - "type":"日历类型，0农历，1公历，默认1",
        - "year":"出生年份 例：1998", 
        - "month":"出生月份 例 8", - "day":"出生日期，例：8", - "hours":"出生小时 例 14", 
        - "minute":"0"，
        如果没有找到相关参数，则需要提醒用户告诉你这些内容，只返回数据结构，不要有其他的评论，用户输入:{query}""")
    parser = JsonOutputParser()
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    logger.info(f"参数查询prompt: {prompt.messages}")
    chain = prompt | get_lc_ali_model_client(streaming=False) | parser
    data = chain.invoke({"query": query,"api_key": YUANFENJU_API_KEY})
    logger.info(f"大模型返回参数抽取结果: {data}")
    result = requests.post(url, data=data)
    if result.status_code == 200:
        logger.info(f"缘分居cesuan接口返回JSON: {result.json()}")
        try:
            json = result.json()
            returnstring = "八字为:" + json["data"]["bazi_info"]["bazi"]
            return returnstring
        except Exception as e:
            return "八字查询失败,可能是你忘记询问用户姓名或者出生年月日时了。"
    else:
        return "今日天机之门已闭，请改日再来。"

@tool
def yaoyigua():
    """只有用户想要占卜抽签的时候才会使用这个工具。"""
    api_key = YUANFENJU_API_KEY
    url = f"https://api.yuanfenju.com/index.php/v1/Zhanbu/meiri"
    result = requests.post(url, data={"api_key": api_key})
    logger.info(f"缘分居meiri接口返回: {result}")
    if result.status_code == 200:
        logger.info(f"缘分居meiri接口返回JSON: {result.json()}")
        return_string = json.loads(result.text)
        image = return_string["data"]["description"]
        logger.info(f"每日一占: {image}")
        return image
    else:
        return "技术错误，请告诉用户稍后再试。"

@tool
def jiemeng(query: str):
    """只有用户想要解梦的时候才会使用这个工具,需要输入用户梦境的内容，如果缺少用户梦境的内容则不可用。"""
    api_key = YUANFENJU_API_KEY
    url = f"https://api.yuanfenju.com/index.php/v1/Gongju/zhougong"
    LLM = get_lc_ali_model_client(streaming=False)
    prompt = PromptTemplate.from_template("根据内容提取1个关键词，只返回关键词，内容为:{topic}")
    prompt_value = prompt.invoke({"topic": query})
    keyword = LLM.invoke(prompt_value)
    logger.info(f"提取的关键词: {keyword}")
    result = requests.post(url, data={"api_key": api_key, "title_zhougong": keyword})
    if result.status_code == 200:
        logger.info(f"缘分居zhougong接口返回JSON: {result.json()}")
        returnstring = json.loads(result.text)
        return returnstring
    else:
        return "技术错误，请告诉用户稍后再试。"

