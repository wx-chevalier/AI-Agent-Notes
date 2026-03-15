import os
from langchain import hub
from langchain.agents import AgentExecutor, create_self_ask_with_search_agent
from langchain_community.tools.tavily_search import TavilyAnswer
from dotenv import load_dotenv

from models import get_lc_o_ali_model_client, ALI_TONGYI_MAX_MODEL, ALI_TONGYI_DEEPSEEK_V3
import langchain
load_dotenv()
#开启调试模型
langchain.debug = True
"""
FIREWORKS_API_KEY = "fw_3ZdBAq9xu2sB59kyLQTD2kmj
TAVILY_API_KEY = "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"
"""

# from langchain_fireworks import ChatFireworks
# llm = ChatFireworks(
#     api_key=os.getenv("FIREWORKS_API_KEY"),
#     model="accounts/fireworks/models/deepseek-v3",
#     max_tokens=256,
# )
llm = get_lc_o_ali_model_client(model=ALI_TONGYI_DEEPSEEK_V3)

# prompt = hub.pull("hwchase17/self-ask-with-search")
# print("hwchase17/self-ask-with-search:",prompt)
from langchain_core.prompts import PromptTemplate
template = '''
Answer the following question. When needed, you can ask follow-up questions and answer them to reach the final answer. Always use English format for follow-up questions and answers:

Examples:
Question: Who lived longer, Muhammad Ali or Alan Turing?
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
So the final answer is: Muhammad Ali

Question: When was the founder of craigslist born?
Are follow up questions needed here: Yes.
Follow up: Who founded craigslist?
Intermediate answer: Craigslist was founded by Craig Newmark.
Follow up: When was Craig Newmark born?
Intermediate answer: Craig Newmark was born on December 6, 1952.
So the final answer is: December 6, 1952

Question: Who was George Washington's paternal grandfather?
Are follow up questions needed here: Yes.
Follow up: Who was George Washington's father?
Intermediate answer: George Washington's father was Augustine Washington.
Follow up: Who was Augustine Washington's father?
Intermediate answer: Augustine Washington's father was Lawrence Washington.
So the final answer is: Lawrence Washington

IMPORTANT: 
1. ONLY use the exact format shown above
2. DO NOT include any explanations or extra text
3. DO NOT use markdown formatting like **bold**
4. ONLY output in the specified format
5. End your response with "So the final answer is: [answer]"

Question: {input}
Are follow up questions needed here: {agent_scratchpad}'''
prompt = PromptTemplate.from_template(template)

# template = '''
# 在回答用户问题时，可以自己提出问题并回答，来增强对问题的理解以提高回答质量。
# 示例一：
# 问题：谁活得更久，穆罕默德·阿里还是阿兰·图灵？
# 这里是否需要后续问题：是的。
# 追问：穆罕默德·阿里去世时几岁？
# 中间答案：穆罕默德·阿里去世时74岁。
# 追问：阿兰·图灵去世时几岁？
# 中间答案：阿兰·图灵去世时41岁。
# 所以最终的答案是：穆罕默德·阿里
# 示例二：
# 问：craigslist的创始人是什么时候出生的？
# 这里是否需要后续问题：是的。
# 追问：craigslist的创始人是谁？
# 中间答案：Craigslist是由克雷格·纽马克创立的。
# 追问：克雷格·纽马克什么时候出生的？
# 中间答案：克雷格·纽马克出生于1952年12月6日。
# 所以最终的答案是：1952年12月6日
# 示例三：
# 问：乔治·华盛顿的外祖父是谁？
# 这里是否需要后续问题：是的。
# 下一篇：谁是乔治·华盛顿的母亲？
# 中间答案：乔治·华盛顿的母亲是玛丽·鲍尔·华盛顿。
# 追问：玛丽·鲍尔·华盛顿的父亲是谁？
# 中间答案：玛丽·鲍尔华盛顿的父亲是约瑟夫·鲍尔。
# 所以最终的答案是：约瑟夫·鲍尔
#
# 用户问题: {input}
# 解决问题:{agent_scratchpad}'''
# prompt = PromptTemplate.from_template(template)
#上面的提示词会执行报错，因为LangChain的self_ask.py中对大模型的输出结果规定了英文格式

# 创建工具（TavilyAnswer: 使用Tavily搜索引擎）max_results=1: 最多返回1条结果 include_raw_content=True: 包含原始内容 name="Intermediate Answer": 工具名称 tavily_api_key=os.getenv("TAVILY_API_KEY"): 使用Tavily API密钥
tools = [TavilyAnswer(max_results=1, include_raw_content=True,name="Intermediate Answer",tavily_api_key=os.getenv("TAVILY_API_KEY"))]
# 创建代理（create_self_ask_with_search_agent: 创建自我询问代理）llm: 大模型 tools: 工具 prompt: 提示词
agent = create_self_ask_with_search_agent(llm, tools, prompt)
# 创建代理执行器（AgentExecutor: 代理执行器）agent: 代理 tools: 工具 verbose=True: 打印详细信息 handle_parsing_errors=True: 处理解析错误
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,handle_parsing_errors=True)

print(agent_executor.invoke({"input": "Are both the directors of Jaws and Casino Royale from the same country?"}))
"""
示例四：
问：《大白鲨》和《皇家赌场》的导演都来自同一个国家吗？
这里是否需要后续问题：是的。
追问：《大白鲨》的导演是谁？
中间答案：《大白鲨》的导演是史蒂文·斯皮尔伯格。
追问：史蒂文·斯皮尔伯格来自哪里？
中间答案：美国。
追问：皇家赌场的导演是谁？
中间答案：皇家赌场的导演是马丁·坎贝尔。
追问：马丁·坎贝尔来自哪里？
中间答案：新西兰。
所以最终的答案是：不是
"""