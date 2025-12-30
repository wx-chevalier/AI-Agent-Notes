from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
from langchain_community.tools import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from models import get_lc_o_ali_model_client

"""
{agent_scratchpad}为必需，中间代理操作和工具输出消息将在这里传递。
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个AI助手"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# 定义llm
llm = get_lc_o_ali_model_client()

# 定义工具
@tool
def magic_function(input: int) -> int:
    """模拟工具函数."""
    return input + 2

#Tavily需要申请api_key才可使用，并设置系统环境变量TAVILY_API_KEY
#没有申请的同学可以使用.env中的api_key，但是因为TAVILY有限制，可能出现调用次数过多导致限额用完或者被封禁等情况
search = TavilySearchResults(max_results=3)
tools = [magic_function,search]

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({"input": "magic_function(3)的值是多少？"})
print(result)
result = agent_executor.invoke({"input": "请问现任的美国总统的年龄的平方是多少? 请用中文告诉我答案"})
print(result)
result = agent_executor.invoke({"input": "你是谁？"})
print(result)






