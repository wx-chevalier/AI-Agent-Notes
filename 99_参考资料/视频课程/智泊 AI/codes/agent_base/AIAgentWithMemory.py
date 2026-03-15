from langchain.memory import ConversationBufferMemory
from langchain_community.tools import TavilySearchResults
from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
from langchain_core.prompts import ChatPromptTemplate

from models import get_lc_o_ali_model_client

llm = get_lc_o_ali_model_client()

@tool
def magic_function(input: int) -> int:
    """模拟工具函数."""
    return input + 2

#Tavily需要申请api_key才可使用，并设置环境变量TAVILY_API_KEY
search = TavilySearchResults(max_results=3)
tools = [magic_function,search]

print(tools)

# 记忆组件
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个AI助手"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,memory=memory)

result = agent_executor.invoke({"input": "你好，我叫云帆，很高兴见到你，请问你是谁？"})
print("result1 = ",result)
result = agent_executor.invoke({"input": "magic_function(3)的值是多少？"})
print("result2 = ",result)
result = agent_executor.invoke({"input": "我是谁？"})
print("result3 = ",result)
