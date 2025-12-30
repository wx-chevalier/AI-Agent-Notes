from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from models import get_lc_o_ali_model_client, ALI_TONGYI_MAX_MODEL
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor, tool
import numexpr
import math

load_dotenv()
# 初始化大模型客户端
llm = get_lc_o_ali_model_client(model=ALI_TONGYI_MAX_MODEL)
@tool
def calculator(expression: str) -> str:
    """使用Python的numexpr库计算表达式。表达式应该是解决问题的单行数学表达式。.
    例子:
        "37593 * 67"
        "37593**(1/5)"
        "37593^(1/5)"
    """
    local_dict = {"pi": math.pi, "e": math.e}
    return str(
        numexpr.evaluate(
            expression.strip(),
            global_dict={},
            local_dict=local_dict,  # 添加常用数学函数
        )
    )
# 设置工具
search = TavilySearchResults(max_results=3)
tools = [calculator,search]

# 设置提示模板
"""提示词中三个变量是必须的：
tools：包含每个工具的描述和参数。
tool_names：包含所有工具名称。
agent_scratchpad：以字符串形式包含以前的代理操作和工具输出。
"""
template = '''
Answer the following questions as best you can. You can use the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should think about what to do
Action: the name of the tool to use, should be one of [{tool_names}]
Action Input: the input to the tool
Observation: the result of the tool
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: Now I know the final answer
Final Answer: the final answer to the original question

Begin!

Question: {input}
Thought: {agent_scratchpad}'''

"""以下是上面提示词的中文意思，注意！这里Question、Action等只能用英文提示词！
因为LangChain在处理时是根据英文单词来进行的，用中文的话会出错，
如果想使用中文提示词，必须自定义？"""
template = ('''
    '尽你所能回答以下问题。如果能力不够你可以使用以下工具:'
    '{tools}
    使用以下格式：'
    'Question: 你必须回答的输入问题'
    'Thought: 你应该始终思考该做什么'
    'Action: 要采取的行动，应该是 [{tool_names}]之一'
    'Action Input: 行动的输入'
    'Observation: 行动的结果'
    '... (这个想法/行动/行动输入/观察可以重复N次)'
    'Thought: 我现在知道最终答案了'
    'Final Answer: 原始输入问题的最终答案'
    '开始!'
    'Question: {input}'
    'Thought: {agent_scratchpad}'
    '''
)
prompt = PromptTemplate.from_template(template)

# 初始化Agent
agent = create_react_agent(llm, tools, prompt)

# 构建AgentExecutor
agent_executor = AgentExecutor(agent=agent,tools=tools,handle_parsing_errors=True,verbose=True)

# 执行AgentExecutor
agent_executor.invoke({"input": """在中国，目前市场上玫瑰花的一般进货价格是多少？大概是多少钱一支？\n如果我在此基础上加价5%，应该如何定价？"""})