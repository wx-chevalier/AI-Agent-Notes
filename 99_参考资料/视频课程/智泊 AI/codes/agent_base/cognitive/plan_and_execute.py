from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain import SerpAPIWrapper
from langchain.agents import Tool
from langchain import LLMMathChain
from dotenv import load_dotenv

from models import get_lc_o_ali_model_client
# 加载环境变量
load_dotenv()
# 配置API密钥和基础URL
llm = get_lc_o_ali_model_client()
# 创建工具
search = SerpAPIWrapper()
llm_math_chain = LLMMathChain(llm=llm, verbose=True)

# 定义工具列表
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="用于回答关于当前事件的问题"
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="用于计算或解决问题，只能处理简单的数学表达式，不能处理列表或复杂数据结构"
    )
]

# 创建计划器（有大模型执行的计划器）
planner = load_chat_planner(llm)
# 创建执行器（有大模型执行的执行器）verbose=True: 打印详细信息
executor = load_agent_executor(llm, tools, verbose=True)

# 创建代理（PlanAndExecute: 计划并执行）verbose=True: 打印详细信息
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

# 运行代理解决实际问题（invoke: 调用）  input: 输入问题
print(agent.invoke({"input": "在中国，100人民币能买几束玫瑰花？请用中文回答问题"}))