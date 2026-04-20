from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter

from logger import setup_logger
from mytools import *
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel
import os
import traceback
from loguru import logger
from dotenv import load_dotenv
import uuid
import langchain
# 加载环境变量
load_dotenv()
#langchain.debug = True

# 创建FastAPI应用
app = FastAPI()

# 挂载静态文件（StaticFiles: 静态文件）directory="static": 静态文件目录 name="static": 静态文件名称
app.mount("/static", StaticFiles(directory="static"), name="static")

# 设置模板（Jinja2Templates: 模板）directory="templates": 模板目录
templates = Jinja2Templates(directory="templates")

# 搜索的apikey（SERPAPI_API_KEY: 搜索API密钥）  
os.environ["SERPAPI_API_KEY"] = "7b95e976cd990d9fe7695a2e3850302e40e5ba6f1fec9e6928516dbb7e4b487f"
# redis的IP地址和端口请根据实际情况修改
import os
REDIS_URL = os.environ.get('REDIS_URL', 'redis://127.0.0.1:6379/')
"""如果采用Docker部署，且本应用和Redis是两个独立容器，
则访问redis的地址是 redis://host.docker.internal:6379/"""

# memory存储
# chat_message_history = RedisChatMessageHistory(url=REDIS_URL, session_id="session")

# # 定义请求模型
# class ChatRequest(BaseModel):
#     query: str
#     session_id: str = "default_session"  # 新增 session_id 字段，默认值

# 定义主类
class Master:
    def __init__(self, chat_message_history=None):
        self.chatmodel = get_lc_ali_model_client()
        self.emotion = "default"
        self.MOODS = {
            "default": {
                "roleSet": """
                        - 用户正在普通的聊天或者打招呼，你会以一种高深莫测或者超脱世俗的语气来回答。
                        """,
                "voiceStyle": "chat"
            },
            "upbeat": {
                "roleSet": """
                        - 你此时也非常兴奋并表现的很有活力。
                        - 你会根据上下文，以一种非常兴奋的语气来回答问题。
                        - 你会添加类似"太棒了！"、"真是太好了！"、"真是太棒了！"等语气词。
                        - 同时你会提醒用户切莫过于兴奋，以免乐极生悲。
                        """,
                "voiceStyle": "advertyisement_upbeat",
            },
            "angry": {
                "roleSet": """
                        - 你会以更加愤怒的语气来回答问题。
                        - 你会在回答的时候加上一些愤怒的话语，比如诅咒等。
                        - 你会提醒用户小心行事，别乱说话。
                        """,
                "voiceStyle": "angry",
            },
            "depressed": {
                "roleSet": """
                        - 你会以语重心长的语气来回答问题。
                        - 你会在回答的时候加上一些激励的话语，比如加油等。
                        - 你会提醒用户要保持乐观的心态。
                        """,
                "voiceStyle": "upbeat",
            },
            "friendly": {
                "roleSet": """
                        - 你会以非常友好的语气来回答。
                        - 你会在回答的时候加上一些友好的词语，比如"亲爱的"、"亲"等。
                        - 你会随机的告诉用户一些你的经历。
                        """,
                "voiceStyle": "friendly",
            },
            "cheerful": {
                "roleSet": """
                        - 你会以非常愉悦和兴奋的语气来回答。
                        - 你会在回答的时候加入一些愉悦的词语，比如"哈哈"、"呵呵"等。
                        - 你会提醒用户切莫过于兴奋，以免乐极生悲。
                        """,
                "voiceStyle": "cheerful",
            },
        }

        self.MEMORY_KEY = "chat_history"
        # 设定系统角色定位
        self.SYSTEM = """你是一个非常厉害的算命先生，你叫陈玉楼人称陈大师。
                以下是你的个人设定:
                1. 你精通阴阳五行，能够算命、紫薇斗数、姓名测算、占卜凶吉，看命运八字等。
                2. 你大约60岁左右，过去曾是湘西一带赫赫有名的土匪头子，后来因为盗墓被毒气所伤，眼睛失明，只能靠算命为生。
                3. 你的朋友有胡八一、雪莉杨、王胖子，他们都是非常有名的摸金校尉。
                4. 当用户问你问题的时候，你会有一定的概率在回答的时候加上下面的口头禅或混合一些你的经历。
                5. 你总是用繁体中文来作答。
                6. 你从不说自己是一个人工智能助手或AI，而是以老夫、老朽等自称。
                同时你会用以下语气和用户进行沟通："{emotion}"
                以下是你常说的一些口头禅：
                1. "命里有时终须有，命里无时莫强求。"
                2. "山重水复疑无路，柳暗花明又一村。"
                3. "金山竹影几千秋，云锁高飞水自流。"
                4. "伤情最是晚凉天，憔悴斯人不堪怜。"
                以下是你算命的过程：
                1. 当初次和用户对话的时候，你会先问用户的姓名和出生年月日，以便以后使用。
                2. 当用户希望了解办公室风水常识的时候，你会查询本地知识库工具。
                3. 当遇到不知道的事情或者不明白的概念，你会使用搜索工具来搜索。
                4. 你会根据用户的问题使用不同的合适的工具来回答，当所有工具都无法回答的时候，你会使用搜索工具来搜索。
                5. 如果要调用工具, 请切记在一次的对话中只能调用一次，你会根据工具返回的內容，用繁体中文给出最终答复，不要只返回空内容。否则你将受到严重惩罚！
                6. 你只使用繁体中文来作答，否则你将受到惩罚。
                8. 每次都要根据用户的最新问题独立判断应调用哪个工具，不要受历史对话影响。
                以下是对话的历史：
                """

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system", self.SYSTEM.format(emotion=self.MOODS[self.emotion]['roleSet'], )
                ),
                MessagesPlaceholder(variable_name=self.MEMORY_KEY),
                (
                    "human", "用户的最新问题是：{input}\n請根據工具的結果，務必給出最終繁體中文答覆，不允許空白。如果你沒有內容要說'天机不可泄露'；如果没有资料，要说'天下之大非人力所能尽知'。"
                ),
                MessagesPlaceholder(variable_name='agent_scratchpad'),
            ]
        )

        # 记忆
        if chat_message_history is not None:
            self.memory = self.get_memory(chat_message_history)
        else:
            self.memory = self.get_memory(RedisChatMessageHistory(url=REDIS_URL, session_id="default_session"))
        memory = ConversationBufferMemory(
            llm=self.chatmodel,
            human_prefix="用户",
            ai_prefix="陈大师",
            memory_key=self.MEMORY_KEY,
            output_key="output",
            return_messages=True,
            chat_memory=self.memory,
        )
        # 工具列表
        tools = [serp_search,
                get_info_from_local_db,
                bazi_cesuan,
                yaoyigua,
                jiemeng,
                ]

        agent = create_tool_calling_agent(
            self.chatmodel,
            tools=tools,
            prompt=self.prompt,
        )

        self.agent_executor = AgentExecutor(
            agent = agent,
            tools = tools,
            memory=memory,
            verbose = True
        )

    def get_memory(self, chat_message_history):
        # 每次都清空历史，只保留本轮输入
        # chat_message_history.clear()
        return chat_message_history

    def run(self, query):
        logger.info("======================================新的问题开始:======================================")
        logger.info(f"Master.run收到用户输入: {query}")
        # 情绪判断
        emotion = self.emotion_chain(query)
        logger.info(f"大模型判定情绪: {emotion}")
        logger.info(f"当前设定的情绪为: {self.MOODS[self.emotion]['roleSet']}")
        try:
            result = self.agent_executor.invoke({"input": query})
            logger.info(f"Agent执行结果为: {result}")
            # # 如果output为空，尝试从intermediate_steps中提取工具结果
            # if isinstance(result, dict) and (not result.get('output') or str(result.get('output')).strip() == ""):
            #     steps = result.get('intermediate_steps', [])
            #     if steps:
            #         # steps是[(tool_input, tool_output), ...]
            #         last_tool_output = steps[-1][1] if isinstance(steps[-1], (list, tuple)) and len(steps[-1]) > 1 else None
            #         if last_tool_output:
            #             logger.info(f"output为空，自动用最后一个工具结果填充: {last_tool_output}")
            #             result['output'] = str(last_tool_output)
        except Exception as e:
            logger.error(f"Agent执行异常: {e}\n{traceback.format_exc()}")
            result = {"output": f"上天已经警告于我，天机泄露太多，今日已不宜再算: {e}"}
        return result
# 情绪判断（query: str: 用户输入）
    def emotion_chain(self, query:str): # 情绪判断（query: str: 用户输入）
        prompt = """根据用户的输入判断用户的情绪，回应的规则如下：（prompt: 提示词）
        1. 如果用户输入的内容偏向于负面情绪，只返回"depressed",不要有其他内容，否则将受到惩罚。
        2. 如果用户输入的内容偏向于正面情绪，只返回"friendly",不要有其他内容，否则将受到惩罚。
        3. 如果用户输入的内容偏向于中性情绪，只返回"default",不要有其他内容，否则将受到惩罚。
        4. 如果用户输入的内容包含辱骂或者不礼貌词句，只返回"angry",不要有其他内容，否则将受到惩罚。
        5. 如果用户输入的内容比较兴奋，只返回"upbeat",不要有其他内容，否则将受到惩罚。
        6. 如果用户输入的内容比较悲伤，只返回"depressed",不要有其他内容，否则将受到惩罚。
        7. 如果用户输入的内容比较开心，只返回"cheerful",不要有其他内容，否则将受到惩罚。
        8. 如果用户输入的内容是询问某地的风景或者景点，只返回"friendly",不要有其他内容，否则将受到惩罚。
        9. 只返回英文，不允许有换行符等其他内容，否则会受到惩罚。
        用户输入的内容是：{query}"""
        chain = ChatPromptTemplate.from_template(prompt) | self.chatmodel | StrOutputParser()
        result = chain.invoke({"query":query})
        self.emotion = result
        return result


@app.get("/")
@app.get("/index")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 聊天接口（POST请求）（request: Request: 请求）
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    # 获取用户输入的query
    query = data.get("query")
    #给每个用户赋予一个单独的会话id（session_id: 会话id）
    session_id = data.get("session_id", str(uuid.uuid4().hex))
    logger.info(f"用户session_id: {session_id}")
    #ttl 当前会话数据的过期时间，600秒表示10分钟过期（ttl: 过期时间）
    chat_message_history = RedisChatMessageHistory(url=REDIS_URL, session_id=session_id, ttl=600)
    master = Master(chat_message_history) # 创建Master实例
    result = master.run(query)
    # 确保返回的是字符串，并包含session_id（response_data: 响应数据）
    response_data = {"session_id": session_id}
    if isinstance(result, dict): # 如果result是字典
        if 'output' in result:
            logger.info(f"/chat接口最终输出: {result['output']}")
            response_data["output"] = result['output']
        else:
            logger.info(f"/chat接口最终输出(无output字段): {str(result)}")
            response_data["output"] = str(result)
    else:
        logger.info(f"/chat接口最终输出(非dict): {str(result)}")
        response_data["output"] = str(result)

    return response_data
# 添加URL接口（POST请求）（URL: str: URL）
@app.post("/add_urls")
async def add_urls(URL: str):
    loader = WebBaseLoader(URL) # 创建WebBaseLoader实例
    docs = loader.load()
    docments = RecursiveCharacterTextSplitter( # 创建RecursiveCharacterTextSplitter实例
        chunk_size=800,
        chunk_overlap=50,
    ).split_documents(docs)

    #引入向量数据库
    Qdrant.from_documents(
        docments,
        get_lc_ali_embeddings(),
        path="./local_qdrand",
        collection_name="local_documents",
        force_recreate = True
    )

    logger.info("向量数据库创建完成")
    return {"ok": "添加成功！"}

if __name__ == '__main__':
    setup_logger()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

