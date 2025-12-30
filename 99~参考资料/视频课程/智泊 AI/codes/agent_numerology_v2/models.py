#可用模型列表，以及获得访问模型的客户端
#实际使用时可以根据自己的实际情况调整
ALI_TONGYI_API_KEY_OS_VAR_NAME = "DASHSCOPE_API_KEY"
ALI_TONGYI_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ALI_TONGYI_MAX_MODEL = "qwen-max-latest"
ALI_TONGYI_PLUS_MODEL = "qwen-plus"
ALI_TONGYI_TURBO_MODEL = "qwen-turbo"
ALI_TONGYI_DEEPSEEK_R1 = "deepseek-r1"
ALI_TONGYI_DEEPSEEK_V3 = "deepseek-v3"
ALI_TONGYI_EMBEDDING_MODEL = "text-embedding-v3"

import os
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import  DashScopeEmbeddings

def get_lc_ali_model_client(temperature = 0,streaming = True):
    '''
    以OpenAI兼容的方式，通过LangChain获得阿里百炼大模型的客户端
    :return: 指定平台和模型的客户端，默认温度=0.0，流式输出
    '''
    return ChatOpenAI(api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME),
                      base_url=ALI_TONGYI_URL,
                      model=ALI_TONGYI_PLUS_MODEL,
                      temperature=temperature,
                      streaming = streaming)

def get_lc_ali_embeddings():
    '''
    通过LangChain获得一个阿里通义千问嵌入模型的实例
    :return: 阿里通义千问嵌入模型的实例，目前为text-embedding-v3
    '''
    return DashScopeEmbeddings(
        model=ALI_TONGYI_EMBEDDING_MODEL, dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME)
)

