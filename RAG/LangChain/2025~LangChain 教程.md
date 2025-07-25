# LangChain 教程

# 一、LangChain 简介

LangChain 是一个用于开发由语言模型驱动的应用程序的强大框架，它主要具备两大核心能力：一是能够将大规模语言模型（LLM）与外部数据源进行连接，二是允许与 LLM 模型进行交互。其优势显著，例如开发简单快速，无需训练特定任务模型就能适配各种应用，且代码入口简洁，底层核心模块主要包括 Prompt 指定、大模型 API 以及三方应用 API 调用。同时，它的泛用性广，基于自然语言对任务的描述即可进行模型控制，几乎不受任务类型限制。不过，LangChain 也存在一些局限性，比如它主要基于 GPT 系列框架设计，适用的 Prompt 对于其他大模型可能无法达到相同表现，若要更换大模型，底层 prompt 往往需要微调；在实际应用中，基于 Prompt Engineering 进行迭代优化较为困难，因为多一句话或少一句话都可能对效果产生巨大影响 。

# 二、核心模块解析

### （一）模型 I/O（Model I/O）

这是 AI 应用的核心交互部分，涵盖输入、模型交互和输出解析三个关键环节。在输入方面，LangChain 提供了对用户输入内容的 prompts 提示管理，允许在提示中使用变量替代变化的内容，从而更好地引导模型生成符合预期的结果。模型调用环节，通过 prompts 组件可将组装后的提示内容发送给不同的模型，LangChain 已对接众多大模型，如 OpenAI、谷歌的 Palm、微软云的 OpenAI 模型等，极大地方便了开发者为 AI 应用选择合适的模型。对于模型输出的内容，LangChain 提供了解析组件，用于对输出结果进行有效处理 。

### （二）检索（Retrieval）

此模块主要与向量数据紧密相关，旨在实现自建知识库以及检索增强生成（RAG）。其具体操作流程包括文档加载，即从指定源读取数据，LangChain 官方提供了丰富多样的加载器可供使用；读取数据源后，需将数据转换为 Document 对象以便后续处理；接着进行文本分割，使用文本分割器对加载进来的 Document 进行分割；由于数据相关性搜索基于向量运算，所以还需将 Document 数据进行向量化，可通过将数据存储到对应的向量数据库中来完成这一转换 。

### （三）链（Chains）

Chain 可理解为任务，一个 Chain 代表一个任务，也能够像链条一样依次执行多个任务。它将 LLMs 和 prompts 结合起来，通过提供的大模型封装和问题字符串模板，即可执行并获得返回结果。例如，通过 LLM 的 llm 变量和 Prompt Templates 的 prompt 可生成 LLMChain，进而运行实际问题。同时，Chain 分为普通 Chain 和 LangChain 表达式（LCEL），LCEL 是更新后的方法，两者可独立使用，且 LCEL 里也能调用 Chain 。LCEL 可用于构建和优化各种自动化和数据处理链条，比如在开发自动化内容摘要系统时，可使用 LCEL 声明性地描述如何从文章中提取关键信息，或者结合不同数据源改进摘要质量 。

### （四）记忆（Memory）

Memory 模块负责以合适的方式存储对话内容。当用户发送问题后，它会先读取历史会话内容，待模型返回答案之后，再将本次交互内容进行存储，从而在长对话过程中，能够随时重新加载历史对话记录，保证对话的准确性和连贯性 。

### （五）代理（Agents）

Agents 能够基于用户输入动态地调用 chains。它可以理解用户的意图，返回特定的动作类型和参数，进而自主调用相关工具来满足用户需求，使应用更加智能化。例如，LangChain 可以将问题拆分为几个步骤，每个步骤根据提供给 Agents 的信息做相关处理。在实际使用中，如通过导入特定工具（如 llm - math 能进行数学计算），初始化 tools、models 并选择使用的 agent，即可实现复杂功能 。

### （六）回调（Callbacks）

回调在 LangChain 中提供了一种强大的方式来干预和管理 LLM 应用的不同阶段。在日志记录和监控方面，可使用回调记录 LLM 的每次调用或监控链条的性能；处理流式数据时，能通过回调处理每个新生成的令牌或消息；出现错误时，可利用回调触发错误处理逻辑，例如重新尝试或发送警报；在聊天模型或交互式应用中，还能通过回调处理用户的每个输入或模型的每个响应 。

# 三、常用工具介绍

### （一）Templates

Templates 包含了一些已经编写好的常用功能模版，开发者可以直接拿来使用，例如构建 RAG 聊天机器人、从非结构化数据中提取结构化数据等场景，都能借助这些模板快速开展工作 。

### （二）LangSmith

LangSmith 是一个强大的工具，可帮助开发者跟踪和评估开发的 AI 应用。它为 LangChain 生态增添了运行监测与分析的能力，能够收集 LLM 应用的各类运行指标，并进行分析展示，让开发者更深入地理解应用的运行状况，从而自信地不断改进和部署应用 。

### （三）LangServe

LangServe 用于将 LangChain 应用转换为可以通过 REST API 访问的服务。开发者的程序实际部署在自己的服务器或云环境上，而 LangServe 负责创建和管理这些服务的 API 接口，极大地方便了应用的部署 。

# 四、安装与配置

### （一）安装 LangChain 库

使用 pip 命令进行安装，在命令行中输入：

```
pip install langchain
```

若要安装特定版本，可指定版本号，如：

```
pip install langchain==0.0.1
```

此外，由于 LangChain 常与 OpenAI 等模型配合使用，还需安装相应的模型库，以 OpenAI 为例：

```
pip install openai
```

### （二）配置 OpenAI 密钥

1.  **代码中直接写入**：在 Python 代码中，可通过以下方式设置 OpenAI 密钥：

```
import os


os.environ["OPENAI_API_KEY"] = "你的OpenAI密钥"
```

1.  **数据库中写入，代码中调用**：将密钥存储在数据库中，然后在代码中编写相应的数据库连接和读取逻辑来获取密钥 。

2.  **环境变量中配置**：在系统环境变量中添加 OPENAI_API_KEY 变量，并将其值设置为你的 OpenAI 密钥。以 Windows 系统为例，可通过 “系统属性” -> “高级” -> “环境变量” 进行设置 。

# 五、Hello World 示例

下面通过一个简单的示例展示 LangChain 的基本使用方法。

```py
from langchain.chat_models import ChatOpenAI

from langchain.chains import LLMChain

from langchain.prompts import PromptTemplate

# 初始化ChatOpenAI模型


llm = ChatOpenAI(model_name="gpt - 3.5 - turbo", temperature=0.7)

# 创建提示模板，定义模型的输入格式


prompt = PromptTemplate(


   input_variables=["question"],


   template="请回答这个问题：{question}"

)


# 创建语言模型链，将提示模板与模型结合


chain = LLMChain(llm=llm, prompt=prompt)


# 运行链，向模型提问


response = chain.run("什么是人工智能？")


print(response)
```

在上述代码中，首先导入所需的模块，然后初始化 ChatOpenAI 模型，设置模型名称和 temperature 参数（temperature 值越高，生成文本的随机性越强）。接着创建一个 PromptTemplate，定义了模型输入的格式，这里使用了一个变量 “question”。之后将提示模板与模型结合创建 LLMChain，最后通过运行 LLMChain 并传入问题，即可得到模型的回答 。

# 六、拓展应用示例

### （一）构建能联网搜索的 AI Agent

```py
from langchain_community.tools import DuckDuckGoSearchRun

from langchain_community.chat_models import ChatOpenAI

from langchain.agents import initialize_agent, AgentType

# 实例化DuckDuckGo搜索工具，作为agent可用的工具之一


search = DuckDuckGoSearchRun()


tools = [search]


# 创建一个LLM（大型语言模型）客户端，连接到DeepSeek的OpenAI兼容接口


llm = ChatOpenAI(


   model='deepseek - chat',


   openai_api_key="你的API密钥",


   openai_api_base='https://api.deepseek.com',


   max_tokens=1024

)


# 初始化智能体Agent，使用Zero - Shot ReAct模式（零样本推理 + 工具调用）


agent = initialize_agent(


   llm=llm,


   agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,


   tools=tools,


   verbose=True,


   handle_parsing_errors=True

)


response = agent.invoke({"input": "请用中文回答这个问题：最近有哪些关于AI Agent的新研究？"})


print(response)
```

在这个示例中，首先导入了 DuckDuckGoSearchRun 工具用于联网搜索信息，以及 ChatOpenAI 类用于与兼容 OpenAI 接口的语言模型交互。然后实例化搜索工具，并将其添加到工具列表中。接着创建 LLM 客户端，连接到 DeepSeek 的 OpenAI 兼容接口，并设置相关参数。最后初始化智能体 Agent，选择 Zero - Shot ReAct 模式，并传入工具列表等参数。运行 Agent 时，它会自动搜索并整合内容回答问题 。

### （二）对超长文本进行总结

```
from langchain.document_loaders import TextLoader

from langchain.text_splitter import CharacterTextSplitter

from langchain.embeddings import OpenAIEmbeddings

from langchain.chains.summarize import load_summarize_chain


from langchain.chat_models import ChatOpenAI

# 加载文档


loader = TextLoader('your_long_text_file.txt')


documents = loader.load()


# 文本分割


text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)


texts = text_splitter.split_documents(documents)


# 初始化语言模型和总结链


llm = ChatOpenAI(temperature=0)


chain = load_summarize_chain(llm, chain_type="map_reduce")


# 运行总结链


summary = chain.run(texts)


print(summary)
```

此示例用于对超长文本进行总结。首先通过 TextLoader 加载文本文件，然后使用 CharacterTextSplitter 将文本分割成较小的块，以适应模型处理能力。接着初始化 ChatOpenAI 模型和总结链，这里选择了 “map_reduce” 类型的总结链。最后将分割后的文本块传入总结链中运行，得到文本的总结内容 。

希望通过本教程，你能对 LangChain 有一个全面且深入的了解，并能够运用它开发出各种强大的语言模型驱动应用。在实际应用中，你可根据具体需求灵活组合和拓展各模块与工具，不断探索 LangChain 的更多潜力 。
