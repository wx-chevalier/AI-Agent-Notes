# 1 Agent 架构综述：从 Prompt 到上下文工程构建 AI Agent

过去两三年，我曾为多家公司的资深开发人员开展 Agent 开发培训；最近一个月，我也一直在为毕业生设计和培训 AI Agent。直到本周， 结合 AI 能力的模拟项目 showcase 完成后，我才真正理清楚：如何针对不同阶段的开发者，系统地构建 Agent 的学习路径。这个过程， 也让我深刻体会到“知识诅咒”的存在——原来自己习以为常的知识，对于初学者来说，可能是最大的障碍。

我们可以简单把学习过程分为四部分：

- 结构化提示词工程 —— 如何工程化设计高效、可复用的提示词。
- 上下文工程与知识检索 —— 知识检索、生成与压缩上下文信息，产生高质量的知识背景。
- 工具函数的系统化设计 —— 设计并实现可供 Agent 调用的工具与接口。
- Agent 规划与多 Agent —— 构建任务规划与执行路径，实现闭环自动化。

开始之前，按理来说，我们需要简单定义一下 AI Agents，而考虑到 Agent 会有大量的不同定义 => 可以参考 Anthropic 在《Building effective agents》中给出真实世界中的示例：

> 一些客户将其定义为完全自主运行的系统，能长期独立运作并使用多种工具完成复杂任务；另一些则用它描述遵循预定义工作流程的系统。

因此，围绕一个提示词的简单任务也可以视为一个 AI Agent；而围绕一个复杂系统、多工具、多步骤的任务，也是一个 AI Agent。

## 1.1 结构化提示词工程

尽管 Context Engineering 是一个非常火的词，但是如何写好 Prompt，依然是我们要入门的重点。网上已经有非常多的提示词相关的内容， 但是从我的经验来看，我们可以把重点放在三个部分上：

- 提示词输入与输出的结构化
- 复杂问题的链式与模块化设计
- 提示词路由分发任务

再配合上一些必要的 AI 框架或者工具，就能非常不错的完成我们的任务。

### 1.1.1 提示词输入与输出的结构化

在现在的开发 Agent 的过程中，尽管模型能生成一部分提示词，但是调提示词依然是工作的重点。我们希望模型输出的内容可以是 JSON、XML 或 Java 类，以便结合其它代码一起使用。

> 提示词（Prompts）是用于引导 AI 模型生成**特定输出**而输入设计艺术与科学。通过对输入的精心设计与措辞，可以有效影响并控制模型的响应方向与结果， 使 AI 生成符合预期的输出。

我们可以直接看 Spring AI 文档的 Structured Output Converter 作为示例：

[![Structured Output Converter](https://github.com/phodal/build-coding-agent-context-engineering/raw/master/images/structured-output-architecture.jpg)](https://github.com/phodal/build-coding-agent-context-engineering/blob/master/images/structured-output-architecture.jpg)

图中的黄色部分即是两个核心：

**格式化的输入指令**

通常来说，我们需要结构提示词模板来动态生成提示词，采用结构化的文本来设计输入：

- 动态提示词模板（PromptTemplate）。采用经典的模板引擎动态结合上下文，如 LangChain 里的 Jinja2，Spring AI 中的 StringTemplate。这种方式允许在运行时注入上下文、用户输入、系统状态等信息，实现灵活的 Prompt 构建。
- 结构化的文本结构。为了保证 AI 输出的可靠性和可解析性，需要对提示词进行结构化设计，包括角色定位（Role）、任务描述（Task）、 约束条件（Constraints）、输出格式等。
- 示例驱动。通过提供示例输入（Few-shots）与期望输出，可以显著提高模型输出的稳定性和一致性。诸如在实现 QA 的时候会给不同场景的实现示例。

**转换模型输出结果**

即针对不同的场景采用合适的输出格式，并实现对应的解析实现与**异常场景处理**。

- 领域特定的输出格式。我们会基于场景的不同，采用 JSON、XML、YAML 或者 Markdown 等不同的设计，以用户体验更好的方式来展示。诸如： JSON 的优点是可直接序列化传输，但是不能实时渲染体验差，不够健壮。YAML 则能更好地处理流式的问题，并且传输成本更低。
- 解析实现。从纯文本中解析出代码块，再进行反序列化与对象映射等处理。使用 Schema 验证（JSON Schema、XSD）确保模型输出字段类型和结构符合约定。
- 异常场景处理。由于模型生成存在不确定性，输出可能存在缺失、类型错误或不符合约定格式的情况。诸如：字段缺失时，使用默认值或回退策略，可触发模型重试生成特定字段

在能力适当的时候，可以基于已有的数据等信息，微调/训练模型来提升在这方面的能力。

### 1.1.2 提示词路由分发任务

在复杂的 AI 系统中，尤其是多 Agent 或多模块协作的场景下，单个提示词往往无法完成所有任务。所以我们需要提示词路由：

[![https://www.ibm.com/think/topics/prompt-chaining](https://github.com/phodal/build-coding-agent-context-engineering/raw/master/images/prompt-chaining-example.png)](https://github.com/phodal/build-coding-agent-context-engineering/blob/master/images/prompt-chaining-example.png)

> 提示词路由（Prompt Routing） 是在多任务、多 Agent 或复杂 AI 流程中，将任务拆分、分析输入并智能分配给最合适模型或子任务提示词的工程化模式。

其核心思想是：通过分析输入和上下文，动态决定信息处理路径、使用哪条提示词或调用哪个工具、子 Agent，从而实现非线性、条件化的任务执行。以典型的 QA 场景为例：

- 非系统相关问题 → 直接告诉用户不支持该种类型的问题
- 基础知识问题 → 调用文档检索和 QA 模型
- 复杂分析问题 → 调用数据分析工具，再生成总结
- ……

通过提示词路由，系统可以根据问题类型智能选择最合适的处理方式，同时保持模块化和可扩展性。在一些 AI 框架里， 诸如 LangChain 里的 RouterChain 就可以提供类似的能力支持，还有诸如于 [Routing by semantic similarity](https://python.langchain.com/docs/how_to/routing/#routing-by-semantic-similarity) 这种方式。

### 1.1.3 复杂问题的链式与模块化设计

在有提示词路由的前提下，复杂问题可以通过**提示链（Prompt Chaining）**进行系统化拆解。提示链允许将一个大任务拆分为多个子任务， 每个子任务对应不同的提示词或模型调用，最后将结果整合。这种方式比较适合于有固定的流程，并且有一些步骤是可以跳过的。

[![https://arxiv.org/pdf/2308.11432](https://github.com/phodal/build-coding-agent-context-engineering/raw/master/images/task-decomposition.jpg)](https://github.com/phodal/build-coding-agent-context-engineering/blob/master/images/task-decomposition.jpg)

这可以实现更好的模块化 设计：

- 每一个子任务都专注于处理特定阶段任务
- 可以按需重写某个子任务，增加或替换提示词
- 根据前一阶段输出动态调整后续提示词

以常见的软件需求为例，产品经理提出的想法可以通过提示链拆解为：

[![img](https://github.com/phodal/build-coding-agent-context-engineering/raw/master/images/knowledge-example.png)](https://github.com/phodal/build-coding-agent-context-engineering/blob/master/images/knowledge-example.png)

1. 创意收集：收集产品创意与初步需求
2. 需求逻辑梳理：理顺需求逻辑和功能优先级
3. 需求预排期：形成初步的需求文档或任务列表
4. 需求定稿：确认最终需求，生成正式文档

每个环节可以由不同的提示词或子 Agent 处理。例如，创意收集可借助具备搜索功能的 AI Agent，需求逻辑梳理可使用 Dify、 Copilot 365 等工具完成。最终，各环节按链式流程执行，同时保持模块化设计的灵活性，可根据需要随时调整或替换子任务。

## 1.2 上下文工程与知识检索

通常来说，我们有 NoCode 和 ProCode 来支持带上下文的 Agent 开发。

- NoCode 方案（适合快速验证）：用低代码平台（如 Dify、N8N、Coze 等）和预配置 RAG 管道，通过 UI 快速配置检索策略。
- ProCode 方案（适合定制化需求）：用框架（LangChain、Spring AI）自定义检索流程和优化策略，能实现多阶段 HyDE + 混合检索 + 重排序管道。

[![img](https://github.com/phodal/build-coding-agent-context-engineering/raw/master/images/context-engineering.jpg)](https://github.com/phodal/build-coding-agent-context-engineering/blob/master/images/context-engineering.jpg)

上下文本身也是提示词的一部分，在没有实现各种自动化之前，我们通常会手动从文档中复制到 AI 聊天工具中。只是随着，我们进入了模型的深水区之后， 我们就需要开始思考自动化的构建方式，也就是用工程化的角度来思考这个问题。开始之前，我们依旧需要定义 AI Agents，这里我们可以引用 Anthropic 官方的 《Effective context engineering for AI agents》给的定义（因为它同样也有科学和艺术）：

> 上下文工程是一门将不断变化的信息宇宙中最相关内容，精心筛选并放入有限上下文窗口的艺术与科学。

### 1.2.1 上下文窗口

简单来说：关注在有限的上下文窗口中挑选最关键的信息，让模型理解和推理更高效。如下是 [Langchain 绘制](https://github.com/langchain-ai/how_to_fix_your_context)的 [Drew Breunig 的总结](https://www.dbreunig.com/2025/06/26/how-to-fix-your-context.html) 的 6 种常见上下文工程技术：

[![6 common context engineering techniques](https://github.com/phodal/build-coding-agent-context-engineering/raw/master/images/langchain-context-enginneering.png)](https://github.com/phodal/build-coding-agent-context-engineering/blob/master/images/langchain-context-enginneering.png)

在这里，我会将其简述为：RAG 与上下文窗口的工程化。一个完整的上下文窗口的内容（即 prompts）通常应该包含：

- 系统提示词部分：
  - 输入指令上下文：告诉它“你是谁”“你要做什么”，包括系统提示、用户输入，还有角色定义。
  - 格式化输出上下文：指定模型输出格式的结构化模式，例如要求以 JSON 格式返回，以确保输出的可用性 。
- 函数调用部分：
  - 工具相关上下文：这赋予了模型与外部世界交互的能力。它包括可用工具或函数的定义，以及调用这些工具后返回的响应 。
- 动态上下文部分：
  - 时间与记忆上下文：短期记忆（Short-Term Memory）、长期记忆（Long-Term Memory）。
  - 外部知识上下文：从文档、数据库等外部信息库里查出来的事实，让模型少犯“胡说八道”的错误。
  - 全局状态/暂存区：模型处理复杂任务时的临时存储，相当于它的“工作记忆”。
  - 外部知识：通过检索增强生成（RAG）等技术，从外部知识库（如文档、数据库）中检索出的信息，用于为模型提供事实依据并减少幻觉 。

除了固定的系统提示词部分，**外部知识的获取**与**记忆**会最大化影响整个窗口，因此对于它们俩的设计与优化便是上下文工程的重中之重。

### 1.2.2 知识的检索增强生成

[![Spring AI RAG](https://github.com/phodal/build-coding-agent-context-engineering/raw/master/images/spring-ai-rag.jpg)](https://github.com/phodal/build-coding-agent-context-engineering/blob/master/images/spring-ai-rag.jpg)

> RAG（检索增强生成，Retrieval-Augmented Generation）是构建 Agent 的核心技术之一，它通过从外部知识库中检索相关信息来增强 大语言模型的生成能力。在代码库问答等复杂场景中，单纯的向量检索往往不够精准，需要组合多种检索策略来提升准确率。

简单来说，就是通过搜索来丰富上下文。根据实现复杂度和场景需求，我们可以将检索策略分为以下几类：

- 关键词检索（Keyword Search）

  。最基础的检索方式，适合精确匹配场景。诸如在代码库中搜索特定的函数名、类名或变量名时，关键词检索往往比语义检索更有效。常见实现包括：

  - **全文检索**：使用 Elasticsearch、Solr 等搜索引擎，采用诸如 BM25, TF-IDF 等算法。
  - **正则表达式匹配**：如 ripgrep、grep 等工具，Cursor 就采用了 ripgrep + 向量检索的混合方式

- 语义化检索（Semantic Search）

  。通过向量嵌入（Embeddings）来理解查询的语义含义，而不仅仅是字面匹配。这对于自然语言查询尤为重要：

  - 使用预训练的嵌入模型（如 OpenAI text-embedding-3-large、Jina embeddings v3）将文本转换为向量
  - 在向量空间中计算查询与文档的相似度（通常使用余弦相似度或点积）

- 图检索（Graph-based Search）

  。图检索不仅关注“内容相似”，更注重关系与上下文依赖。

  - 代码场景下：构建代码调用关系图、依赖关系图，利用 AST（抽象语法树）提取方法、类、构造函数等结构
  - 示例：微软的 [GraphRAG](https://github.com/microsoft/graphrag)，Aider 的 repomap，又或者是 Joern，CodeQL 这种基础设施

而在检索之前，为了确保生成的检索结果可靠，需要引入**查询改写（Query Rewriting）**，即将用户的模糊意图逐步转化为数据库能够高效执行的精确查询。 修改用户的原始查询来提升其与知识库中文档的相关性，解决了自然语言问题与存储数据块之间的“阻抗不匹配”问题 。

#### 1.2.2.1 代码场景下的 RAG 示例

通常来说，多种不同的检索策略可以组合使用，以提升检索效果。如下是向量数据库 LanceDB 官方给的一个[Codebase RAG 实现](https://blog.lancedb.com/rag-codebase-1/)：

[![img](https://github.com/phodal/build-coding-agent-context-engineering/raw/master/images/codebase-indexing-with-graph.png)](https://github.com/phodal/build-coding-agent-context-engineering/blob/master/images/codebase-indexing-with-graph.png)

除了在 indexing 阶段使用了 TreeSitter 来生成知识，在[检索阶段](https://blog.lancedb.com/building-rag-on-codebases-part-2/) 还会使用：

- HyDE（假设性文档嵌入）：先让模型根据查询生成一个“假设性”文档或代码片段，再用这个生成的内容做向量搜索，这样更容易找到语义相关的代码。
- BM25（关键字搜索）：传统的关键字搜索算法，擅长找包含精确术语或 API 名称的代码，也可以和向量搜索配合使用。
- 混合检索（Hybrid Search）：把 BM25 和语义搜索结合起来，既能精确匹配关键字，也能理解代码语义，通过调整两者权重获得更优结果。
- 重排序（Re-ranking）：在向量搜索得到初步结果后，再用交叉注意力机制对结果重新排序，提高最终答案的相关性和准确度。

当然了，在前面的 indexing 阶段，这个示例还会生成**元特征数据**，即对每个元素或代码片段，我们先生成代码的文本描述，再将该描述进行向量嵌入， 以获取代码的所有元特征，其特征是通过微调后的 LLM 提取的。

### 1.2.3 上下文窗口的工程化

[![https://docs.claude.com/en/docs/build-with-claude/context-windows](https://github.com/phodal/build-coding-agent-context-engineering/raw/master/images/context-window.svg)](https://github.com/phodal/build-coding-agent-context-engineering/blob/master/images/context-window.svg)

两年前，[GitHub Copilot](https://code.visualstudio.com/docs/copilot/chat/prompt-crafting) 为补全为构建的上下文系统是业内 最值得研究的上下文系统（没有之一）：

- 持续的信号监控。Copilot 插件会持续监控来自 IDE 的一系列信号，以动态调整上下文的优先级。诸如于插入或删除字符、当前编辑的文件和语言的改变，光标移动、滚动位置变化、文件的打开与关闭。

- 上下文来源的

  优先级排序

  。在发给模型的最终提示词里，会根据优化级来进行排序和筛选：

  - 最高优先级：光标位置周围的代码，包括光标前后的内容，这是最直接的上下文 。
  - 高优先级：当前正在编辑的文件的其余部分。
  - 中等优先级：在 IDE 中打开的其他文件或标签页（即“邻近文件”）。
  - 辅助上下文：其他信息也被纳入考量，包括文件路径、仓库 URL、代码中的导入语句（imports），以及 RAG 检索到的代码信息。

- 上下文长度约束下的提示词组装。根据上述优先级对每个信息片段进行“评分”，然后组装出一个最优的提示。

这也就可以为我们提供一个非常不错的参考：

- 新鲜度优先。最近编辑或访问的内容获得更高优先级，过时内容权重逐步衰减。
- 信号融合与动态评分。融合多种编辑信号（如光标移动、文件切换、导入变更等），动态调整上下文权重。
- 滑窗与增量更新。采用滑动窗口机制，仅对变化部分进行增量更新，避免全量重建。
- 预算感知与自动截断。实时估算 token 占用，接近限制时自动裁剪或摘要低优先级内容。

当然这是一种非常复杂的设计，只值得你在足够高价值的系统中采用这样的设计。而结合现在流行的各种 Cursor Rule/Spec，采用诸如 AGENTS.md 用持久化记忆 （Memory System）存储跨会话的关键信息，为后续查询提供长期背景信息。

### 1.2.4 Agentic 检索

[![https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/](https://github.com/phodal/build-coding-agent-context-engineering/raw/master/images/agentic-rag.png)](https://github.com/phodal/build-coding-agent-context-engineering/blob/master/images/agentic-rag.png)

> Agentic 指的是一种让 AI 系统具备 自主感知、动态决策与目标导向执行能力 的特性，使其能够在任务过程中主动优化上下文、生成检索策略并持续自我迭代。

在 AI Coding 领域，诸如 Cursor、Claude Code 等，我们可以观察其运行的过程，其本质是 Agent 来执行 RAG。它与普通的 RAG 相比，它更加容易拿到 丰富的上下文，进而确保上下文在整个过程中是不缺失的。我们可以看到现有比较成熟的一些 AI 应用的示例：

- Cursor 会优化采用 `file + ripgrep` 的方式来直接检索代码，在结果不够多时，再调用向量化检查或者 Git 历史等相关的检索
- Google DeepResearch 在生成的过程中，也是类似的过程来完成某个研究：上下文工程主流工具的识别、工具功能和差异的初步掌握、下一步行动：工具细节的深入挖掘

简单来说，对于复杂的检索，我们可以将其构建为一个 Agent，由 Agent 来判断采用何种检索工具和策略，在上下文不够的时候，继续使用新的参数调用工具 以拿到足够的上下文。

#### 1.2.4.1 DeepResearch 示例

如下是 Langchain AI 构建的 [Open DeepResearch](https://github.com/langchain-ai/open_deep_research) 过程示例：

[![img](https://github.com/phodal/build-coding-agent-context-engineering/raw/master/images/open-deep-research.jpg)](https://github.com/phodal/build-coding-agent-context-engineering/blob/master/images/open-deep-research.jpg)

Deep Research Agent 展示了一种更加系统化的 Agentic 检索方法：

1. 将任务拆分为规划阶段（Manager Agent） 和执行阶段（Execution Agent）
   - Manager Agent 负责任务理解、子任务拆分以及检索策略设计
   - Execution Agent 负责实际搜索、网页或文档抓取、内容解析
2. 在检索过程中，Agent 会维护对主题结构、已覆盖子问题和信息缺口的状态，以决定下一步探索方向
3. 可在关键阶段插入用户审查（HITL 模式），增强控制与精度
4. 最终，Agent 会将收集到的碎片信息整合成结构化报告，并附带来源引用

通常观察他们的交互与思考过程，会更好地帮助我们去理解这个过程。在此基础上，也可以看到诸如 Agentic Context Engineering 通过进一步让 LLM 自主生成、整理和迭代上下文，实现智能化、可扩展的上下文管理，从而优化复杂任务的检索与推理效率。

[![https://www.arxiv.org/pdf/2510.04618](https://github.com/phodal/build-coding-agent-context-engineering/raw/master/images/agentic-context-engineering.jpg)](https://github.com/phodal/build-coding-agent-context-engineering/blob/master/images/agentic-context-engineering.jpg)

即，基于历史的会话或者经验来优化如何检索，以让 Agent 更加适合于场景。

## 1.3 Agent 工具系统的工程化设计

在构建 Agent 的过程中，工具系统（Tool System） 的设计是最能体现工程思维的一环。它决定了 Agent 能做什么、能做多好，以及能否与外部世界高效协作。 工具可以是一切的 API，如数据查询（如数据库访问）、现实世界操作（如发送邮件、预订会议）或与其他服务协同的接口。如我们在前面提到的 RAG 在 Agentic 下也是 工具的一种类型，诸如 LlamaIndex 便提供了这种显式的封装：

- FunctionTool：可以将任何 Python 函数轻松封装成一个代理可用的工具。
- QueryEngineTool：可以将任何数据查询引擎（例如，一个向量索引）转换为一个工具，使代理能够在其上进行查询和推理

这种围绕数据为中心的方式，可以简化我们对工具的理解。

### 1.3.1 语义化工具：为 Agent 设计的函数接口

**工具**本质上是一类语义可理解的函数接口。它们不仅包含逻辑执行能力，更携带了让模型理解的元信息：

- 名称（name）: 工具的唯一标识符，通常是函数名，例如 `getWeather`。
- 描述（description）: 对工具功能、目的和适用场景的自然语言描述。这是至关重要的一环，因为模型主要依赖这段描述来判断何时以及如何使用该工具 。
- 参数（parameters）: 一个定义工具输入参数的对象，包括每个参数的名称、数据类型（如 string, number）、描述以及是否为必需参数 。

在执行机制上，常见的两种范式是：

- ReAct 框架 (Reasoning + Acting)：ReAct 范式的核心是让 LLM 交错地生成“思考”（推理轨迹）和“行动”（工具调用）， 从而形成一个显式的思考-行动-观察循环 。
- 直接函数调用 (Direct Function Calling) ：这是一种更为结构化的方法。LLM 在单步推理中判断用户的查询可以通过调用一个或 多个预定义函数来最好地解答。然后，它会输出一个结构化的 JSON 对象，明确指出需要调用的函数名称及其参数 。

我们需要根据模型的支持情况，以及设计的交互、意图来决定使用哪种方式调用。

### 1.3.2 工具的设计原则

通常来说，在构建 Coding Agent 的时候，我们会遵循如下的原则：

- 语义清晰：工具的名称、描述以及其参数的命名必须对 LLM 来说是极度清晰、富有描述性且无歧义的。工具的 description 字段视为一种面向 AI 的 “微提示词” 来精心撰写
- 无状态的**客观**函数：只封装复杂的技术逻辑或领域知识，避免做出战略性或主观性的决策
- 原子性与单一职责：每个工具都应只负责一个且仅一个明确定义的功能，即执行一个原子操作。如果是 Agent 作为工具也应该遵循类似的原则，只完成一件事情。
- 最小权限：每个工具应该只被授予完成其明确定义的任务所必需的最小权限和能力。

#### 1.3.2.1 基于 Workflow 的工具编排：任务链式的设计

它也适用于在非编程领域的 AI Agent。基于上述的原则，我们可以将 “为我计划下周去北京的旅行”——分解为一组离散的、具有单一职责的工具。

- search_flights(origin: str, destination: str, outbound_date: str, return_date: str): 搜索航班信息。
- search_hotels(location: str, check_in_date: str, check_out_date: str, adults: int): 搜索酒店信息。
- get_local_events(query: str, date: str): 获取特定日期的本地活动或景点信息。
- book_cruise(passenger_name: str, itinerary_id: str): 预订邮轮行程。
- lookup_vacation_packages(query: str): 查询度假套餐

这种编排方式的关键特征是：可预测性强、逻辑清晰、易于在平台中建模为可视化流程（如 DAG）。它非常适合 流程稳定、任务有依赖关系的 Agent（如旅行、客服、数据管道类场景）。

#### 1.3.2.2 基于分类的工具调用：动态意图决策

诸如 [Understanding GitHub Copilot’s Internal Architecture](https://iitsnl.com/blog/understanding-github-copilots-internal-architecture/) 中介绍的 Copilot 编排器会根据“意图分类器”（Intent Classifier）对用户请求的分析结果，决定调用一个或多个内部工具来完成任务：

- 文件操作：包括 `read_file`（读取文件）、`edit_file`（编辑文件）和 `create_file`（创建新文件）等，使 Copilot 能够直接与用户的代码库进行交互 。
- 代码执行：通过 `run_in_terminal` 工具，Copilot 可以在用户的终端中执行命令，例如运行测试或构建脚本 。
- 搜索与分析：这是最关键的工具集之一，包括传统的 `grep_search`（文本搜索）、`list_code_usages`（列出代码引用），以及最强大的 `semantic_search`**（语义搜索）** 。

这种模式的关键特征是：灵活性高、可扩展性强，但依赖良好的分类体系与语义匹配能力。它更适合动态场景，如代码生成、调试、文档问答等。

### 1.3.3 采用 MCP 协议，构建可组合的工具网络

当工具数量与 Agent 数量不断增长时，我们需要一种机制来标准化描述、动态注册与跨 Agent 调用工具。 MCP（Model Context Protocol） 正是为此设计的通用协议层。通过 MCP，AI Agent 不再依赖硬编码的接口或特定系统，而是能够以统一格式调用工具、 获取数据或与其他 Agent 协作。MCP 的核心价值在于标准化、动态化和可组合：

- 标准化：统一工具调用格式，使不同 Agent 可以共享工具集。
- 动态化：支持运行时注册和访问工具， Agent 可以根据任务需求选择最合适的工具。
- 可组合：不同 Agent 和工具之间可以像搭积木一样组合，实现复杂任务的分解与协同执行。

结合前面设计的原子工具函数，MCP 可以将这些工具整合为一个可复用、可协作的工具网络，让 Agent 在解决复杂问题时更加灵活和高效。

#### 1.3.3.1 其它工具网络

另外我们也可以看到 GitHub Copilot Extension，或者 Claude Code Plugin 这样的出现也在预示着，哪怕有了 MCP 和 A2A 这样的协议， AI Agent 生态并不会如我们预料的那么统一。诸如 https://github.com/wshobson/agents 项目就记录着（2025.10.14）：

> 一个面向生产环境的综合系统，由 84 个专用 AI Agent、15 个多 Agent 工作流编排器 和 44 个开发工具 组成，这些工具被组织为 62 个聚焦且单一职责的插件，用于 Claude Code。

## 1.4 Agent 规划与超越单体 Agent

> Agent 是使用 AI 来实现目标并代表用户完成任务的软件系统。其表现出了推理、规划和记忆能力，并且具有一定的自主性，能够自主学习、适应和做出决定。 - Google Cloud

Agent 是目标为导向的，为了实现目标通常情况下需要**感知**-**规划**-**行动**，还有记忆，而复杂的 AI Agent 系统则会包含**协作**、**自我完善**等能力。而在前面的内容里，我们已经介绍了几个基本的能力：

- 通过**结构化提示词与提示链**，Agent 具备了规划与决策的思维结构；
- 通过**上下文工程**，Agent 获得了“感知世界”的能力，能够从外部知识与环境中捕获信息；
- 通过**工具系统** 的工程化设计，Agent 获得了与外部世界交互、执行任务的行动力。

基于此之上，Agent 进一步发展的方向在于：

- **协作（Collaboration）** —— 多 Agent 之间通过 A2A（Agent-to-Agent）通信协议或任务分配机制协同工作，实现角色分工与信息共享；
- **自我完善（Self-improvement）** —— Agent 通过记忆系统与反思机制积累经验，优化自己的提示词与规划策略，从而具备持续学习与自我演化能力。

而由于这是一个快速发展的领域，

### 1.4.1 模块化的系统提示词：Agent 的思维蓝图

构建一个有效的 Agent 的第一步，是定义它的“思维蓝图”——即系统提示词（System Prompt）。优秀的系统提示词设计不仅定义了 Agent 应该做什么，也明确了不应该做什么。在 Coding Agent 领域，一个 Agent 的系统提示词往往极为复杂。例如 [Cursor](https://github.com/x1xhlol/system-prompts-and-models-of-ai-tools/blob/main/Cursor Prompts/Agent Prompt 2025-09-03.txt) 的系统提示词中，包含了关于角色、工具调用、安全边界、任务规划等详细规范。

结合 Cursor、Claude Code、Augment、Junie 等工具，我们可以总结出一系列模块化设计实践：

- **结构分层与模块化**：用清晰的层次（角色/通信/工具/安全/任务）组织提示，避免“大一统”文本，便于维护与动态加载。
- **工具优先级与并行化**：优先专用工具且能并行就并行，显著降低延迟与成本（如并行调用 `read_file` 读取多文件，编辑用 `search_replace` 而非 sed）。
- **安全边界与权限模型**：默认沙箱最小权限，危险操作需显式授权（如 `required_permissions: ["network"|"git_write"|"all"]`），禁止对 `main/master` 强推等高风险动作。
- **任务管理最小充分**：多步骤复杂任务用 TODO 管理（创建后第一个标记为 in_progress，完成即刻 completed），简单直接任务立刻执行。
- **上下文唯一性与安全修**改：代码编辑要求唯一可定位的上下文（`old_string` 在文件中唯一，前后各 3–5 行），多处修改分次执行，避免误改。
- **交流规范与用户体验**：隐藏内部工具名，用自然语言“说-做-总结”，保持简洁可扫读；用 backticks 标注文件/函数名，必要时给最小可用示例

这种从单体提示词向模块化、层次化、动态化演进的设计，正如从单体应用向微服务架构的转变，为 Agent 的高级推理、系统可扩展性与可维护性提供了结构支撑。

### 1.4.2 从检索到规划：使用 Prompt 让 Agent 拆解目标

仅仅告诉 Agent “制定一个计划”是远远不够的，我们必须通过一套明确的原则来指导其分解过程，就像为软件模块制定规约一样。 单体 Agent 的智能上限，往往取决于其“规划能力”——能否将模糊目标拆解为明确的、可执行的子任务。

这涉及两种核心策略：

- **预先分解**：这种策略也被称为静态规划，它在任务执行开始之前，就将整个复杂任务完整地分解成一个子任务序列或计划。
- **交错分解**：这种策略也被称为动态规划，它不在任务开始时制定完整计划，而是在执行过程中动态地决定下一个子任务。

例如，BabyAGI 的架构就体现了这种“任务驱动”型规划： 它包含三个核心 Agent —— task_creation_agent（任务生成）、execution_agent（任务执行）和 prioritization_agent（任务优先级排序），形成了一个不断循环的任务更新与执行系统。

而在现代系统（如 Augment、Claude Code）中，规划逻辑往往以 todo_spec 的形式内嵌在系统提示词中，具备以下特点：

- **原子性与行动导向**：每个待办事项都应该是一个独立的、不可再分的“原子”任务
- **有意义的抽象层级**：待办事项不应是琐碎的操作性动作（如“读取文件 a.txt”或“修复 linter 错误”），而应是高层级的、有意义的、非平凡的任务。
- **适当的范围**：规范倾向于“更少、更大的待办事项”，而不是一个冗长的微小步骤列表。
- **以实现为中心**：如果用户的请求是要求实现某个功能，那么 Agent 生成的待办事项列表本身就是最终的计划。

通过这种结构化规划，Agent 能够把“用户需求”转化为“系统计划”，为多 Agent 协作奠定语义接口。

### 1.4.3 多 Agent 协作体系：从个体到组织

[![img](https://github.com/phodal/build-coding-agent-context-engineering/raw/master/images/multiple-agent-architectures.png)](https://github.com/phodal/build-coding-agent-context-engineering/blob/master/images/multiple-agent-architectures.png)

单体 Agent 的能力是有限的，而多 Agent 系统（Multi-Agent System, MAS）则是适合智能体系统发展的工程化方向。 正如微服务体系通过拆解单体应用来实现高内聚、低耦合，多 Agent 系统通过拆分智能体职责，实现了智能的横向扩展。 通过让多个 Agent 能够通过协作实现更复杂的目标，也能类似于“团队”在软件开发中的协同工作。

常见协作拓扑（参考 [LangGraph](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)、AutoGen 等）：

- 主管-专家模式（层级结构）：一个“主管 Agent”（Supervisor）或“协调员 Agent”（Coordinator）负责接收高层级的用户目标，将其分解为一系列子任务，然后根据每个子任务的性质，将其分派给相应的“专家 Agent” 。
- 并行模式（群体智能）：也称为“并发模式”或“蜂群模式”。多个 Agent 同时独立地执行相同的任务或任务的不同部分，然后将它们的输出进行综合 。
- 顺序模式（流水线）：Agent 们像在一条流水线上一样，按照预定义的顺序依次工作。前一个 Agent 的输出成为后一个 Agent 的输入 。
- 网络模式（对话式/动态模式）：Agent 可以在一个多对多的网络中自由交流，没有固定的层级结构。下一个行动的 Agent 通常是根据对话的流程动态决定的 。

多 Agent 拓扑结构的选择，直接反映了待解决问题的底层结构。架构并非任意选择，而是试图创建一个能够镜像问题依赖图的“认知模型”。 当然，也不可避免会遇到类似于微服务架构中的复杂度等各种问题。

#### 1.4.3.1 A2A 协议：构建 Agent 网络，加速智能能力共享

2A 专为 Agent-to-Agent 通信而设计，与处理 Agent-to-Tool 通信的模型上下文协议（MCP）等其他标准形成互补。它扮演着公共互联网协议的角色，允许不同的 Agent 系统相互连接和互操作。

尽管，我们并不一定需要引入 A2A 架构，如我们在 AutoDev 中实现的机制是，将 A2A 协议的 Agent 以 MCP 工具的形式暴露给 Agent 使用， 在不添加系统复杂度的情况下，实现了 Agent 与 Agent 之间的协作。

#### 1.4.3.2 自我完善：反思、记忆与评价闭环

一个演化中的 Agent 的真正力量，来自于反思循环与持久记忆系统的紧密整合。

- 反思机制：Agent 回顾自己的输出，识别错误并生成改进建议；
- 记忆存储：持久化任务经验与上下文（如 `AGENTS.md`、Knowledge Graph），为后续任务提供长期参考。

对于记忆而言，应该根据新近度、相关性和重要性对记忆进行加权检索的机制，以及能够自主决定记住什么、忘记什么以及如何组织信息的反思性记忆管理系统 。

> 一个先进 Agent 架构的最终目标，是创建一个自我强化的飞轮：行动产生经验，反思将经验提炼为知识，记忆存储知识以改进未来的行动。这将 Agent 从一个静态的程序，转变为一个动态的学习实体。

## 1.5 小结

> 系统提示词（System Prompt）在 Agent 系统中的地位，远超一份简单的指令集；它实际上是 Agent 的核心“操作系统”，需要以系统架构设计的高度来对待提示词和上下文工程。

利用 Markdown 或 XML 等标记语言来构建结构化的指令模块，可以显著提高 LLM 对复杂规则的理解和遵循能力。 通过明确的角色激活、详尽的行为规范、以及“即时”加载数据等上下文工程技术，开发者可以为 Agent 塑造一个稳定、可预测的“认知环境”， 从而将其行为引导至期望的轨道上。优秀的上下文工程是实现 Agent 行为可靠性的基础。

相关资源：

- https://docs.spring.io/spring-ai/reference/api/structured-output-converter.html
- Agentic Design Patterns： https://docs.google.com/document/d/1rsaK53T3Lg5KoGwvf8ukOUvbELRtH-V0LnOIFDxBryE/edit?tab=t.0
- Agentic Context Engineering: https://www.arxiv.org/pdf/2510.04618
- A Survey on Large Language Model based Autonomous Agents: https://arxiv.org/pdf/2308.11432
- [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [AGENTIC RETRIEVAL-AUGMENTED GENERATION: A SURVEY ON AGENTIC RAG](https://arxiv.org/pdf/2501.09136)
- [How to build reliable AI workflows with agentic primitives and context engineering](https://github.blog/ai-and-ml/github-copilot/how-to-build-reliable-ai-workflows-with-agentic-primitives-and-context-engineering/)
