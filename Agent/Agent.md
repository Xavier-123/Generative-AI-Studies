# 一、基础

## LLM 底层基础

### Transformer 原理

```

```



### 注意力变体（MQA/GQA/MLA）

```

```



### 上下文窗口原理

```

```



------



# 二、微调

## LLM 微调

微调体系：SFT、RLHF、DPO、ORPO、RM 训练逻辑，全参 / LoRA/QLoRA 微调适用场景；

### SFT

### LoRA

### QLoRA

### DPO

### GRPO

### RLHF

### Agent 微调

- **Agent 专有微调：** 了解如何通过微调（SFT）提升模型在特定 Agent 任务上的表现（例如提升 Function Calling 的准确率、提高遵循复杂格式的能力）。
- **基于强化学习的 Agent 训练（RL for Agent）**

### 框架：

- [LLaMA Factory](https://llamafactory.readthedocs.io/?utm_source=chatgpt.com)
- [TRL](https://huggingface.co/docs/trl/index?utm_source=chatgpt.com)



## Embedding / Reranker 微调





## 常见问题

------



# 三、Agent

Q1：[模型上下文限制为什么催生 Agent？](https://chatgpt.com/share/6a2121b4-751c-83ea-8c38-4897d262ee1f)

------



## 3.1 核心理论

### **3.1.1 经典架构拆解**

#### CoT/GoT

CoT让模型：

> 不直接回答
>
> 而是一步一步思考

GoT让模型：

> 推理像图
>
> 多个思路同时探索



#### Self-Ask

**Self-Ask**（自问）是一种提示工程（Prompt Engineering）技术，旨在让大语言模型通过**显式地提出并回答自己的中间问题**，来逐步推导出最终答案。

它的核心思想是模仿人类的思考过程：当面对复杂问题时，人们会先拆解出子问题，自言自语地逐个解答，然后综合得出最终结论。



#### Plan&Solve

核心逻辑就是：

1. **Plan**：让大模型先制定解决问题的步骤
2. **Solve**：让大模型按照步骤执行，最终给出答案



#### Reflexion

核心逻辑：**大模型先生成初步答案 → 自我反思检查错误 → 基于反思修正答案**（一轮 Reflexion 迭代）



#### ReAct

推理 + 工具调用双循环逻辑，Few-shot 构造；

如果面试官问「ReAct 本质是什么」：

> ReAct 是一种 Agent 推理范式，将 LLM 的推理（Reasoning）和工具调用（Acting）交替进行。核心循环为：
>
> Question → Thought → Action → Observation → Thought → Action → Observation → Final Answer
>
> 本质上就是一个 while 循环：
>
> 1. LLM 决定是否调用工具
> 2. Agent 执行工具
> 3. 工具结果作为 Observation 回填上下文
> 4. LLM 基于新的上下文继续推理
> 5. 直到产生 Final Answer



| 技术         | 核心目标     | 是否规划 | 是否工具调用 | 是否反思 | 复杂度 |
| ------------ | ------------ | -------- | ------------ | -------- | ------ |
| CoT          | 提升推理能力 | ❌        | ❌            | ❌        | ★      |
| GoT          | 提升复杂推理 | 部分     | ❌            | ❌        | ★★     |
| Self-Ask     | 拆解问题     | 部分     | 可选         | ❌        | ★★     |
| Plan & Solve | 先规划后执行 | ✅        | 可选         | ❌        | ★★★    |
| ReAct        | 推理+行动    | 动态     | ✅            | ❌        | ★★★★   |
| Reflexion    | 自我纠错     | 动态     | ✅            | ✅        | ★★★★★  |



### **3.1.2 三大核心模块**

- **工具调用（Function Calling）**：函数描述构造、参数校验、错误重试、多工具串联；
- **记忆系统 Memory**：短时上下文记忆、向量库长期记忆（RAG+Agent 结合）、分层记忆；
- **规划 Planning**：任务拆解（子任务拆分）、动态纠错、多轮迭代重规划。

### **3.1.3 Agent 常见痛点**

#### 	一、 幻觉（Hallucination）

**表现**：Agent 凭空捏造事实、虚构不存在的工具参数，或在回答中给出错误的信息。

- **优化方案**：
  1. **RAG（检索增强生成）深度融合**：
     - 在 Agent 决策前或生成阶段，接入知识库（向量数据库、图数据库或企业内部搜索接口），将确定性的事实作为 Context（上下文）喂给大模型，强制大模型“基于参考资料回答，若无参考则承认不知道”。
  2. **自我审视与纠错机制（Self-Reflection）**：
     - 引入自我反思工作流（如 Reflexion 框架）。在 Agent 产出最终答案或执行动作前，设计一个内部提示词（或引入一个更轻量/更专注的 Critic Agent）对输出进行“事实性校验”（Fact-checking），若发现不合逻辑或无事实依据，则进行内部重打回。
  3. **约束性解码（Constrained Decoding）**：
     - 利用特定框架（如 Outlines、Instructor 或 JSON Mode）强制 Agent 的输出必须符合特定的 Schema（如严格的 JSON 格式）。这能极大降低大模型在输出格式和参数命名上的“幻觉”。



#### 二、 工具错选（Incorrect Tool Selection）

**表现**：在面临多个工具时，Agent 选择了错误的工具，或无法正确理解工具的参数要求。

- **优化方案**：
  1. **分层路由与工具动态注入（Hierarchical Routing）**：
     - 避免一次性把几十个工具一股脑推给 Agent（这会干扰大模型的注意力）。采用**两阶段路由**：先由一个高层 Agent（Router）决定当前任务属于哪个大类，再将该大类下的少数几个具体工具注入给下层 Agent，减少干扰项。
  2. **优化工具的语义描述（Semantic Description）**：
     - 工具的 name（名称）和 description（描述）是大模型选择工具的唯一依据。应像写高质量 Prompt 一样设计工具描述：清晰说明工具的“适用场景”、“输入参数格式”及“输出结果代表什么”。
  3. **少样本学习（Few-Shot Examples）**：
     - 在 System Prompt 中加入具体的 Few-shot 示例，展示在面临模糊或相似请求时，如何正确甄别并选择相应的工具。
  4. **Schema 强校验与反馈环**：
     - 当 Agent 传错工具参数时，代码层（如 Pydantic 校验）应捕获错误，并将具体的报错信息（如“参数 age 应该为整型，但你输入了字符串 'twenty'”）作为反向 Observation 反馈给大模型，引导其自动修正并重新调用。



#### 三、 循环调用（Infinite Loop / Repetitive Calling）

**表现**：Agent 陷入死循环，不断用相同的参数调用同一个工具，或者在某两三个步骤之间反复打转，无法推进任务。

- **优化方案**：
  1. **硬性截断（Max Iterations & Timeouts）**：
     - 在 Agent 执行引擎（Executor）中设置硬性上限，如最大迭代步骤（max_iterations=5）或最大执行时间限制。达到上限后强行中断，并向用户报错或转入人工客服。
  2. **动作历史去重与相似度检测（Action History De-duplication）**：
     - 维护一个执行历史队列（History Trace），记录每次的“Tool-Input-Output”。在每次决策前，通过比对新生成的 Action 与历史 Action 的相似度（或完全匹配），一旦检测到连续 2 次及以上完全一致的调用，即判定为陷入死循环，触发“破局提示”（Breaker Prompt）或中断。
  3. **引入显式状态机约束（State Machine / Graph-based）**：
     - 放弃纯粹由大模型自主控制的 ReAct 自由流，改用基于状态图的框架（如 LangGraph）。将复杂的任务拆解为节点（Nodes）和有向边（Edges），限制节点之间的跳转逻辑，从架构层面阻断非法循环。



#### 四、 任务越界（Task Boundary Violation）

**表现**：Agent 受到用户提示词的诱导（Prompt Injection），开始执行与自身定位不符的任务，或越权访问敏感资源。

- **优化方案**：
  1. **安全护栏机制（Guardrails）**：
     - 在输入和输出端部署防护层（如 Guardrails AI、LlamaGuard 或自定义敏感词/分类模型）。
     - **输入护栏**：检测用户请求是否包含恶意注入、越权尝试或偏离主题的提问，提前拦截。
     - **输出护栏**：在 Agent 生成内容后、交付给用户前，检测是否包含敏感数据泄露或违规言论。
  2. **最小特权原则与环境隔离（Least Privilege）**：
     - 不要给予 Agent 对应的 API Key 或数据库连接过高的权限。在基础设施层实现 RBAC（基于角色的权限控制），Agent 只能调用其职责范围内的受控 API。
  3. **关键步骤引入人机协同（Human-in-the-Loop, HITL）**：
     - 对于高风险、破坏性、涉及资金或敏感信息修改的操作（如发送邮件、转账、删除数据），设计硬性的人工审批节点，只有人类确认后，Agent 才能继续执行。



#### 五、 多步骤路径跑偏（Multi-step Path Deviation）

**表现**：在长链条（Long-horizon）任务中，Agent 随着执行步骤的增加，逐渐忘记了最初的终极目标，或者由于某一步工具返回了干扰信息，导致后续的执行路径完全漂移。

- **优化方案**：
  1. **“计划与执行”解耦（Plan-and-Solve）**：
     - 不要让 Agent 边想边做（传统的 ReAct 容易迷失）。采用“先规划，后执行”的模式。
     - **Planner**：先生成一个完整的多步骤执行计划（Step 1, Step 2, Step 3...）。
     - **Executor**：逐一执行计划中的子任务，并在每个子任务完成后更新状态。
  2. **动态重规划与目标锚定（Dynamic Re-planning）**：
     - 在每次工具执行完毕后，要求 Agent 评估：1. 原始目标是什么？ 2. 当前进度到哪里了？ 3. 刚才的工具结果对既定目标是否有帮助？ 4. 是否需要修正后续的计划？ 将全局目标（Original Goal）始终固化在 Context 的醒目位置（目标锚定）。
  3. **记忆管理与上下文压缩（Memory Summary）**：
     - 长链条任务会导致 Context Window 极速膨胀，冗余的中间工具输出会稀释大模型对核心目标的注意力。
     - 不建议直接把所有的历史对话和工具输出塞给模型。应当采用**滑动窗口记忆**或**摘要记忆（Summary Memory）**，只保留“已达成的阶段性成果”和“接下来的行动点”，过滤掉无用的调试日志和工具冗余返回。



------



## 3.2 Agent框架

### LangGraph

很多企业已经从LangChain迁移到LangGraph。

需要掌握：

- StateGraph
- Node
- Edge
- Conditional Edge
- Memory
- Checkpoint
- Human-in-the-loop

最好能手写：

- ReAct Agent
- Plan-and-Execute Agent
- Multi-Agent



### OpenClaw



#### 常见问题

Q1：AutoGPT、BabyAGI、OpenAI Agent 框架设计思路

```
AutoGPT解决的是“让模型自主行动”，BabyAGI解决的是“让模型管理任务”，而OpenAI Agent解决的是“让模型可靠地调用工具并完成业务流程”。
```





------



## 3.3 Tool Calling

**Function Calling** = 早期官方叫法

**Tool Calling** = 现在的通用、升级版叫法

Agent本质：LLM + Tool + Memory + Planning

很多人在学习 Agent 时，会把 **Tool Calling** 和 **Function Calling** 当成同一个概念，但实际上它们是 **包含关系**：

> Function Calling 是一种实现机制
>  Tool Calling 是一种能力范式

```
Agent
 └── Tool Calling（调用工具）
      ├── Function Calling（调用本地函数）
      ├── API Calling（调用远程接口）
      ├── Database Query（查询数据库）
      ├── Code Interpreter（执行代码）
      ├── Search Engine（搜索引擎）
      └── MCP Tool（MCP工具）
```



### MCP

“MCP（Model Context Protocol）是由 Anthropic 最近推出的一个**开放标准协议**，它的核心目的是**解决 AI 模型与外部数据源、工具之间的连接壁垒问题**。

MCP协议理解为，使用 mcp.server.fastmcp.FastMCP 对象，将python函数转换为大模型能理解的固定格式描述

可以将 MCP 协议中的 FastMCP 对象理解为一个**‘翻译器’和‘适配器’**：它自动将 Python 函数的函数名、文档字符串（Docstring）、参数类型注解（Type Hints）进行解析，封装成大模型能够通过 JSON 格式高效处理的 **Schema（结构定义）**。

重点：

- MCP Client（MCP 客户端）
- MCP Server（MCP 服务端）
- Tool Discovery（工具发现机制）
- Tool Invocation（工具调用流程）

**很多公司已经开始要求：**

> 会开发MCP Server

### Skill

主流出现在 **Agent、Function Calling / Tool Calling** 体系里：

- 本质：**大模型可调用的「可执行能力单元」**
- 形态：函数、接口、脚本、插件，带固定入参、出参、描述
- 核心流程：LLM 理解用户意图 → 选择 Skill → 拼装参数 → 调用执行 → 拿回结果继续对话
- 特点：**绑定单个 Agent / 应用**，能力封闭，多用于单智能体内部工具调用

| 维度         | Skill / Tool(Function Calling)           | MCP (Model Context Protocol)                                 |
| ------------ | ---------------------------------------- | ------------------------------------------------------------ |
| **定位**     | 功能单元、能力集合                       | 通信协议、交互标准                                           |
| **层级**     | 应用层「功能」                           | 传输层 / 会话层「协议规范」                                  |
| **耦合度**   | 强耦合：Skill 写死在 Agent 代码 / 配置里 | 完全解耦：客户端和服务端分离，跨进程 / 跨机器                |
| **作用范围** | 单 Agent 内部使用                        | 通用互通：任意 LLM 客户端 ↔ 任意 MCP 服务                    |
| **能力范围** | 侧重**工具函数调用**                     | 不止工具，还包含**上下文读写、资源访问、会话、文件、日志**等 |
| **生态目标** | 给单个智能体装插件                       | 打造统一的 AI 资源互联生态                                   |



### Function Calling / Tool Calling 



> 让 LLM 不直接回答问题，而是生成一个函数调用请求。

#### 需要熟悉

- Function Calling（目的是调用工具）
- JSON Schema（告诉大模型函数参数长什么样）
- Structured Output（目的是结构化结果）
- Tool Routing（从多个工具中选择最合适的工具）



### 高频问题

#### Q1：Tool Description设计

```tex
（1）说清楚什么时候用
（2）说清楚什么时候不用
（3）参数解释清楚
（4）提供样例

Tool Description本质是在给模型做路由指导。我会重点描述工具适用场景、不适用场景、参数含义以及调用示例，因为Tool Calling错误很多时候不是模型能力问题，而是Description歧义导致的。
```



#### Q2：Few-shot

```tex
如何利用示例提升模型表现？

生产环境一般：3~10个样例
重点覆盖：高频场景；易混淆场景；边界场景

Few-shot本质是给模型建立输入输出映射关系。在分类、工具选择和结构化输出任务中效果非常明显。我通常会优先覆盖容易混淆的Case，而不是一味增加样本数量。
```



#### Q3：Tool选择机制

```
Agent如何决定调用哪个Tool？
方案1：纯LLM Routing（简单，但不稳定）
方案2：Rule + LLM（Query → 规则过滤 → 候选Tool → LLM选择）
方案3：Embedding Routing（工具向量化，Query Embedding → Tool Embedding → Similarity Search → TopK Tool → LLM选择）

Tool选择一般不完全依赖LLM。我更倾向于Rule Filter + Embedding Recall + LLM Decision的三层路由架构，先缩小候选范围，再让模型决策，可以显著降低误调用率和Token成本。
```



#### Q4：如何提高Tool Calling准确率？

```python
方法1：优化Tool Description
方法2：增加Few-shot
方法3：减少候选Tool数量
方法4：参数校验
方法5：Tool Retry
方法6：Self-Reflection
方法7：分层Agent

# 回答
提高Tool Calling准确率我通常从五个方面优化：
第一，设计清晰的Tool Description，明确适用和不适用场景；
第二，通过Few-shot给模型提供正确调用示例；
第三，采用Rule Filter或Embedding Recall缩小候选Tool范围，而不是让模型直接从大量工具中选择；
第四，使用JSON Schema或Pydantic进行参数约束和校验；
第五，引入Retry和Reflection机制，让模型能够根据错误信息重新选择或修正参数。
在实际Agent项目中，Tool Calling问题大约70%来自工具描述和路由设计，而不是模型本身，因此我会优先优化Tool设计和路由层。
```



#### Q5：工具调用数据集构造

```

```



#### Q6：如何优化模型原生 Function Calling 能力

```python
通常希望考察你是否理解 Prompt → Schema → Tool Selection → Parameter Extraction → Tool Execution → Result Injection 整条链路。

1. Tool Description 优化（收益最大）
2. Few-shot Function Calling
3. JSON Schema 优化
4. Tool Routing（工具路由）
5. ReAct / Multi-step Function Calling（对复杂任务采用 ReAct 或 Multi-Step Agent，使模型具备多轮工具调用能力，而不是单轮 Function Calling。）
6. Function Calling 数据微调（最高阶）

# 回答
我通常从六个层面优化：
Tool Description 优化，明确工具职责、边界和调用示例；
Few-shot Tool Calling，通过示例教会模型何时调用工具以及如何填充参数；
JSON Schema 优化，增加字段描述、枚举约束和参数校验规则；
Tool Routing，先召回 Top-K 候选工具，减少工具干扰；
Multi-step Agent，利用 ReAct 支持复杂任务的多轮工具调用；
基于 Tool Call Logs 的 SFT 微调，从根本提升工具选择和参数抽取能力。

在实际项目中，收益通常遵循：
Tool Description > Tool Routing > Few-shot > JSON Schema > SFT
前四项往往无需训练即可显著提升 Function Calling 准确率，是生产环境中最常见的优化手段。
```



------



## 3.4 RAG



### 文档解析

- **痛点：** 如何解析复杂的 PDF、Word、PPT 格式？特别是双栏排版、表格（Table）、图片、OCR 识别。
- **核心技术：** 布局分析（Layout Analysis，如 LayoutLM、PaddleOCR）、专门解析表格的工具（如 Table Transformer、Unstructured、PyMuPDF）。



### Chunk策略

Fixed Chunk (固定大小分块)

```python
工作原理：
	固定大小分块是最简单、最基础的分块方法。它直接按照固定的字符数（Character-based）或 Token 数（Token-based）将文本切分成大小相同的块。为了避免语义在边界处被硬生生切断，通常会设置一个重叠区（Overlap）。
例如：Chunk 大小设为 500 字，重叠区设为 50 字。这意味着第二个 Chunk 会包含第一个 Chunk 的最后 50 个字。
优点：
    实现非常简单，计算开销极低。
    容易预测和控制 Token 的消耗。
缺点：
	容易破坏语义完整性。即使有重叠区，一句话或一个完整的段落也可能被切成两半，导致检索时丢失上下文。
适用场景：
    格式非常规整、没有复杂段落逻辑的文本。
    原型开发阶段，或对检索精度要求不高的快速基线测试。
```



Semantic Chunk (语义分块)

```python
工作原理：
	语义分块不再依赖硬性的字数限制，而是根据文本内容的意思来进行切分。通常的做法是：
	1.先将文本切成句子。
	2.计算相邻句子之间的向量相似度（Embedding Similarity）。
	3.如果相邻两句的相似度低于设定阈值，说明话题发生了转换，就在这里进行切分。
此外，简单一些的语义分块也可以基于 Markdown 标题、段落分隔符（\n\n）或标点符号（如句号）来进行。

优点：
    能最大程度保证每个 Chunk 内部语义的完整性和独立性。
    检索出来的文本块通常是一个完整的观点或主题，有利于 LLM 理解。
缺点：
    计算成本较高。如果使用向量相似度切分，需要对每个句子生成 Embedding 并计算相似度。
    Chunk 的大小不固定，可能会出现极长或极短的 Chunk，较难控制输入给 LLM 的 Token 数量。
适用场景：
    叙事性较强、段落主题明确的文档（如小说、法律条文解释、分析报告）。
```



Parent Child Chunk (父子分块 / 检索与生成分离)

```
工作原理：
父子分块的核心思想是**“小块检索，大块输入”**。它将文档切分为两种尺寸的块：
	子块（Child Chunk）： 较小的文本块（例如 100-200 Token），用于生成向量并进行索引。
	父块（Parent Chunk）： 较大的文本块（例如 1000-2000 Token，或者子块所在的整个段落/章节）。
	在检索阶段，系统在向量数据库中匹配最相似的子块。但是，在将检索结果喂给 LLM 进行生成时，系统会根据子块的 ID 自动替换并返回其对应的父块。

优点：
	解决了检索和生成的矛盾。小块在向量化时语义更聚焦，检索更精准；大块在生成时能提供更丰富的上下文，防止 LLM 产生幻觉或断章取义。
缺点：
	数据结构较为复杂，需要在数据库中维护父子关系的映射。
	存储开销略微增加。
适用场景：
	专业技术文档、API 文档、学术论文。这些文档中往往包含细节数据（适合子块定位），但回答问题需要结合上下文（适合父块生成）。
```



Hierarchical Chunk (层级分块)

```
工作原理：
	层级分块可以看作是父子分块的延伸或更复杂的变体。它将文档组织成一个多级的树状结构（例如：文档 -> 章节 -> 段落 -> 句子，或者通过算法将相似的 Chunk 进行聚类并生成高层摘要）。
	一个典型的实现是 RAPTOR 算法：它先将底层叶子节点（细粒度文本）进行聚类，然后用 LLM 为每个聚类生成一个摘要（高层节点），再对这些摘要进行聚类生成更高层的摘要，最终构建出一棵树。
	在检索时，系统会根据问题的范围（大局观问题还是细节问题），决定从树的哪一层进行检索。

优点：
	既能回答微观/细节问题（如“某某产品的尺寸是多少？”——定位到叶子节点），也能回答宏观/全局问题（如“这本书的核心观点是什么？”——定位到高层摘要节点）。
缺点：
	实现和维护成本极高。
	如果使用 LLM 逐层生成摘要，建库阶段（Embedding 和 Prompt 消耗）的成本会非常高昂。
适用场景：
	长篇研究报告、整本书籍分析、复杂的合规性文档等需要兼顾宏观与微观视角的信息提取任务。
```

### 检索增强

查询变换与扩展

```python
# 常见方法:
1. Query Rewrite（查询改写）
2. Query Expansion（查询扩展）
3. Multi Query
```

Sparse Retrieval

```python
# 稀疏检索,传统搜索引擎使用的方法
# 常见方法:
1. TF-IDF
2. BM25

# 优点
	快，可解释
# 缺点
	无法理解语义
```

Dense Retrieval

```python
# 用Embedding做检索
# 常见Embedding模型
OpenAI：text-embedding-3-large
BGE：bge-large-zh、bge-m3
Qwen：Qwen3-Embedding

# 优点
	理解语义
# 缺点
	精确匹配较差
```

Graph Retrievald

```python
# 适用于
	知识图谱
    代码仓库
    Agent Memory
    企业知识库
# 常见项目
	GraphRAG
```

Hybrid Search

```python
# 实际生产最常用
Sparse Retrieval + Dense Retrieval

# 主流实现
Elasticsearch：BM25 + Vector
Pinecone：Hybrid Search
Weaviate：Hybrid Search
Milvus：BM25 + Dense
```

Rerank

```python
# 有时 Rerank 比 Retriever 更重要

# 常见结构
Cross Encoder

# 常见Rerank模型
BGE：bge-reranker-large
Cohere Rerank
Qwen：Qwen3-Reranker
```



### 向量数据库

- [Milvus](https://milvus.io/?utm_source=chatgpt.com)
- [Qdrant](https://qdrant.tech/?utm_source=chatgpt.com)
- [Weaviate](https://weaviate.io/?utm_source=chatgpt.com)



### 高频问题

#### Q1：Cross Encoder vs Bi Encoder 

掌握 Agentic RAG（智能体检索）：Agent 自主决定何时检索、检索什么、如何重构 Query、如何评估检索结果（Self-RAG, Corrective RAG）。

------

## 3.5 Context Engineering

上下文工程（Context Engineering）这是2026年越来越火的方向。

核心思想：如何在有限上下文窗口内，为模型动态构造“最有价值的输入环境（Context）”。



### 3.5.1 Prompt Engineering

**提示词工程**，就是**设计、优化、调试给大语言模型的输入指令（Prompt / 提示词）**，在**不改动模型本身**的前提下，让 AI 精准按照你的要求输出结果的技术与方法。



#### 自动提示词优化

自动提示词优化（Automated Prompt Optimization, APO）

##### 1. 为什么需要 APO？

在日常开发中，传统的提示词工程通常由人类工程师手动完成：

1. **耗时且效率低**：改一个词，需要重新测试成百上千条数据。
2. **容易“拆东墙补西墙”**：你为了解决 Case A 修改了提示词，结果却导致原本正确的 Case B 和 Case C 报错了。
3. **不可迁移**：当你把底层大模型从 GPT-4 换成 Claude 时，原本精雕细琢的提示词可能直接失效，所有工作又得重来。

APO 的出现，就是为了把这种“手工业”变成类似于传统机器学习的“工业化训练”。



##### 2. APO 是如何工作的？

APO 的核心思想是**将机器学习中的“模型训练（梯度下降）”逻辑套用在文本上**。它将提示词视为“待训练的参数”，通过以下循环自动迭：

1. **准备数据集（Dataset）**：包含输入数据（如“请翻译这句话”）和标准答案（Ground Truth）。
2. **运行与评估（Loss Function）**：使用初始提示词在数据集上运行，用算法（或另一个 LLM 裁判）评估输出质量，找出出错的案例（Bad Cases）[[1](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQFer6wRI4G9cKs0qiThNp_8p-VeIMJBw7t_bAK02JhorQOIBEjaHuhTk1ivRYY1gxvMqKC1IKTS9E50BqSCMqHLtWBvA2k4Y15qj_ZdJk5bSQLVyeBIga2mobwPIDNB_LX9tIOIbhLbhBIP9jEsDsVw3HDL3WCJkUoM3P9ropwY-FwjEHAw3-h3jvSm5rtYgqksYEKUEacw)]。
3. **错误分析（Gradient/梯度）**：让一个“评判大模型”分析这些出错的案例，生成自然语言形式的诊断报告（例如：“当前提示词没有明确规定输出格式，导致大模型输出了多余解释”）。这在 APO 中被称为**“文本梯度”（Textual Gradient）**。
4. **生成候选提示词（Optimizer）**：让一个“优化大模型”根据诊断报告，修改并生成几个新的候选提示词。
5. **筛选与迭代（Selection）**：在验证集上测试这些新的候选提示词，保留表现最好的一个（或使用束搜索 Beam Search 等算法保留前几名），然后进入下一轮迭代，直到性能达到瓶颈。



##### 3. 代表性的 APO 技术

在学术界和工业界，目前有几种主流的 APO 实现方案：

- **APO / [PromptWizard](https://github.com/microsoft/PromptWizard) (微软)**：提出使用自然语言“梯度”来批评当前的提示词，并在语义相反的方向上自动重写提示词。
- **OPRO (Google DeepMind)**：将大模型作为优化器（LLMs as Optimizers）。它在 prompt 中展示历史尝试过的提示词及其对应的得分，让大模型根据这些“历史轨迹”自动推导出得分更高的新提示词。
- **DSPy (斯坦福)**：将提示词工程转变为“声明式编程”。你只需要定义输入输出，DSPy 的编译器（Compiler）会自动帮你优化提示词、自动挑选最合适的 Few-shot 示例，完全避开手动写 Prompt 的过程。
- **TextGrad (斯坦福)**：将上述过程进一步泛化。它不仅能优化 Prompt，还能自动优化代码和文本答案，把整个 LLM 系统的优化做得像 PyTorch 的自动求导一样系统化。



##### 4. APO 的优势

- **数据驱动**：提示词的优化结果完全基于真实测试集的反馈，避免了人类的主观偏好。
- **抗回归（Regression-proof）**：在迭代过程中，系统会确保新的提示词在整个数据集上的综合表现变好，而不会因为解决了一个新 Bug 而引入三个旧 Bug。
- **自动化适配大模型**：当企业更换底层模型时，只需把新模型接入 APO 管道，重新跑一遍自动化优化流程，就能在几小时内得到一套最契合新模型的提示词。



##### 高频问题

#### Q1：prompt优化

```tex
（1）角色定义（Role）：明确模型身份
（2）任务拆解（Task Decomposition）：不要让模型一次做很多事。（Chain of Thought、Structured Prompting）
（3）输出约束（Output Constraint）：降低格式错误率，提高自动化处理能力
（4）边界条件说明：比如 不知道时输出：UNKNOWN，不要编造信息
```



------



### 3.5.2 Context Selection

核心问题：

> 哪些信息应该放进上下文？

常见方法：

#### Rule-Based

```
保留最近10轮
```

#### Embedding Retrieval

```
query_embedding → TopK Similar Context
```

#### LLM Selection

```
让LLM决定保留哪些内容
```



------



### 3.5.3 Context Compression

常见方法：

#### Summarization

```
100轮对话 → 摘要
```

#### Semantic Compression

```
保留结论
删除冗余细节
```

#### Structured Compression

主要解决大语言模型（LLM）在处理**长上下文、多轮对话**时的三个痛点：

- **超出上下文窗口**：几轮对话后Token过多导致模型“忘记”开头。结构化压缩后，长对话被精简为几十个Token的关键字段。
- **注意力分散**：原始对话中充斥“嗯…那个…刚才你说的……”等噪声。提取JSON结构能引导模型直接聚焦于`intent`和`constraint`，提升任务成功率。
- **成本与延迟**：压缩后的JSON输入比原始长文本短得多，能显著降低API调用成本和响应延迟。

```python
{
  "intent": "...",
  "constraint": "...",
  "history": "..."
}

# 原始输入（啰嗦，约80 Token）：
用户A：“嗨，你好，我想了解一下，就是……呃……我下周可能要去上海，你知道的，上海天气怎么样？哦对了，顺便问一下，去那边需不需要带伞啊？大概下周三的样子。”
助手：“好的，请问是下周三吗？”
用户A：“对对对，就是下周三。”

# 结构化压缩后（仅约20 Token）：
{
  "intent": "查询天气",
  "constraint": {"地点": "上海", "时间": "下周三", "关注点": ["是否需要带伞"]},
  "history": "用户确认时间为下周三"
}
```

典型框架：

- MemGPT
- LangGraph Memory
- Claude Memory

------



### 3.5.4  Retrieval Engineering

RAG实际上属于Context Engineering的重要组成部分。

常见方法：

#### Query Rewrite

用户：

```
它什么时候发布的？
```

改写：

```
GPT-5什么时候发布？
```

#### Retrieval

```
BM25
Dense Retrieval
Hybrid Search
Graph Retrieval
```

#### Rerank

```
Top50 → Top5
```



#### Context Packing

把检索结果组织进Prompt。

------

### 3.5.5 Memory Engineering

记忆工程，Agent最重要的能力之一。

#### Short-term Memory

- **工作记忆 / 短期记忆**
- **定义**：Agent 当前正在进行的任务链条中所必需的即时信息（例如正在运行的代码片段、当前的子目标）。
- **技术实现**：通常直接驻留在 LLM 当前的上下文窗口中，生命周期较短，任务结束即释放。



#### Conversational Memory

- **对话记忆**
- **定义**：单次或跨会话（Session）中，用户与 Agent 之间的聊天历史记录。
- **技术实现**：SQL 关系表、NoSQL 键值对，通过滑动窗口（Sliding Window）或自动摘要（Summary）在输入时动态压缩和加载。



#### Episodic Memory

- **情景记忆**
- **定义**：记录“过去发生了什么”。它保存了 Agent 的历史行为轨迹、调用过的工具日志（Tool Logs）、执行成败以及从中获得的反馈（即“前车之鉴”）。
- **技术实现**：利用事件溯源（Event Sourcing）设计，通过时间戳、成功/失败标签将每一次的执行轨迹序列化保存，支持相似任务场景下的少样本检索（Few-shot Retrieval）。



#### Semantic Memory

- **语义记忆**
- **定义**：Agent 掌握的静态或相对持久的事实、概念与常识（即 RAG 所依赖的外部知识库）。
- **技术实现**：向量数据库、混合检索系统（Hybrid Search）。



#### Procedural Memory

- **程序记忆 / 工作流与工具记忆**
- **定义**：关于“如何做一件事”的记忆。包括 Agent 需要遵循的具体操作规程、业务流，以及它所拥有的工具箱定义（Toolbox）。
- **技术实现**：动态工具检索系统。当 Agent 拥有成百上千个 API 时，不应把所有工具描述直接丢进 Prompt，而是将工具定义存储在数据库中，由 Agent 动态语义检索并加载当前需要的工具。



#### Entity Memory

- **实体记忆**
- **定义**：关于特定人、项目、系统或对象的结构化事实（例如：用户 A 喜欢用 Python、项目 B 的截止日期是明天）。
- **技术实现**：通常使用图数据库（Knowledge Graph）或带有 Vector 扩展的关系型 SQL 表，用于维护高度结构化的属性关联。





### 3.5.6 Tool Context Engineering

工具上下文工程，Agent大量依赖Tool。

核心问题：

> Tool结果如何注入上下文？

#### Tool Description

#### Tool Output Formatting

#### Tool Routing

#### Function Calling Schema



### 3.5.7 Agent Planning Context

规划上下文，Agent决策依赖上下文。

#### ReAct

#### Plan-and-Solve

#### Reflexion

#### Tree of Thoughts



### 3.5.8 Context Optimization

（上下文优化）

#### Token Budget Allocation

#### Context Ranking

#### Dynamic Context





你最近一直在研究：

- Parlant
- Journey
- Strategy Registry
- Context Compression

这正好契合企业需求。

需要掌握：

### Context优化

- Context Window管理
- Long Context压缩
- Memory Selection
- Context Ranking

### Memory系统

- Short-Term Memory
- Long-Term Memory
- Semantic Memory
- Episodic Memory

面试可能直接问：

> 如何解决200K上下文导致的性能下降？



长期记忆

长任务执行

状态管理

多步规划

动态上下文构建

------



## 3.6 Agent评测

这是很多候选人的短板。

重点：

- **评测方法：** 如何评估一个 Agent 的好坏？了解自动评测（如 Ragas、G-Eval、使用 GPT-4 作为裁判）与人工评测的结合。
- **链路追踪：** 熟练使用或了解 Agent 监控工具（如 LangSmith、Langfuse、Phoenix），能够分析 Agent 运行过程中的瓶颈（如哪一步调用耗时最长、哪一步 Prompt 出现幻觉）。
- **评估**：如何判断Agent任务成功？设计成功率、步骤效率、工具调用准确率等指标。熟悉LangSmith或自建简单的trace log。
- **可观测性**：记录每一步的thought/action/observation，方便调试。面试中展示你做的Agent能看到完整执行轨迹。
- **评估体系**：了解 AgentBench、GAIA 等主流评测集。
- **Trace 分析**：如何追踪 Agent 的思考链路（Thought Trace），定位是推理错了、检索错了还是工具调错了。
- **LLM-as-a-Judge**：使用更强的模型来自动评估 Agent 任务完成度的方法论。

### Offline Eval

- Accuracy
- Tool Success Rate
- Hallucination Rate

### Online Eval

- User Satisfaction
- Task Completion

框架：

- [LangSmith](https://www.langchain.com/langsmith?utm_source=chatgpt.com)
- [DeepEval](https://deepeval.com/?utm_source=chatgpt.com)
- Langfuse
- Phoenix

------



## 3.7 Agentic RL

Search-R1

TinyZero

ReTool

Multi-Turn-RL-Agent

DeepResearcher



## 3.8 Multi-Agent



------

# 四、部署

Python 进阶：异步 asyncio、Pydantic（函数参数结构化，Function Calling 必用）、FastAPI 封装 Agent 服务接口；

向量数据库：Chroma、FAISS、Milvus、Qdrant（熟练建库、插入、检索）；

大模型本地部署：vLLM/SGLang 部署开源模型（Qwen/Llama3/Mistral），对接私有 LLM 做本地 Agent；

容器：Docker 打包 Agent 项目，基础 Docker-Compose。



## 4.1 LLM推理

量化基础：FP8、GPTQ、AWQ，推理优化基础（vLLM/SGLang/PagedAttention），看懂推理链路。

重点准备：

### 推理框架

- [vLLM](https://docs.vllm.ai/?utm_source=chatgpt.com)
- [SGLang](https://docs.sglang.ai/?utm_source=chatgpt.com)

面试高频：

- Prefix Cache
- KV Cache
- Continuous Batching
- Paged Attention
- Speculative Decoding



## 4.2 **端侧 Agent**

------



# 五、其它

## **A2A**



## CodeAct

PlanAct



## Skills



## Harness

### Harness Evolving

#### AutoPE

#### Memory Evolve

#### AlphaEvolve

#### Hyper-Agents





------



# 六、项目

## 项目1：上下文优化Agent

这个和你目前研究方向最匹配：

```
Raw Context
      ↓
Journey Strategy
      ↓
Context Compression
      ↓
Tool Selection
      ↓
Response Generation
```

把你之前讨论的：

- Strategy Registry
- Dual Decision Engine
- Journey
- Context Optimization

做成开源项目。



### 🎯 面试问题预测

**Q1：为什么选择"无训练"方案而不是直接微调模型？**

> **回答要点**：微调成本高（算力、数据标注）、周期长、难以快速迭代；无训练方案通过推理阶段积累经验，零算力成本、实时可更新，适合资源受限的业务场景。可对比 RAG（检索外部知识）与本方案（自生成内部经验）的差异。

**Q2：为什么选择 PromptWizard 和 TextGrad 两套框架？它们的核心差异是什么？**

> **回答要点**：PromptWizard 是微软提出的示例驱动搜索方案，适合有标注数据的分类/问答场景；TextGrad 将"文本梯度"概念引入优化，适合需要多跳推理的复杂任务。两者互补，统一 Config 层降低切换成本。





## 项目2：无训练 LLM 推理增强与提示词自动优化系统

**【Situation · 情境】** 随着大语言模型（LLM）在垂直业务场景（如意图识别、数学推理）中的落地应用，业界面临两大痛点：一是模型无法通过任务数据持续积累领域知识；二是手工设计 Prompt 成本高、质量不稳定、难以系统化优化。现有方案要么依赖昂贵的微调，要么缺乏可解释的优化路径。

**【Task · 任务】** 负责独立设计并实现一套 **无需参数训练的 LLM 推理增强框架**，核心目标是通过在推理阶段动态积累、更新"先验经验"来提升模型在特定任务上的准确率，同时集成多种自动提示词优化策略，支持分类、抽取等多类型任务。

**【Action · 行动】**

1. **创新设计 TrainingFreeGRPO 算法**：借鉴强化学习中 GRPO 的奖励分组思想，通过多路并行采样（ThreadPoolExecutor 多线程）让 LLM 对同一问题生成 N 条推理轨迹，再由模型自身对比正误轨迹提炼可泛化的"经验条目"，以 JSON 格式持久化存储，形成动态经验库。
2. **构建经验生命周期管理**：设计 add / modify / merge 三种经验操作指令，实现跨 Batch 的经验增量更新与批量合并去重（压缩阶段），有效防止经验库膨胀与知识冲突。
3. **集成双轨提示词优化方案**：整合 PromptWizard（基于示例驱动的搜索优化）与 TextGrad（基于文本梯度的反向传播优化）两套第三方框架，通过统一配置层（OmniContextConfig）灵活切换，支持多种业务场景。
4. **工程化设计**：采用异步主入口（asyncio）、环境变量统一配置（dotenv）、结构化日志（logger）和进度可视化（tqdm），保证系统可观测性与可维护性。

**【Result · 结果】**

- 在目标任务上，模型引入先验经验后推理准确率提升 **X%**，相比基线 zero-shot 推理有显著改善
- 自动提示词优化流程将 Prompt 迭代周期从人工 **Y 天**缩短至自动化 **Z 小时**
- 经验库经压缩机制维持在合理规模，单次推理延迟增加不超过 **A ms**
- 系统支持多任务类型扩展，后续可复用至 RAG、Agent 等更复杂场景

## 二、🎯 面试问题预测（10 题）

### 🔧 核心技术选型

**Q1：为什么选择"无训练"方案而不是直接微调模型？**

> **回答要点**：微调成本高（算力、数据标注）、周期长、难以快速迭代；无训练方案通过推理阶段积累经验，零算力成本、实时可更新，适合资源受限的业务场景。可对比 RAG（检索外部知识）与本方案（自生成内部经验）的差异。

**Q2：为什么选择 PromptWizard 和 TextGrad 两套框架？它们的核心差异是什么？**

> **回答要点**：PromptWizard 是微软提出的示例驱动搜索方案，适合有标注数据的分类/问答场景；TextGrad 将"文本梯度"概念引入优化，适合需要多跳推理的复杂任务。两者互补，统一 Config 层降低切换成本。

------

### 🔥 核心挑战与解决方案

**Q3：经验库如何避免"噪声经验"污染后续推理？**

> **回答要点**：通过两阶段机制：① 单 Batch 内先生成 candidate_experiences 而非直接更新，避免单条坏样本直接写入；② Epoch 结束后的压缩阶段通过 LLM 二次审查合并冗余、淘汰低质量条目。可进一步讨论设置置信度阈值等改进方向。

**Q4：多线程并发调用 LLM API 时，如何处理限速（Rate Limit）问题？**

> **回答要点**：通过 `max_workers` 参数控制并发数，根据 API 配额动态调整；可引申讨论指数退避重试策略、Token 用量监控、批量请求 vs 并发请求的权衡。

**Q5：经验的"增删改合并"操作由 LLM 自身决策，如何保证 JSON 格式输出的鲁棒性？**

> **回答要点**：代码中使用 `split('```json')[-1].split('```')[0]` 做格式抽取，外层 try/except 兜底，失败时跳过而非崩溃（continue）。可进一步讨论引入 Pydantic 校验、Few-shot 格式示例强化输出稳定性。

------

### 💡 技术难点深挖

**Q6：GRPO 算法的核心思想是什么？你的 TrainingFreeGRPO 与原版 GRPO 有何本质区别？**

> **回答要点**：原版 GRPO 通过分组采样计算相对奖励并反向传播更新模型权重；本方案"借用"分组采样的思路，但奖励信号转化为文本经验而非梯度，用 Prompt 上下文替代参数更新，实现 inference-time learning。

**Q7：如何评估先验经验的质量与有效性？**

> **回答要点**：当前方案以最终答案正确率作为隐式反馈；可扩展讨论：引入显式的经验有效性评分（如某条经验在多少问题上有帮助）、A/B 测试（加/不加经验的准确率对比）。

**Q8：`temperature=1.0` 的设计意图是什么？**

> **回答要点**：高温度增加输出多样性，保证 N 条采样轨迹覆盖不同推理路径（正确与错误），为经验提取提供更丰富的对比素材。若 temperature 过低，N 条输出高度相似，经验提取价值有限。

------

### 📊 业务视角

**Q9：这套系统如何从实验原型落地到生产环境？你会做哪些工程化改造？**

> **回答要点**：① 经验库改为向量数据库（如 Chroma，代码中已有依赖）实现语义检索而非全量注入；② 增加经验版本管理与回滚机制；③ 异步任务队列处理大批量数据；④ 监控 API 调用成本与经验库增长曲线。

**Q10：如果业务方反馈模型在新数据上效果下降，你会如何排查？**

> **回答要点**：① 检查经验库是否存在过拟合训练分布的噪声经验；② 对比加/不加经验的推理结果，定位经验质量问题；③ 检查数据分布漂移（新数据与训练数据的 prompt 模式差异）；④ 必要时清空经验库重新从新数据中提取。

------

> 💡 **填充建议**：在"结果"部分的量化占位符（X%、Y 天、Z 小时等），建议根据你实际测试的数据集（如 GSM8K、AIME-24 等代码注释中可见的数据集）补充真实实验数值，会大幅提升简历说服力。