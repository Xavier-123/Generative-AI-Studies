# 一、基础

## LLM 底层基础

### Transformer 原理

Transformer 用 Self-Attention 替代了 RNN 的递归结构，使每个 Token 都能够直接关注序列中的所有 Token，从而解决长距离依赖问题并实现并行训练。其核心由 Multi-Head Attention、Position Encoding、Feed Forward Network、Residual Connection 和 LayerNorm 构成。Encoder 负责理解输入，Decoder 负责自回归生成文本，而现代大语言模型（GPT、Llama、Qwen等）大多采用 Decoder-Only 架构，通过预测下一个 Token 完成预训练和生成任务。

Transformer的核心思想：

> 不要一步一步传递信息。
>
> 让每个Token直接看到所有Token。

Self-Attention到底是什么



Attention怎么计算



Self-Attention直观理解



Multi-Head Attention



为什么需要Position Encoding



Transformer整体结构



Encoder做什么



Decoder做什么



Feed Forward是什么



为什么LLM只保留Decoder



为什么除以√dk

```

```



### 注意力变体（⭐）

MQA / GQA / MLA

```

```



### 上下文窗口原理

```

```



------



# 二、微调

## LLM 微调（⭐⭐）

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

### 框架

- [LLaMA Factory](https://llamafactory.readthedocs.io/?utm_source=chatgpt.com)
- [TRL](https://huggingface.co/docs/trl/index?utm_source=chatgpt.com)
- swift
- unsloth
- verl



## Embedding / Reranker 微调（⭐）





## 常见问题

------



# 三、Agent

Q1：[模型上下文限制为什么催生 Agent？](https://chatgpt.com/share/6a2121b4-751c-83ea-8c38-4897d262ee1f)

------



## 3.1 核心理论（⭐⭐）

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



## 3.4 RAG（⭐）



### 3.4.1 文档解析

- **痛点：** 如何解析复杂的 PDF、Word、PPT 格式？特别是双栏排版、表格（Table）、图片、OCR 识别。
- **核心技术：** 布局分析（Layout Analysis，如 LayoutLM、PaddleOCR）、专门解析表格的工具（如 Table Transformer、Unstructured、PyMuPDF）。



### 3.4.2 Chunk策略

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

### 3.4.3 检索增强

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



### 3.4.4 向量数据库

- [Milvus](https://milvus.io/?utm_source=chatgpt.com)
- [Qdrant](https://qdrant.tech/?utm_source=chatgpt.com)
- [Weaviate](https://weaviate.io/?utm_source=chatgpt.com)

### 高频问题

#### Q1：Cross Encoder vs Bi Encoder 

掌握 Agentic RAG（智能体检索）：Agent 自主决定何时检索、检索什么、如何重构 Query、如何评估检索结果（Self-RAG, Corrective RAG）。

------

## 3.5 Context Engineering（⭐⭐）

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



## 3.6 Harness Engineering（⭐⭐）



------



## 3.7 能力评测（⭐⭐）

这是很多候选人的短板。

重点：

- **评测方法：** 如何评估一个 Agent 的好坏？了解自动评测（如 Ragas、G-Eval、使用 GPT-4 作为裁判）与人工评测的结合。
- **链路追踪：** 熟练使用或了解 Agent 监控工具（如 LangSmith、Langfuse、Phoenix），能够分析 Agent 运行过程中的瓶颈（如哪一步调用耗时最长、哪一步 Prompt 出现幻觉）。
- **评估**：如何判断Agent任务成功？设计成功率、步骤效率、工具调用准确率等指标。熟悉LangSmith或自建简单的trace log。
- **可观测性**：记录每一步的thought/action/observation，方便调试。面试中展示你做的Agent能看到完整执行轨迹。
- **评估体系**：了解 AgentBench、GAIA 等主流评测集。
- **Trace 分析**：如何追踪 Agent 的思考链路（Thought Trace），定位是推理错了、检索错了还是工具调错了。
- **LLM-as-a-Judge**：使用更强的模型来自动评估 Agent 任务完成度的方法论。





### 3.7.1  RAG能力评测

RAG系统的评估通常分为两个核心部分：**检索阶段（Retrieval）\**和\**生成阶段（Generation）**。

#### 1. 评估方法

行业内目前最广为接受的评估方法是**RAG三元组（RAG Triad）框架**（由TruEra/TruLens率先提出）[[3](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQFQb_W0s_DBLGHWL4jFw4f5xDM3TBAA3vpOxb1osVDyOzKbEiP0JeAOinfNnQkR2S20vlPKI8fZfnTj7juDyEPtrB20fwcBH_KpEV0yGjgkbN2AkUbL_VLw2hBgsC9gMILzA32rjvOKXnCLJxtQ5vBPFlC9VSlvxW8%3D)]，它将评估细化为三个维度：

- **上下文相关性（Context Relevance）**：检索出来的文档片段是否真的与用户的提问相关？（评估检索器，避免引入无关噪声）[[2](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQEvytduq4E5bK53IMocOs_doc3KM-hMVh0M32xI30_3vHA9Iv6FPzZVuaxSNk9oO_rQwPKvGazVuc3Ad8E53shCxc4b_JNcBuHb5JMpRiYWHWkynLrgEyT6Y1x8y1InrldoQ6PAkFZEt4wCWeR1vGNEnD0uDX-ECUCYLePJMGeicXxtCFOvDbrVqj0aWVZPYw%3D%3D)]
- **忠实度/幻觉度（Faithfulness / Groundedness）**：大模型生成的回答是否完全基于检索出来的上下文？有没有凭空捏造事实？（评估生成器，检测幻觉）
- **回答相关性（Answer Relevancy）**：大模型最终的回答是否真正解答了用户的提问？[[2](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQEvytduq4E5bK53IMocOs_doc3KM-hMVh0M32xI30_3vHA9Iv6FPzZVuaxSNk9oO_rQwPKvGazVuc3Ad8E53shCxc4b_JNcBuHb5JMpRiYWHWkynLrgEyT6Y1x8y1InrldoQ6PAkFZEt4wCWeR1vGNEnD0uDX-ECUCYLePJMGeicXxtCFOvDbrVqj0aWVZPYw%3D%3D)]

此外，还有以下传统和辅助评估指标：

- **传统检索指标**：命中率（Hit Rate）、平均逆文档排名（MRR）、NDCG、召回率（Recall）等（用于独立调优向量数据库和检索算法）[[1](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQEpnMLa4E3nXrGTjow_Rpnm15f3hPEljX9LIAZNPZXAcS2qkrpsm2kZx14BgAcHct4ZkZfPUQJVRS9NzRxxicUgRoBPOdZVIRUuX2uRvLlkHI4siQqo6rRHpoQsss7Ph1TH5lWFEKeWBkRSmGLAazGMO_zq)]。
- **LLM-as-a-Judge（大模型作为裁判）**：使用能力更强的大模型（如 GPT-4）对生成质量进行语义层面的打分或成对比较（Pairwise Comparison）。

#### 2. 主流评测工具

- **Ragas (Retrieval Augmented Generation Assessment)**[[2](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQEvytduq4E5bK53IMocOs_doc3KM-hMVh0M32xI30_3vHA9Iv6FPzZVuaxSNk9oO_rQwPKvGazVuc3Ad8E53shCxc4b_JNcBuHb5JMpRiYWHWkynLrgEyT6Y1x8y1InrldoQ6PAkFZEt4wCWeR1vGNEnD0uDX-ECUCYLePJMGeicXxtCFOvDbrVqj0aWVZPYw%3D%3D)]
  - **特点**：目前最流行的开源RAG评测框架。基于研究论文提出了一系列无参考（Reference-free）指标，无需人工标注的黄金标准（Golden Dataset）即可利用大模型对上下文精度、忠实度、回答相关性进行自动化打分[[2](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQEvytduq4E5bK53IMocOs_doc3KM-hMVh0M32xI30_3vHA9Iv6FPzZVuaxSNk9oO_rQwPKvGazVuc3Ad8E53shCxc4b_JNcBuHb5JMpRiYWHWkynLrgEyT6Y1x8y1InrldoQ6PAkFZEt4wCWeR1vGNEnD0uDX-ECUCYLePJMGeicXxtCFOvDbrVqj0aWVZPYw%3D%3D)]。
  
- **DeepEval (Confident AI)**[[2](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQEvytduq4E5bK53IMocOs_doc3KM-hMVh0M32xI30_3vHA9Iv6FPzZVuaxSNk9oO_rQwPKvGazVuc3Ad8E53shCxc4b_JNcBuHb5JMpRiYWHWkynLrgEyT6Y1x8y1InrldoQ6PAkFZEt4wCWeR1vGNEnD0uDX-ECUCYLePJMGeicXxtCFOvDbrVqj0aWVZPYw%3D%3D)]
  - **特点**：开源的LLM系统单元测试框架，完美兼容 Pytest[[2](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQEvytduq4E5bK53IMocOs_doc3KM-hMVh0M32xI30_3vHA9Iv6FPzZVuaxSNk9oO_rQwPKvGazVuc3Ad8E53shCxc4b_JNcBuHb5JMpRiYWHWkynLrgEyT6Y1x8y1InrldoQ6PAkFZEt4wCWeR1vGNEnD0uDX-ECUCYLePJMGeicXxtCFOvDbrVqj0aWVZPYw%3D%3D)]。它提供了丰富的RAG评估指标，并且由于其单元测试的属性，极易与CI/CD管道集成，适合在代码部署前进行回归测试。
  
- **TruLens (Truera)**[[3](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQFQb_W0s_DBLGHWL4jFw4f5xDM3TBAA3vpOxb1osVDyOzKbEiP0JeAOinfNnQkR2S20vlPKI8fZfnTj7juDyEPtrB20fwcBH_KpEV0yGjgkbN2AkUbL_VLw2hBgsC9gMILzA32rjvOKXnCLJxtQ5vBPFlC9VSlvxW8%3D)]
  - **特点**：RAG三元组概念的鼻祖[[3](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQFQb_W0s_DBLGHWL4jFw4f5xDM3TBAA3vpOxb1osVDyOzKbEiP0JeAOinfNnQkR2S20vlPKI8fZfnTj7juDyEPtrB20fwcBH_KpEV0yGjgkbN2AkUbL_VLw2hBgsC9gMILzA32rjvOKXnCLJxtQ5vBPFlC9VSlvxW8%3D)]。提供强大的可视化面板，能够直观地追踪哪一次查询的哪个环节（检索还是生成）出现了瓶颈，非常适合开发阶段的调试和调优。
  
- **Arize Phoenix**
  - **特点**：开源的LLM可观测性和评估平台[[2](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQEvytduq4E5bK53IMocOs_doc3KM-hMVh0M32xI30_3vHA9Iv6FPzZVuaxSNk9oO_rQwPKvGazVuc3Ad8E53shCxc4b_JNcBuHb5JMpRiYWHWkynLrgEyT6Y1x8y1InrldoQ6PAkFZEt4wCWeR1vGNEnD0uDX-ECUCYLePJMGeicXxtCFOvDbrVqj0aWVZPYw%3D%3D)]。支持OpenTelemetry协议（OTel-native），能够将检索到的文本转化为向量空间进行可视化（UMAP），方便定位检索“死角”[[3](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQFQb_W0s_DBLGHWL4jFw4f5xDM3TBAA3vpOxb1osVDyOzKbEiP0JeAOinfNnQkR2S20vlPKI8fZfnTj7juDyEPtrB20fwcBH_KpEV0yGjgkbN2AkUbL_VLw2hBgsC9gMILzA32rjvOKXnCLJxtQ5vBPFlC9VSlvxW8%3D)]。
  
- **LangSmith / Langfuse**
  - **特点**：链路追踪（Tracing）领域的顶流工具。虽然它们本身是可观测性平台，但提供了强大的数据集管理、人工反馈收集和自动打分流程，对串联了多步检索的复杂RAG管道十分友好。
  
  



### 3.7.2 Agent能力评测

相比RAG，Agent（智能体）的评估要复杂得多。Agent 通常涉及多轮对话、工具调用（Tool Use）、自我规划（Planning）以及外部环境交互。它的失败往往是组合性的（例如：即使每一步工具调用都正确，但整体逻辑推理错误，导致任务失败）[[9](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQHH3zV5XPTS031AmsWRMqpp8tez5_iotqefNuoU-KD8kZTtQXnyfZgNaFmWcMzHdL_vbRlsvDepA_ycBgXdHrl9k4SZWXvfN8EC-eNDn82MHb4dg3-dCVlNVSa_MxyCK1ZzwjdP4eZeL4HSVN4vXAI7D83iSVE0ZGGO5XCV5_2S_DQqh9Fb0CmmbgUJC3e3XKTN5jO9G9XILcCq3Q%3D%3D)]。

#### 1. 评估方法

- **轨迹评估（Trajectory & Trace Evaluation）**：不仅看最终结果，更要分析Agent决策路径中的每一步[[7](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQHcEdf9OJaRylSWyHWPgeBml8sTqo5XbsbZ26hkYS5r8rxmKrJ-mmnjcT5nxafdXz8t0clKwGce5LX849mUfvWYQt2H6_no_VNCpVvZQf-gqfNapLiEF2XugTwOIRdjgxK-o0BexRjtnWP1UyEiE3HI)]。例如：是否调用了正确的工具？传参是否正确？是否存在死循环或重复无意义尝试？
- **任务成功率（Task Success Rate）**：端到端的通过率测试（Pass/Fail）。判断Agent在沙盒环境（如代码库、模拟网页）中操作后，是否达到了最终预期状态。
- **效率与成本指标**：
  - **执行步数（Steps）**：达到目标用了多少步？
  - **消耗Token及费用（Cost）**：单次任务消耗了多少API费用[[11](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQELEAfwDL9FzlMOmSUAzfoYQVyayLHUMmAe9IwiCx4wrAA-R7zEcO6t5uHK3Id1IJJP4Opk6L7LJ0mYRg5AFIbzHV1knyEKijJ88PHB35mxuf3WCUrP0jkGGMYMp1eNvIHv0_5ZlAsXIPy79_ETf583N8GpNN46dhg6zifNeA%3D%3D)]？
  - **时间（Latency）**：执行任务耗时多久？
- **异常处理与纠错能力（Robustness）**：当遇到工具调用报错、网络超时或用户干扰时，Agent是否能够自我纠错并继续执行。

#### 2. 评测基准（Benchmarks）

在Agent评测中，高质量的测试集/模拟环境（基准）比单纯的工具更关键。目前业界公认的主流基准有：

- **SWE-bench**：软件工程智能体基准[[8](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQGWfqYWht6807GcS0oHJxxtxJb1BCQXBL9Uu49knnErgSKP7pIbBeKdJnfnOFTSWsXxAVg4y781cJlOr5IogPH4GxgZa0wniS3sdFmgSoHN2CgrqhHms2ZRvaMV7eBJneMoXrxn4tWWqyHa)]。要求Agent在真实的、庞大的GitHub开源代码库中定位并解决真实的Issue[[8](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQGWfqYWht6807GcS0oHJxxtxJb1BCQXBL9Uu49knnErgSKP7pIbBeKdJnfnOFTSWsXxAVg4y781cJlOr5IogPH4GxgZa0wniS3sdFmgSoHN2CgrqhHms2ZRvaMV7eBJneMoXrxn4tWWqyHa)]。由于需要读写代码、运行测试，这是目前难度最高、最贴近生产环境的评测之一。
- **GAIA (General AI Assistants)**：通用AI助手基准[[8](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQGWfqYWht6807GcS0oHJxxtxJb1BCQXBL9Uu49knnErgSKP7pIbBeKdJnfnOFTSWsXxAVg4y781cJlOr5IogPH4GxgZa0wniS3sdFmgSoHN2CgrqhHms2ZRvaMV7eBJneMoXrxn4tWWqyHa)]。包含各种需要多模态理解、长跨度推理、复杂工具使用（如查网页、读取PDF、运行Python脚本等）的实际生活与工作任务[[8](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQGWfqYWht6807GcS0oHJxxtxJb1BCQXBL9Uu49knnErgSKP7pIbBeKdJnfnOFTSWsXxAVg4y781cJlOr5IogPH4GxgZa0wniS3sdFmgSoHN2CgrqhHms2ZRvaMV7eBJneMoXrxn4tWWqyHa)]。
- **WebArena / Mind2Web**：网页交互智能体基准[[8](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQGWfqYWht6807GcS0oHJxxtxJb1BCQXBL9Uu49knnErgSKP7pIbBeKdJnfnOFTSWsXxAVg4y781cJlOr5IogPH4GxgZa0wniS3sdFmgSoHN2CgrqhHms2ZRvaMV7eBJneMoXrxn4tWWqyHa)]。模拟真实网页环境（如电商、社交论坛、Gitlab等），评估Agent在网页端进行导航、搜索、下单等端到端操作的能力[[8](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQGWfqYWht6807GcS0oHJxxtxJb1BCQXBL9Uu49knnErgSKP7pIbBeKdJnfnOFTSWsXxAVg4y781cJlOr5IogPH4GxgZa0wniS3sdFmgSoHN2CgrqhHms2ZRvaMV7eBJneMoXrxn4tWWqyHa)]。
- **BFCL (Berkeley Function-Calling Leaderboard)**：加州大学伯克利分校提出的函数调用基准[[8](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQGWfqYWht6807GcS0oHJxxtxJb1BCQXBL9Uu49knnErgSKP7pIbBeKdJnfnOFTSWsXxAVg4y781cJlOr5IogPH4GxgZa0wniS3sdFmgSoHN2CgrqhHms2ZRvaMV7eBJneMoXrxn4tWWqyHa)]。专门用来评估模型在单步/多步/并行工具调用、参数提取上的准确率，是智能体工具调用能力的“黄金标准”。
- **AgentBench**：清华大学等机构提出的多维度评测框架，涵盖操作系统（OS）、数据库（DB）、网络浏览、卡牌游戏等八大虚拟交互环境[[8](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQGWfqYWht6807GcS0oHJxxtxJb1BCQXBL9Uu49knnErgSKP7pIbBeKdJnfnOFTSWsXxAVg4y781cJlOr5IogPH4GxgZa0wniS3sdFmgSoHN2CgrqhHms2ZRvaMV7eBJneMoXrxn4tWWqyHa)]。

#### 3. 评测与可观测性工具

由于Agent链路的非确定性，评测工具往往与可观测性追踪（Tracing）深度绑定：

- **LangSmith (针对 LangGraph/LangChain 等)**
  - **特点**：在评估多轮对话、多Agent协作（Multi-agent）和状态机工作流方面极具优势。它能清晰展现每个Agent的决策分支、记忆读取和工具交互 trace。
- **DeepEval (Confident AI)**
  - **特点**：引入了针对Agent的专项评估指标（如Tool Correctness），支持对带有有向无环图（DAG）结构的Agent工作流进行端到端回归测试，帮助团队在CI/CD阶段拦截有缺陷的Agent发布。
- **Braintrust**
  - **特点**：集成CI/CD的Agent测试工具，支持大规模高并发的Agent运行模拟，能将多次实验的评估指标与代码提交（PR）关联，适合工程化成熟度高的团队。
- **W&B Weave / Comet Opik**
  - **特点**：传统的机器学习实验追踪工具进军Agent领域的产品。提供零配置的Agent Tracing、提示词/参数优化器（Agent Optimizer）以及轻量级SLM（小语言模型）本地打分评估器。
- **Galileo**
  - **特点**：支持将离线测试与实时线上护栏（Guardrails）相结合。不仅在发布前评估Agent的轨迹合理性，还能在生产环境中监控Agent是否产生了死循环、越权工具调用或越轨行为。



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



## 3.8 Agentic RL（⭐）

Search-R1

TinyZero

ReTool

Multi-Turn-RL-Agent

DeepResearcher



## 3.9 Multi-Agent（⭐）

Here are the main patterns for building multi-agent systems, each suited to different use cases:

| Pattern                                                      | How it works                                                 |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| [**Subagents**](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents) | A main agent coordinates subagents as tools. All routing passes through the main agent, which decides when and how to invoke each subagent. |
| [**Handoffs**](https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs) | Behavior changes dynamically based on state. Tool calls update a state variable that triggers routing or configuration changes, switching agents or adjusting the current agent’s tools and prompt. |
| [**Skills**](https://docs.langchain.com/oss/python/langchain/multi-agent/skills) | Specialized prompts and knowledge loaded on-demand. A single agent stays in control while loading context from skills as needed. |
| [**Router**](https://docs.langchain.com/oss/python/langchain/multi-agent/router) | A routing step classifies input and directs it to one or more specialized agents. Results are synthesized into a combined response. |
| [**Custom workflow**](https://docs.langchain.com/oss/python/langchain/multi-agent/custom-workflow) | Build bespoke execution flows with [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview), mixing deterministic logic and agentic behavior. Embed other patterns as nodes in your workflow. |

------

# 四、部署

Python 进阶：异步 asyncio、Pydantic（函数参数结构化，Function Calling 必用）、FastAPI 封装 Agent 服务接口；

向量数据库：Chroma、FAISS、Milvus、Qdrant（熟练建库、插入、检索）；

大模型本地部署：vLLM/SGLang 部署开源模型（Qwen/Llama3/Mistral），对接私有 LLM 做本地 Agent；

容器：Docker 打包 Agent 项目，基础 Docker-Compose。



## 4.1 LLM推理优化（⭐）

量化基础：FP8、GPTQ、AWQ，推理优化基础（vLLM/SGLang/PagedAttention），看懂推理链路。



## 4.2 服务端部署

- [vLLM](https://docs.vllm.ai/?utm_source=chatgpt.com)
- [SGLang](https://docs.sglang.ai/?utm_source=chatgpt.com)
- mindie

面试高频：

- Prefix Cache
- KV Cache
- Continuous Batching
- Paged Attention
- Speculative Decoding



## 4.3 端侧部署

### 1. ONNX Runtime Mobile (ORT Mobile)

- 定位：**跨平台端侧推理首选**，适配 Android/iOS/ 车机 Linux，支持 ONNX 模型
- 优势：体积小、功耗低、算子优化成熟，支持 CPU/GPU/NPU 异构加速
- 适用：绝大多数轻量 LLM、多模态模型，工程落地最稳
- 场景：手机 APP、常规车机座舱

### 2. TensorFlow Lite (TFLite / TFLite Micro)

- 定位：谷歌端侧推理框架，分标准版 + 微型版
- 优势：生态完善、编译简单，TFLite Micro 适配**极低算力 MCU / 低配车机**
- 局限：大 LLM 算子支持偏弱，更适合小模型、CV、语音
- 场景：低端车机、物联网终端、嵌入式设备

### 3. llama.cpp（现象级开源）

- 核心：纯 C/C++ 实现，无依赖，极致轻量化
- 亮点：支持 GGUF/GGML 量化（4bit/8bit）、CPU/GPU 加速，跨平台无敌
- 适配：Android、iOS、Linux 车机、嵌入式 Linux，**车载 / 移动端离线 LLM 首选**
- 用法：可封装 JNI/OC 接口集成进 APP、车机应用，社区模型最全



## 4.4 端侧 Agent





## 4.5 标准OpenAI接口

### requset



### response



------



# 五、其它

## **A2A**



## CodeAct



## Skills（⭐）



## Harness

### Harness Evolving

#### AutoPE

#### Memory Evolve

#### AlphaEvolve

#### Hyper-Agents





------



# 六、项目

## 6.1 工作经历

### 6.1.1 **马上消费金融有限公司**（2025.10-至今）

#### A. [对话数据标注自进化Agent系统](https://chatgpt.com/c/6a2a4aff-22ac-83ea-bdf4-3631c4dbf48d)

##### 简历版

```markdown
项目名称：对话数据标注自进化Agent系统
技术栈：Agent Framework, LLM, Prompt Optimization, Text Grad, GRPO, Dynamic RAG, LangGraph
项目描述：针对大模型在垂直业务落地中手工 Prompt 调试成本高、质量不稳定的痛点，设计并实现了一套参数无关（Parameter-Free）的推理增强系统，通过在标注过程中动态沉淀领域经验与自动优化 Prompt，实现低成本迭代。
核心工作：
	1.	自进化标注与非参数化 GRPO 优化：利用 Few-shot 引导构建分类体系；适配 GRPO 的对比机制，通过多类别样本对比提炼分类边界；结合 Text Grad 思想分析推理轨迹并生成修正规则，沉淀至经验库。该机制在实际业务中将“电销黑名单”分类准确率由 81.9% 提升至 89.96%，“试驾邀约”意图识别由 58% 提升至 73%（主流场景平均提升 8% 以上）。
	2.	分层协同与动静隔离架构：构建“主分类 + 易错诊断 + 专有二分类”分层 Agent 协同机制。将高消耗的迭代反思置于异步“编译期”运行，线上“运行期”采用“精简 Prompt + 动态检索经验库”的轻量化架构，使策略调优周期由天级缩短至小时级，降低推理延迟与 Token 消耗。
	3.	防漂移评测与动态经验剪枝：建立自动化黄金验证集（Golden Set）回归测试机制以防止语义漂移；采用向量与 BM25 混合检索按需召回规则，并设计定期合并去重机制控制经验库规模。将单次线上推理的延迟增量控制在秒级，确保了系统运行的高性能与低延迟。

```



##### 模拟面试

```markdown
https://chatgpt.com/c/6a2d55bd-b3a8-83ea-9608-e0771a9229e3

项目名称：[对话数据标注自进化Agent系统](https://chatgpt.com/c/6a2a4aff-22ac-83ea-bdf4-3631c4dbf48d)

**项目角色**：独立设计与研发负责人

**项目背景**：针对大模型在垂直业务（如意图识别、复杂多分类、逻辑推理）落地中面临的**手工 Prompt 调试成本高、质量不稳定、领域知识难以持续积累**等痛点，设计并实现了一套无需参数微调（Parameter-Free）的推理增强系统。该系统通过在推理/标注过程中动态沉淀领域经验、自动优化 Prompt，实现了低成本、可解释的性能迭代。

**技术栈**：Agent Framework, LLM API, Prompt Optimization, Text Grad (文本梯度), GRPO, Dynamic RAG, Workflow Orchestration

**核心职责与技术实现**

1. **自进化语料标注机制**
   - 设计并实现自进化语料标注 Agent，利用少量种子样本（Few-shot Seeds）引导大模型自顶向下构建分类知识体系。
   - 建立多维推理链路分析模块，实现对海量对话数据的自动标注与持续优化，解决传统人工标注成本高、迭代慢的问题。
2. **非参数化 Multi-class GRPO 与 Text Grad 协同优化**
   - **范式扩展**：将 GRPO（群体相对策略优化）的“单样本-多轨迹”对比机制适配至提示词优化场景。通过在单个 Batch 中引入多类别样本，引导模型对跨类别的推理轨迹进行对比，显式提炼分类边界。
   - **文本梯度反思**：基于 Text Grad 思想，通过 Agent 自省分析正逆向推理轨迹的差异，生成文本级“梯度修正意见”，并将提炼出的泛化规则以结构化 JSON 格式沉淀至动态经验库。
   - **自适应反馈调节**：设计与 Batch 准确率挂钩的自适应奖励函数。在低准确率区间降低更新门槛（侧重探索与快速修正），在高准确率区间收紧更新门槛（侧重利用与系统稳定），平滑引导 Prompt 与经验库的协同演进。
3. **分层 Agent 协同架构与动静隔离优化**
   - 构建由“多分类主 Agent + 易错类别诊断 Agent + 专有二分类 Agent”组成的分层协同闭环。
   - **运行期与编译期解耦**：将重资源的 Multi-class GRPO 迭代、Text Grad 反思等步骤放在异步“编译期”运行；线上真实推理（运行期）则采用“精简版 Prompt + 动态检索关联经验”的轻量化架构，降低系统延迟与 Token 消耗。
4. **量化评测门禁系统与防漂移机制**
   - **黄金验证集（Golden Set）防漂移门禁**：为防范自进化过程中的“语义漂移”与“灾难性遗忘”，建立自动化评测 Pipeline。每次 Prompt 或经验库迭代更新前，强制在黄金验证集上进行回归测试，指标未发生退化方可准入上线。
   - **经验库动态剪枝与路由**：引入基于向量与 BM25 的混合检索机制，在线推理时仅按需召回 Top-K 相关规则；同时设计定期合并与去重机制，保持经验库的规模处于合理区间。

**效果与业绩：**

- **多场景准确率显著提升**：方案在多个实际业务场景中完成落地验证。其中，“电销黑名单”分类准确率由 **81.9% 提升至 89.96%**，“试驾邀约”意图识别准确率由 **58% 提升至 73%**；多数主流场景的标注准确率平均提升 **8% 以上**，标注一致性与标准化程度得到明显改善。
- **具备鲁棒的自诊断与偏差发现能力**：在“质检分类”等未见明显指标提升的复杂场景中，利用系统的“易错类别发现 Agent”与自评估机制，主动诊断并定位出评估数据源自身存在的标注偏差，避免了算法盲目拟合噪声，验证了系统评估体系的客观性与严谨性。
- **标注效率跃升与人工降本**：通过全自动化的 Prompt 与经验库迭代流程，将策略调优周期由传统人工的**数天缩短至小时级**；大幅减少了标注过程中的人工干预和校验频次，显著提升了海量数据的整体标注吞吐量。
- **高工程性能与泛化扩展性**：通过引入动态经验库压缩机制，单次推理延迟增量控制在极低水平（毫秒级），满足高并发标注的性能要求；系统分层架构具备良好扩展性，可无缝复用至 RAG、Agent 等更复杂的上下文学习（In-Context Learning）场景。
```



**面试时的口语化讲法：**
 我做的是一个面向对话语料标注的自进化智能体系统。这个系统的目标是解决人工标注贵、慢，而且遇到新业务和新表达时很难快速适应的问题。整体上分成三层：先用通用粗分类做初始标注，再根据易错类别做自进化优化，最后通过细分类模块对高混淆样本逐级校验。系统还会自动优化提示词，让大模型在不训练参数的情况下持续提升标注效果。



##### 模拟问题

###### Q1：为什么选择"无训练"方案而不是直接微调模型？

> **回答要点**：微调成本高（算力、数据标注）、周期长、难以快速迭代；无训练方案通过推理阶段积累经验，零算力成本、实时可更新，适合资源受限的业务场景。可对比 RAG（检索外部知识）与本方案（自生成内部经验）的差异。

###### Q2：为什么选择 PromptWizard 和 TextGrad 两套框架？它们的核心差异是什么？

> **回答要点**：PromptWizard 是微软提出的示例驱动搜索方案，适合有标注数据的分类/问答场景；TextGrad 将"文本梯度"概念引入优化，适合需要多跳推理的复杂任务。两者互补，统一 Config 层降低切换成本。

###### Q3：经验库如何避免"噪声经验"污染后续推理？

> **回答要点**：通过两阶段机制：① 单 Batch 内先生成 candidate_experiences 而非直接更新，避免单条坏样本直接写入；② Epoch 结束后的压缩阶段通过 LLM 二次审查合并冗余、淘汰低质量条目。可进一步讨论设置置信度阈值等改进方向。

###### Q4：多线程并发调用 LLM API 时，如何处理限速（Rate Limit）问题？

> **回答要点**：通过 `max_workers` 参数控制并发数，根据 API 配额动态调整；可引申讨论指数退避重试策略、Token 用量监控、批量请求 vs 并发请求的权衡。

###### Q5：经验的"增删改合并"操作由 LLM 自身决策，如何保证 JSON 格式输出的鲁棒性？

> **回答要点**：代码中使用 `split('```json')[-1].split('```')[0]` 做格式抽取，外层 try/except 兜底，失败时跳过而非崩溃（continue）。可进一步讨论引入 Pydantic 校验、Few-shot 格式示例强化输出稳定性

###### Q7：如何评估先验经验的质量与有效性？

> **回答要点**：当前方案以最终答案正确率作为隐式反馈；可扩展讨论：引入显式的经验有效性评分（如某条经验在多少问题上有帮助）、A/B 测试（加/不加经验的准确率对比）。
>
> **双路消融对比**：
>
> - **方法**：在同一验证集（Validation Set）上运行两路 Pipeline：
>
>   - **对照组（Base Line）**：仅使用原始 Prompt，不引入 JSON 经验库。
>   - **实验组（Exp Line）**：在输入端注入当前版本的 JSON 经验库。
>
> - **评估指标**：计算 **准确率 ** 和 **F1-Score** 增量
>
>   。若某条经验加入后 ΔAcc < 0，说明该经验存在负迁移，应予以剔除或降级。
>
> **留一交叉验证**：
>
> 1. **留一交叉验证（Leave-One-Out Validation）**：
>    - **方法**：针对经验库中的多条经验 E={e1,e2,...en}E={e1,e2,...en}，每次有意识地移除其中一条 ei，观察整体表现的变化。
>    - **评估指标**：计算经验的**边际贡献度**。边际贡献为负或接近于零的经验会被标记为“低效经验”，触发清理机制。

###### Q8：新沉淀的经验与既有经验产生逻辑冲突，如何解决？

> 1. **语义冲突检测（Conflict Detection）**：
>    - **方法**：利用专有的“审计 Agent”或符号化规则，对新生成的 JSON 经验与既有经验进行两两比对。
>    - **评估案例**：若经验 A 指出“‘退款’关键词属于‘售后服务’”，而新经验 B 指出“‘退款’且伴随抱怨语时应归入‘投诉’”，系统需评估两者的覆盖边界。
>    - **解决机制**：若判定为冲突，则触发“规则融合（Merge）”或“条件细化（Refinement）”，通过补充前置条件（如 Context-specific constraints）来消解冲突。
> 2. **语义冗余度与合并（Redundancy & Deduplication）**：
>    - **方法**：计算新经验与旧经验在向量空间（Embedding）中的余弦相似度。
>    - **评估指标**：当相似度超过设定阈值（如 0.85）时，判定为表述冗余。系统将自动调用大模型进行“规则归并”，用更具泛化性的表述合并这两条经验。

###### Q9：如何提升跨样本的泛化能力？

> **回答要点**：高温度增加输出多样性，保证 N 条采样轨迹覆盖不同推理路径（正确与错误），为经验提取提供更丰富的对比素材。若 temperature 过低，N 条输出高度相似，经验提取价值有限。
>
> 1. **跨批次泛化测试**：
>    - **方法**：将在批次 T 中提炼出的经验，立即在完全独立的批次 T+1、T+2 上进行盲测。
>    - **评估指标**：**泛化衰减率。如果在提炼批次上准确率提升 15%，但在新批次上仅提升 1%，说明该经验过拟合了特定批次的语料（如特定客户的说话口癖），属于低质量经验。
> 2. **生存周期与置信度衰减**：
>    - **方法**：给每条 JSON 经验赋予一个初始置信度（Confidence Score）和调用计数器。
>    - **评估机制**：在后续标注中，每当某条经验被成功检索并调用：
>      - 若导致标注正确：置信度增加（奖励）；
>      - 若导致标注错误：置信度大幅扣减（惩罚）。
>      - 当置信度低于阈值，或者长期未被调用，该经验将被自动物理删除或归档，以保持经验库的“代谢活性”。
> 3. **引入“黄金验证集”的门禁机制**：
>    - **方法**：在每次经验库或 Prompt 更新后，不能仅依赖当前 Batch 的准确率，还需自动在代表性历史样本集上进行微调级别的回归测试。

###### Q10：`temperature=1.0` 的设计意图是什么？

> **回答要点**：高温度增加输出多样性，保证 N 条采样轨迹覆盖不同推理路径（正确与错误），为经验提取提供更丰富的对比素材。若 temperature 过低，N 条输出高度相似，经验提取价值有限。

###### Q11：这套系统如何从实验原型落地到生产环境？你会做哪些工程化改造？

> **回答要点**：① 经验库改为向量数据库（如 Chroma，代码中已有依赖）实现语义检索而非全量注入；② 增加经验版本管理与回滚机制；③ 异步任务队列处理大批量数据；④ 监控 API 调用成本与经验库增长曲线。

###### Q12：如果业务方反馈模型在新数据上效果下降如何排查？

> **回答要点**：① 检查经验库是否存在过拟合训练分布的噪声经验；② 对比加/不加经验的推理结果，定位经验质量问题；③ 检查数据分布漂移（新数据与训练数据的 prompt 模式差异）；④ 必要时清空经验库重新从新数据中提取。



###### Q11：先介绍一下这个项目。不要讲技术细节，重点讲业务背景、为什么要做、最终解决了什么问题。

```markdown

```

###### Q12：听起来像RAG+Prompt优化。为什么不直接微调？SFT不比Prompt优化效果更好吗？

```markdown
原因1：场景变化太快(SFT 周期非常长)
原因2：数据规模不足(SFT 500~3000条标注数据)
原因3：解释性要求高(SFT 难解释)
```

###### Q13：你提到把GRPO改造成Prompt优化。详细讲一下。

```markdown
传统GRPO：优化对象是模型参数。
Multi-Class GRPO：
    把优化对象从：Model Weight
    变成：Prompt + 经验库规则

下一轮推理时：
动态召回。效果逐步提升。
```

###### Q14：Multi-Class GRPO和普通GRPO区别是什么？

```markdown
普通GRPO：
    同一问题
    多个轨迹比较
    
Multi-Class GRPO：
	每个batch中包含多个类别
	跨类别边界比较
	
Hard Negative
→发现问题

TextGrad
→总结规律

经验库
→长期记忆
```

###### Q15：为什么不在线学习？

```markdown
1.稳定性不可控。
2.GRPO+TextGrad 一次优化需要几十到上百次LLM调用，线上无法接受。
```

###### Q16：经验库越来越大怎么办？

```markdown
第一层：Embedding召回，经验向量化，TopK=10。
第二层：BM25召回，关键词补充，防止向量漏召回。
第三层：规则聚合，规则合并
第四层：去除老旧经验，检索次数少的去掉
```

###### Q17：线上出现过最严重的问题是什么？

```markdown
某批数据本身标注错误。
系统把错误样本当真理学习。
生成大量错误规则。
```

###### Q18：



------

#### B. 零幻觉推理研究

##### 简历版

**项目名称**：大模型零幻觉推理研究

```markdown
**项目名称**：大模型零幻觉推理研究
**技术栈**： Python、PyTorch、Qwen、RAG、FAISS、DSPy、RAGAS、Agent、vLLM
**项目描述**：
面向大模型事实幻觉与推理幻觉问题，研发基于检索增强、推理验证和自我反思机制的零幻觉推理框架，提升复杂问答与Agent场景下的可信推理能力。
**核心工作**：
    设计 Query Rewrite→Retrieval→Reasoning→Verification→Reflection 五阶段推理框架，实现动态知识增强与可信推理；
    引入 CoVe、Self-Consistency 与 Reflection Agent，实现答案核验、路径投票及自动纠错机制；
    基于 RAGAS、HaluEval 构建自动化评测体系，实现幻觉率监控与持续优化。
```



##### 模拟面试

```markdown
**项目名称**：大模型零幻觉推理研究
**技术栈**：Python、PyTorch、Transformers、RAG、FAISS、LoRA、DSPy、RAGAS、Graph of Thoughts、Self-RAG、Self-Consistency、Self Reflection、Claim Verification、CoVe（Chain of Verification）、Agent、Prompt Engineering
**项目描述**针对大语言模型在知识问答、复杂推理及Agent场景中存在的事实幻觉（Factual Hallucination）和推理幻觉（Reasoning Hallucination）问题，构建“检索增强 + 推理验证 + 自我反思”的零幻觉推理框架。通过引入动态知识检索、多路径推理、答案验证及可信度评估机制，实现模型回答过程的可追溯、可验证与可纠错，显著提升复杂任务场景下的事实准确率与推理可靠性。相关研究表明，CoVe、自验证推理、检索增强及多路径推理是当前缓解幻觉的重要方向。

解决幻觉（Hallucination）的方法已经从单纯的 RAG，发展到了 RAG + Reasoning + Verification + Agent 的组合体系。

**核心工作**
    1. 构建多阶段零幻觉推理框架
    设计 Query Rewrite → Retrieval → Reasoning → Verification → Reflection 五阶段推理链路；
    基于 Hybrid Search（BM25 + Dense Retrieval）实现动态知识召回；
    通过 Agent Router 自动判断是否需要外部知识检索，减少参数知识幻觉；
    支持多轮推理与复杂任务分解。
    2. 设计推理验证与自纠错机制
    引入 CoVe（Chain of Verification）验证链，对模型输出进行事实核验；
    采用 Self-Consistency 多路径推理投票机制，对不同推理路径进行一致性评估；
    实现 Reflection Agent，对低置信度回答自动触发重新检索与二次推理；
    构建基于规则 + LLM Judge 的幻觉检测模块，实现事实冲突识别与推理错误定位。研究表明推理幻觉往往来源于错误回溯和逻辑链中的隐性错误，因此过程级验证尤为重要。
    3. 建立可信评测体系
    基于 RAGAS、HaluEval、HalluQA 等基准构建评测方案；
    设计 Faithfulness、Answer Correctness、Groundedness、Hallucination Rate 等指标；
    开发自动化评测流水线，实现 Prompt、RAG策略及推理框架的持续优化；
    在内部测试集上显著降低模型幻觉率，提高事实一致性和推理准确率。现有研究普遍将检索增强、验证推理和专门的幻觉检测评测作为提升可信度的重要手段。
```



##### 模拟问题

###### Q1：基础介绍

```markdown
这个项目主要解决的是大模型在真实应用中两个核心问题：

1.事实幻觉（Factual Hallucination）
	模型会编造不存在的事实，比如错误的政策条款、错误的技术参数。
2.推理幻觉（Reasoning Hallucination）
	模型即使事实是对的，但推理链是错的，比如多步推理中中间步骤出错但最终看起来“合理”。

在真实场景（知识问答、Agent、企业问答系统）里，这两个问题会导致：
    回答不可控
    企业知识库失真
    Agent执行错误动作

所以我们做的是一个“零幻觉推理框架”，核心思想是：
	不只让模型“回答问题”，而是让它“可验证地回答问题”。
```

###### Q2： **你们怎么定义“幻觉”？是怎么量化的？**

```markdown
1）事实幻觉（Factual Hallucination）
	定义为：Answer 中的关键事实与可信知识源（retrieval / dataset / knowledge base）不一致
	量化方式：
        是否有 unsupported claim（RAGAS）
        是否存在 retrieval mismatch
        是否无法在 evidence 中找到支撑
2）推理幻觉（Reasoning Hallucination）
	定义为：推理链中任意 step 逻辑不成立，但最终答案可能看似正确
	量化方式：
        step-level verification（CoVe）
        self-consistency disagreement rate
        judge model 对 reasoning chain 的打分
```

###### Q3：claim 是怎么拆的？是规则拆还是 LLM拆？会不会不稳定？

```markdown
我们用了混合方式：
1）规则拆（轻量场景）
    按句号 / 逻辑连接词拆句
    适合 FAQ / 短文本
2）LLM Claim Extraction（主方案）
用 prompt 让模型输出：
    atomic claims（原子事实）
    每个 claim 对应 evidence query
```



###### Q4：Query Rewrite → Retrieval → Reasoning → Verification → Reflection，这个链路是串行的吗？还是有并行？为什么这样设计？

```markdown
整体是主链路串行 + 局部并行优化：
主链路：

Query → Rewrite → Retrieval → Reasoning → Verification → Reflection

原因：

幻觉问题本质是“逐步放大错误”
必须强制 step-wise gating

并行优化点：
1）Retrieval 并行
BM25 + Dense Retrieval 同时跑
FAISS + keyword index hybrid
2）Reasoning 多路径并行（Self-Consistency）
多个 CoT sampling
最后 vote
```

###### Q5：你这个链路 latency 怎么控制？不会很慢吗？线上怎么用？

```markdown
1）Router机制（关键）

不是所有请求都走全链路：

简单问题 → 直出 LLM
知识类 → RAG
高风险 → 全链路验证

👉 用 Agent Router 做 classification

2）Early Exit（提前终止）

如果：

retrieval confidence 高
reasoning consistency 高

👉 直接跳过 Reflection

3）缓存机制
retrieval cache（query embedding）
reasoning cache（similar query patterns）
```

###### Q6：Router 是规则还是模型？如果模型，它训练数据从哪来？

```markdown
我们用的是 轻量分类模型 + prompt distillation：

方法1（初期）
rule-based：
含“多少/谁/什么时候” → RAG
否则 → direct
方法2（最终）

训练一个 small classifier：

输入：

query embedding
intent features（NER / keyword）

标签：

direct / rag / full-verification

数据来源：

人工标注 + LLM生成弱标签（bootstrapping）
```

###### Q7：CoVe 和普通 RAG 最大区别是什么？

```markdown
普通 RAG：

“找资料 → 直接回答”

CoVe：

“先生成 → 再逐条验证 → 再修正”
```

###### Q8：CoVe 验证用的是什么模型？如果还是同一个 LLM，不是自嗨吗？

```markdown
1）Same Model Self-check（弱）
同一个 LLM
prompt：“检查是否有错误”

优点：便宜
缺点：容易自我强化错误

2）Cross Model Verification（中）
generator ≠ judge
用 Qwen / GPT / Claude 不同模型交叉验证
3）Retrieval-grounded verification（强）
每个 claim 必须匹配 evidence
用 embedding similarity + NLI judge

最终采用：

“模型 + 检索 + NLI 三重验证”
```

###### Q9：Self-RAG 和 CoVe 有什么本质区别？你为什么两个都用？不是重复吗？

```markdown
区别在“发生时机不同”，解决不同问题：

CoVe：

post-hoc verification（事后检查，修正已生成错误）

Self-RAG：

generation-in-loop（生成过程中动态检索，防止生成过程跑偏）
```

###### Q10：你这个系统上线后，幻觉率下降多少？怎么测的？有没有AB对比？

```markdown
数据集
    内部知识问答集（5k）
    HaluEval
    企业FAQ
```

###### Q11：最难优化的是哪个模块？为什么？

```markdown
最难的是：

Reflection Agent（自反思模块）

问题在于：

1）容易“过度纠错”
正确答案被误判为错误
2）不稳定
同一问题不同运行结果不同
解决方案：
1）引入置信度阈值
confidence > 0.85 → 不触发 reflection
2）规则 + LLM hybrid judge
规则过滤明显正确答案
LLM只处理边界case
```

###### Q12：如果让你重做这个项目，你会砍掉哪个模块？为什么？

```markdown
我会考虑：

砍掉“过重的 multi-path Self-Consistency”

原因：

成本太高（推理次数线性增长）
在很多企业QA场景收益有限
CoVe + retrieval 已经覆盖大部分收益

优化方向会变成：

“轻量验证 + 强检索 + selective reasoning”
```

###### Q13：

```markdown

```

###### Q14：





一个基于 LangGraph 的 Agentic RAG 原型项目，用来演示如何通过查询处理、多源检索、证据精炼和反思推理来降低问答幻觉，并输出可追溯的证据。

项目特点

- 基于 `LangGraph` 搭建工作流图，按状态驱动模块流转。
- 包含完整的 Agentic RAG 核心链路：
  - Query Processor
  - Retriever
  - Evidence Refiner
  - Reflective Reasoner
  - Orchestrator
- 提供简单的评测示例，便于验证答案与证据质量。
- 代码结构清晰，适合继续扩展为真实检索系统。



------



### 6.1.2 **浪潮通信**（2021.07-2025.10）

大模型训推与异构算力适配平台 → 通用通信大模型训练  → 智能体平台；

#### A. [大模型训推与异构算力适配平台]()

##### 简历版

```markdown
**项目名称**：大模型训推与异构算力适配平台

**技术栈**： Python、CUDA、CANN、MindIE、vLLM、SGLang、FastAPI、Docker、LoRA/QLoRA、模型蒸馏、LLM

**项目描述**：
	搭建覆盖数据处理、模型训练、模型蒸馏、推理部署及模型管理的一站式大模型平台，支持GPU、NPU（昇腾910B）、DCU等异构算力环境，实现大模型、多模态模型、Embedding及Reranker模型的统一训推与部署。累计完成10+模型跨平台适配，模型交付周期由3天缩短至1天以内，整体效率提升60%以上。

**核心工作**：
	1.数据与训练平台建设： 自研数据标注与蒸馏工具，构建高质量训练数据流水线；封装PT、SFT、DPO、蒸馏等训练范式，支持Full Fine-tuning、LoRA、QLoRA等微调方案。
	2.异构算力适配： 基于CUDA/CANN完成GPU、昇腾NPU及DCU环境适配，解决模型转换、算子兼容及国产化迁移问题，实现10+模型跨平台部署。
	3.推理与平台化建设： 集成vLLM、SGLang、MindIE等推理引擎，基于FastAPI与Docker实现模型一键部署、版本管理、灰度发布及全流程可视化管理。

**项目成果**：
    完成10+大模型及多模态模型跨硬件适配；
    模型交付周期由3天缩短至1天以内；
    部署周期由1天缩短至30分钟；
    运维成本降低50%，整体研发效率提升60%以上。
```



##### 模拟面试

```markdown
**项目名称**：大模型训推与异构算力适配平台

**技术栈**： Python、CUDA、CANN、MindIE、vLLM、SGLang、FastAPI、Docker、LLMs、LoRA/QLoRA、模型蒸馏、Embedding、Reranker、多模态模型

**项目描述**:
搭建面向企业级AI应用的一站式大模型训推与异构算力适配平台，覆盖数据处理、模型训练、模型蒸馏、推理部署、模型管理及跨硬件适配全生命周期能力。平台底层对接Kubernetes实现任务调度与资源编排，上层屏蔽GPU / NPU（昇腾910B）/ DCU等异构算力差异，统一提供模型训练与推理API能力。支持LLM、多模态模型、Embedding、Reranker及语音模型统一纳管，累计完成10+模型跨平台迁移与生产部署，将模型交付周期由3天缩短至1天以内，推理部署从1天缩短至30分钟，整体研发与运维效率提升60%以上。

**核心工作**:
1. 数据工程与模型蒸馏体系建设
    构建“数据生成 → 清洗 → 质量评估 → 回流优化”的闭环数据流水线，实现训练数据自动化生产
    基于教师模型（LLM）构建CoT级知识蒸馏机制，自动生成高质量问答对与推理链数据
    设计多维数据质量评估指标（语义重复度、困惑度、任务一致性），提升数据有效性
    将低质量人工标注依赖降低 50%+
2. 多范式训练平台研发
    统一抽象 PT / SFT / DPO / PPO/ GRPO/ KD 四类训练范式，构建可插拔训练执行框架
    支持 Full Fine-tuning / LoRA / QLoRA 自动策略选择（基于显存与任务类型）
    设计训练任务编排系统（Pipeline化），支持断点恢复、失败重试与任务优先级调度
    实现训练过程指标可观测（loss / grad norm / lr / GPU util）
3. 异构算力适配与国产化迁移
    构建 CUDA ↔ CANN 双后端适配层，实现训练与推理代码跨硬件兼容
    完成昇腾 910B（MindIE）推理链路适配，解决算子差异与图编译不稳定问题
    针对 embedding / reranker / LLM / 多模态模型建立统一转换与部署规范
    在 GPU / NPU / DCU 环境完成 10+ 模型迁移上线
4. 高性能推理与服务部署优化
    集成vLLM、SGLang、MindIE等主流推理引擎，实现不同硬件环境下推理框架统一调度；
    优化 KV Cache / batch 合并 / continuous batching 提升吞吐能力
5. 模型资产管理与平台化建设
    构建统一 Model Registry（LLM / Embedding / Reranker / 多模态）
    提供 FastAPI 标准化服务接口 + Docker 化部署链路
    对接 Kubernetes（作为底层资源调度系统，非核心自研部分）
    
**项目成果**
    支撑GPU、NPU、DCU三类异构算力环境统一管理；
    完成10+模型跨平台适配与生产部署；
    模型交付周期由3天缩短至1天以内；
    模型部署周期由1天缩短至30分钟；
    运维成本降低50%，整体研发与交付效率提升60%以上；
    支撑多个企业级AI应用场景快速落地与迭代。
```



##### 模拟问答

###### Q1：你这个平台和 Kubeflow / MLflow / SageMaker 有什么区别？

```
我们不是简单的MLOps工具替代，而是做了**“大模型训推一体 + 异构算力适配”的工程抽象层”。

三点核心差异：
1.面向大模型优化，而不是传统ML
    支持 LoRA / QLoRA / DPO / PPO / GRPO
    支持 KV cache / continuous batching 推理优化
2.异构算力统一抽象
    GPU（CUDA）+ NPU（CANN/MindIE）+ DCU统一调度
    解决算子级别兼容问题，而不是任务级调度
3.训推闭环
    MLflow只管tracking
    我们是：数据 → 训练 → 蒸馏 → 推理 → 灰度发布
```

###### Q2：你说支持 DPO / PPO / GRPO，这些你真的实现了吗？

```markdown
不是从零实现算法本身，而是做了**“训练范式工程封装 + pipeline标准化”**：

PPO / DPO / GRPO 使用开源框架（如 TRL / DeepSpeed-Chat）
我们做的是：
    数据格式统一（prompt / preference / reward）
    loss wrapper 封装
    多训练策略调度器

GRPO部分：
    本质是 group-based reward optimization
    我们在 reward ranking 层做了 batch-level grouping
    
追问兜底：
    GRPO和PPO区别你用一句话讲清？
    reward model是谁训练的？
    reward scaling怎么做？
```

###### Q3：为什么不用全量微调？

```markdown
✅标准回答：
核心是三点工程约束：
1.显存成本
    7B模型 full FT 需要 80GB+
    LoRA 只需 20~30%
2.训练效率
    QLoRA 4bit quant + LoRA adapter
    提升训练吞吐 2~3倍
3.多任务适配
    adapter可插拔
    支持多业务场景共享 base model
    
追问兜底：
    NF4为什么比 INT4更适合LLM？
    LoRA rank怎么选？
    merge adapter 会不会掉点？
```



###### Q4：vLLM 和 SGLang 为什么能提升吞吐？

```markdown
vLLM核心：
    PagedAttention → KV cache 分页管理
    避免显存碎片化

SGLang：
    structured generation + graph execution
    减少 token-by-token overhead

我们平台：
    按模型类型自动选择引擎
    decoder-only → vLLM
    tool-use / structured → SGLang

追问兜底：
    KV cache 为什么是瓶颈？
    continuous batching怎么实现？
    为什么不是 TensorRT-LLM？
```

###### Q5：你的推理优化做了什么？

```markdown
三个层面：

算子层
    fused attention
    RMSNorm融合
调度层
    continuous batching
    dynamic batch size
缓存层
    KV cache reuse
    prefix caching

效果：
    QPS 提升 2.5x
    P95 latency 降低 40%

追问：
    batch size动态怎么调？
    OOM怎么防？
    GPU利用率怎么测？
```

###### Q6：CUDA → CANN 迁移你怎么做的？

```markdown
主要三步：
算子映射
    CUDA kernel → CANN operator
    不支持的用自定义算子补
图编译适配
	PyTorch → ONNX → MindIE graph
精度对齐
	FP16/BF16误差控制

追问兜底：
    哪些算子最难迁移？
    attention怎么适配？
    昇腾性能瓶颈在哪？
```

###### Q7：GPU和NPU结果一致性怎么保证？

```markdown
1.数值层
	FP16/BF16统一精度策略
2.计算图层
	固定seed + deterministic ops
3.输出校验
	KL divergence < threshold
	token一致率评估
🔥追问：
    attention softmax误差怎么处理？
    哪些任务最容易 drift？
```

###### Q8：你说自动生成数据，怎么防幻觉？

```markdown
三重过滤机制：
rule filter
	regex / schema validation
model judge
	用 teacher model 做 consistency check
embedding filter
	去除低相似 / off-topic sample
🔥追问：
    teacher hallucination怎么办？
    数据质量怎么量化？
```

###### Q9：蒸馏具体怎么做？

```markdown
我们做的是：
    logit distillation + instruction distillation
    loss = CE + KL(student || teacher)
重点优化：
    引入 CoT distillation（只保留 reasoning skeleton）
    降低冗余推理路径
🔥追问：
    CoT会不会泄露噪声？
    student capacity不够怎么办？
```

###### Q10：训练任务失败怎么恢复？

```markdown
✅标准回答：
基于 checkpoint + state recovery：
    每 N step 自动 checkpoint
    保存 optimizer state + lr scheduler
    支持 resume training
🔥追问：
    DDP 多机断了怎么恢复？
    有没有 partial rollback？
```

###### Q11：你说效率提升60%，怎么算的？

```markdown
指标拆解：
    交付周期：3天 → 1天（pipeline自动化）
    部署时间：1天 → 60min（one-click deploy）
    人工操作减少：约 60%+
🔥追问：
    有没有 A/B test？
    是平均值还是P95？
    哪一环节贡献最大？
```

###### Q12：推理性能指标？

```markdown
标准回答模板：
    QPS：xxx
    P95 latency：xxx ms
    tokens/s：xxx
    GPU utilization：xx%
🔥追问：
    baseline是多少？
    提升来自哪里？
```

###### Q13：K8s你做了吗？

```markdown
K8s是底层资源编排系统，由基础设施团队提供，我们的工作主要在其之上做AI训练与推理任务的调度抽象层和模型生命周期管理，包括任务封装、资源请求描述、模型部署流水线等。
```

###### Q14：你这个“训推与异构算力平台”整体是解决什么问题的？

```markdown
这个平台主要解决企业在大模型落地过程中三个核心问题：

第一是模型全生命周期管理碎片化，从数据、训练、评估到部署缺乏统一链路；
第二是异构算力环境复杂，包括GPU、昇腾NPU、DCU，不同框架（CUDA/CANN）导致模型迁移成本高；
第三是推理与训练系统割裂，训练与推理各自独立，缺乏统一调度与模型资产管理能力。

因此我们构建了一个统一平台，把数据、训练、推理和部署统一成一个闭环系统。
```

Q15：你这个平台整体架构是怎么设计的？

```markdown
API层（FastAPI）：统一对外提供训练、推理、模型管理接口
调度层（Task Orchestration）：管理训练任务、推理任务生命周期
模型层（Model Registry）：管理LLM/embedding/reranker版本
执行层（Execution Engine）：对接 vLLM / SGLang / MindIE
资源层（K8s + GPU/NPU/DCU）：负责资源调度与容器运行
```

Q16：调度层你们自己做了吗？还是K8s？

```markdown
K8s负责底层容器与资源调度，我们在其上做的是AI任务调度抽象层：

包括：
    训练任务 DAG 编排
    模型依赖关系管理
    GPU显存/算力感知调度策略
    推理服务生命周期控制
```

Q17：你觉得你这个项目最大的技术难点是什么？

```
最大难点不是单点技术，而是：

👉 “异构算力 + 多模型 + 多推理引擎”的统一抽象

核心挑战：
    不同框架语义不一致
    推理路径复杂
    性能优化需要跨层协同
```

Q17：



Q17：





#### B. **[通信领域通用大模型训练与对齐优化](https://chatgpt.com/c/6a2b7987-3fa8-83ea-9c84-3d5a6946ffff)**

##### 简历版

```markdown
**项目名称**：通信领域通用大模型训练与对齐优化
**技术栈：**PyTorch、QLoRA、LoRA、Post-Train、SFT、DPO、分布式训练、模型评测体系
**项目概述**：基于开源大模型构建通信行业专属大模型训练体系，完成Post-Train持续预训练、SFT监督微调及DPO偏好对齐全流程研发；搭建数十GB通信领域语料库与多任务指令数据集，采用 QLoRA+DeepSpeed 实现低成本训练；构建专业评测体系与数据闭环优化机制，最终通信场景问答准确率提升15%，幻觉率下降7%，模型成功落地智能运维、故障诊断、招投标分析等多个生产场景。
**核心职责**：
    1.	搭建通信领域 Post-Train → SFT → DPO 完整训练流水线； 
    2.	构建数十GB领域语料、数万条 SFT 数据及偏好对齐数据集； 
    3.	基于 QLoRA+DeepSpeed 实现高效增量训练，显著降低训练成本； 
    4.	建立自动评测与数据回流闭环，持续优化模型专业能力与稳定性。
```

##### [模拟面试](https://chatgpt.com/c/6a2d0d19-d93c-83ea-83d5-c276de57e2eb)

```markdown
**项目名称**：通信领域通用大模型训练与对齐优化
**技术栈：**PyTorch、QLoRA、LoRA、Post-Train、SFT、DPO、分布式训练、模型评测体系
**项目概述**：针对通信行业算网运维、通信故障诊断、招投标分析、财务稽核等核心场景，基于 Qwen2.5-32B 开展领域化训练与能力对齐，构建覆盖“领域知识注入→指令理解→偏好对齐”的全流程训练体系；最终相较原生开源底座模型，业务场景问答准确率提升15%，模型幻觉率降低7%，故障排查场景任务完成率提升12%。
**核心职责**：
    1. **数据集构建**：整合行业文档、历史故障工单、招投标资料、运维手册等内外部私有数据，分别构建面向领域知识注入的 Post-train 无标注语料，高质量通信专属 SFT 训练数据集，以及构建通信领域偏好 DPO 数据集；`设计完整数据处理流水线，完成文本解析(PDF/Word/HTML等异构文档解析)、格式统一、脏数据过滤、相似度去重(MinHash+Embedding相似度去重)、业务数据脱敏（PII敏感信息脱敏）、高质量样本筛选等工作。`
    2. 通信领域持续预训练（Post-Train）:构建通信专业语料训练集、实现领域增量训练Pipeline、设计学习率Warmup与Cosine Decay策略、采用QLoRA + DeepSpeed降低训练成本、监控Loss、Perplexity等训练。对比实验：LoRA 和 全量参数
    3. 多阶段监督微调（SFT）:覆盖多个任务：故障定位问答、运维工单分析、CLI指令解析、招投标信息抽取、财务规则审核、专业知识问答。第一阶段：增强指令跟随、格式遵循和多轮对话能力；第二阶段：重点强化通信术语理解、故障分析推理和专业知识生成能力。
    4. 强化学习对齐（DPO）：设计评价维度：专业准确性、事实一致性、推理完整性、回答可执行性、幻觉控制
    5. **模型评测与迭代**：搭建通信领域专项评测体系。通用能力：BLEU、ROUGE、MT-Bench；专业能力：通信知识问答准确率、故障定位成功率、工单解析准确率、招投标信息抽取准确率；安全与稳定性：Hallucination Rate、Consistency Score、Refusal Rate。建立：数据分析、模型训练、自动评测、错误归因、数据优化、重新训练的持续迭代优化闭环。
    6. 落地赋能**：将成型通信大模型无缝接入公司智能体开发平台，实现垂直模型多业务场景复用。
```

基于开源大模型构建通信行业专属大模型训练体系，完成Post-Train持续预训练、SFT监督微调及DPO偏好对齐全流程研发；搭建数十GB通信领域语料库与多任务指令数据集，采用QLoRA+DeepSpeed实现低成本训练；构建专业评测体系与数据闭环优化机制，最终通信场景问答准确率提升15%，幻觉率下降7%，模型成功落地智能运维、故障诊断、招投标分析等多个生产场景。



| 数据类型       | 规模   | 用途         |
| -------------- | ------ | ------------ |
| Post-Train语料 | 数十GB | 领域知识注入 |
| SFT指令数据    | 数万条 | 业务能力训练 |
| Preference数据 | 数千组 | 偏好对齐训练 |



##### 模拟问题

###### Q1：请先介绍一下你这个项目，重点讲清楚业务背景、目标以及你负责的部分。

```markdown
这个项目的目标是构建通信领域专属大模型，覆盖算网运维、通信故障诊断、招投标分析、财务稽核等多个业务场景。

项目启动时存在三个核心问题：
    通用大模型缺乏通信领域知识，对于专业术语和故障场景理解不足；
    对通信运维流程缺乏认知，回答容易出现幻觉；
    无法满足企业内部业务要求，例如工单分析、招投标审核等专业任务。

因此我们设计了：
领域数据构建
      ↓
Post-Train领域知识注入
      ↓
SFT任务能力训练
      ↓
DPO偏好对齐
      ↓
专项评测体系
      ↓
持续迭代优化

我主要负责：
    数据集建设全流程
    Post-Train训练Pipeline
    SFT训练体系设计
    DPO偏好数据构建
    评测体系建设
    模型迭代优化
```

###### Q2：为什么要采用 Post-Train + SFT + DPO 三阶段训练，而不是直接SFT？

```markdown
因为三者解决的问题不同。
# Post-Train（属于知识缺失问题）
解决：模型不知道通信知识
例如：BBU是什么？AAU是什么？OTN和PTN区别？

# SFT（属于任务能力问题）
解决：知道知识，但不会按照任务要求使用
例如：根据告警日志定位故障，按照JSON格式输出

# DPO（DPO负责学习偏好）
解决：多个答案都正确，哪个更符合专家偏好
例如：
故障分析：
答案A：
	检查光功率
答案B：
    先检查告警等级
    再检查链路状态
    最后验证光功率
显然B更符合专家经验。
```

###### Q3：为什么选择DPO而不是RLHF？

```markdown
原因1：实现复杂度
RLHF需要：
    SFT → Reward Model → PPO
    训练链路长。
    工程复杂。

原因2：训练稳定性
PPO经常出现：
    Reward Hacking
    KL Collapse
    梯度震荡
    训练成本高。

原因3：数据现状
    已经有大量：
    Chosen，Rejected
    偏好数据。
```

###### Q4：DPO核心公式是什么？

```markdown

```

###### Q5：为什么不用ORPO或者SimPO？

```markdown
我们做过实验。
ORPO
优点：
	简单
	资源消耗小
缺点：
	专业场景提升有限

SimPO
优点：
	训练更稳定
缺点：
	对高质量偏好数据要求更高
```

###### Q6：通信数据是怎么构建的？

```markdown
主要来源：
    运维手册
    历史工单
    故障案例库
    招投标文件
    规范标准
    FAQ知识库
    
数据处理流程：
    文档解析
    ↓
    清洗
    ↓
    去重
    ↓
    脱敏
    ↓
    质量评估
    ↓
    训练集
```

###### Q7：为什么使用MinHash+Embedding双重去重？

```markdown
因为单纯MinHash无法解决语义重复。

第一层：MinHash，速度快。
第二层：Embedding，计算向量相似度。
```

###### Q8：Embedding阈值怎么定？

```markdown
测试：
    0.80
    0.85
    0.90
    0.92
    0.95
随机抽取：人工判断是否重复。
```

###### Q9：为什么选择QLoRA而不是全量微调？

```markdown
核心原因：成本
```

###### Q10：QLoRA为什么不会明显掉点？

```markdown
原因是：

训练时：
基础权重冻结
但前向传播仍使用完整知识。

实际更新的是：
ΔW = BA
低秩矩阵。

对于通信场景：
知识迁移量有限。
LoRA已经足够。
```

###### Q11：训练过程中遇到过什么线上问题？

```markdown
问题1：灾难性遗忘
    通信能力提升，通用能力下降
    原因：
        领域数据占比过高
    解决：
        加入20%通用数据混训

问题2：格式幻觉
	要求JSON输出。模型输出：解释文字+JSON
    解决：
        增加格式约束SFT样本
    并引入：
        Format Reward

问题3：故障分析逻辑跳步
	直接输出结论。没有排查过程。
    解决：
        加入：CoT数据
        例如：告警分析 → 链路分析 → 设备分析 → 故障定位
```

###### Q12：问答准确率提升15%，这个指标怎么测出来的？

```markdown
构建独立测试集：
    通信知识问答，2000条，专家人工标注标准答案。

评测方式：
    LLM Judge + 人工抽检
    
评分维度：
    事实正确性
    专业术语准确性
    逻辑完整性
```

###### Q13：如果让你重新做一次，你觉得最大的优化点是什么？

```markdown
第一：RAG增强训练
	指在训练或微调大模型本身时，利用RAG技术动态引入外部知识，来生成更高质量的训练数据。例如：先通过RAG检索相关信息，再让模型基于这些信息生成带引用的回答，然后用这些合成数据去微调模型。这能提升模型的事实性和引用能力。
	
第二：Agent能力训练

```

###### Q14：为什么选择Qwen2.5-32B而不是7B、14B、72B？

```markdown
主要从效果、成本和部署三个维度考虑。
从32B到72B：参数增加2倍，效果只提升2%，收益明显递减。
```

###### Q15：你说收益递减，依据是什么？

```markdown
参考Scaling Law。
模型能力近似满足：
L(N)=AN−α+B

其中：
N = 参数量
L = Loss（Loss 越小，模型能力、精度、效果越好）

随着参数增加：
Loss下降越来越慢

	我们判断收益递减，主要依据大模型缩放定律。模型损失 Loss 和参数量满足幂函数关系，参数量越大，Loss 下降速度越慢。结合通信场景实测：7B 到 14B、14B 到 32B 参数扩容收益都很可观，但从 32B 提升到 72B 后，效果提升明显放缓，因此判定已经进入边际收益递减区间。
```

###### Q16：Post-Train学习率怎么设置？

```markdown
Post-Train属于继续预训练，不是从头训练。
学习率过大会：覆盖原有知识（灾难性遗忘）

采用：Cosine Decay + Warmup
学习率：2e-5（通过实验所得）
Warmup：3%
```

###### Q17：怎么看训练发散？

```markdown
观察Loss曲线
监控：Gradient Norm
```

###### Q18：Post-Train阶段主要看哪些指标？

```markdown
1 Loss：训练是否正常。
2 Perplexity：语言建模能力。
3 通信专项评测：每隔500 step，跑专项Benchmark。
```

###### Q19：Checkpoint如何选择？

```markdown
Loss最低 ≠ 效果最好

every 1000 step 保存，并测：
    通信QA
    故障诊断
    工单分析
综合打分。
```

###### Q20：为什么做两阶段SFT？

```markdown
阶段1：通用指令能力。
训练：
    指令跟随
    格式遵循
    多轮对话
    
阶段2：专业能力。
训练：
    通信知识
    故障推理
    工单分析
    
如果直接混训：
模型容易：格式能力不足 或者 专业能力不足
```

###### Q21：SFT学习率为什么比Post-Train小？

```markdown
1e-5，实际上SFT更容易过拟合。
SFT数据量：几十万
远小于：Post-Train数十亿Token

学习率过大：快速记忆样本，泛化能力下降。
```

###### Q22：SFT训练多久停止？

```markdown
采用Early Stop。
监控：Validation Loss
```

###### Q23：DPO训练最重要的参数是什么？

```markdown
β（beta），DPO核心超参。
控制：偏好学习强度

| β    | 效果  |
| ---- | --- |
| 0.01 | 学不动 |
| 0.05 | 较好  |
| 0.1  | 最优  |
| 0.5  | 过拟合 |
```

###### Q24：DPO Loss下降是否代表效果提升？

```markdown
不一定。
经常出现：DPO Loss下降
但是：幻觉率上升

# 同步监控：
    Hallucination Rate
    Consistency
    Task Accuracy
```

###### Q25：Qwen2.5-32B训练为什么选择ZeRO-3？

```markdown
FP16下：32B × 2Byte ≈ 64GB

训练时还有：
Optimizer
Gradient
Activation

实际：>200GB

ZeRO-1
    仅切Optimizer
    显存不足。

ZeRO-2
    切Optimizer+Gradient
    仍然不足。

ZeRO-3
    切：
    Parameter
    Gradient
    Optimizer
    全部分片。

最适合32B训练。
```

###### Q26：ZeRO-3缺点是什么？

```markdown
通信开销大。
频繁：
    All Gather
    Reduce Scatter
```

###### Q27：梯度累积怎么算？

```markdown
公式：
Global Batch = Micro Batch × GPU数 × Gradient Accumulation

例如：
Micro Batch = 2
GPU = 8
GA = 16
那么：
Global Batch=2×8×16=256
```

###### Q28：如何提升Token利用率？

```
很多训练资源浪费在Padding。

Packing，把多个短样本拼接。
50+100+200+300
凑满：2048窗口
```

###### Q29：如果老板只给你8张A800，你怎么训练Qwen2.5-32B？

```
QLoRA
+
ZeRO-3
+
Gradient Checkpoint
+
FlashAttention2
+
Sequence Packing
```

Q30：



#### C. **企业级智能体平台**

##### 简历版

```markdown
项目名称：企业级智能体研发平台
技术栈：Python、LLM、Multi-Agent、RAG、Milvus、ASR/TTS、评测算法
项目描述：面向企业客户的低代码智能体开发平台，支持任务拆解、工具调用与检索增强生成（RAG）。本人负责LLM能力增强与RAG评测体系建设。
核心工作：
    •	LLM与检索优化：设计多层级RAG架构，实现稀疏与稠密多路召回；结合Prompt优化、答案融合与推理拆解降低模型幻觉。在测试集评估中，幻觉率降低约19%，回答准确率提升约25%。
    •	自动化评测构建：搭建RAG自动化评测模块，围绕忠实度、答案相关性、上下文精度/召回率及实体召回率进行全链路评估，替代传统人工抽检，降低约60%的评估人力成本。
    •	语音能力集成：联动ASR/TTS语音模型，实现多终端语音交互功能，提升了平台在企业级交付场景中的适配能
```



##### 模拟面试

```markdown
**项目名称：企业级智能体研发平台**
**技术栈**：Python、LLM、Multi-Agent、RAG、Milvus向量库、ASR/TTS、评测算法
**项目描述**：面向企业级客户打造低代码智能体开发平台，支持任务拆解、工具调用、自主决策、检索增强生成能力，助力业务快速搭建专属自动化智能体；本人负责LLM能力增强与RAG全维度评测体系搭建两大核心模块。
**核心工作**：
    1、LLM能力增强。优化多层级RAG检索架构，融合稀疏检索、稠密检索多路召回策略；搭配Prompt优化、答案融合、推理拆解等技术，有效降低模型幻觉，提升复杂问题回答准确率；优化后知识库问答幻觉率下降19%，答案整体准确率提升25%。

    2、RAG评估功能开发。搭建自动化评测模块，围绕**忠实度、答案相关性、上下文精度/召回率、实体召回率**五大核心指标，完成全链路自动化评测；替代传统人工抽检模式，降低60%以上RAG评估成本；

    3、平台能力适配：联动语音模型，实现智能体语音交互能力，适配多终端使用场景，完善平台企业级交付能力。
```



##### 模拟问题

###### Q1：请你先整体介绍一下这个“企业级智能体研发平台”的架构设计？

```markdown
这个平台整体是一个面向企业的低代码智能体开发系统，核心目标是让业务人员可以快速构建具备工具调用、RAG检索、多轮推理能力的智能体。
整体链路如下：

用户请求入口层
    支持文本/语音输入（ASR）
    请求进入 API Gateway（Python + FastAPI）
智能体编排层（Multi-Agent Orchestrator）
    根据任务类型进行 agent routing
    支持 planner agent + executor agent 架构
    进行任务拆解与子任务分发
知识增强层（RAG）
    多路召回：
        稀疏检索（BM25）
        稠密检索（Embedding + Milvus）
    rerank + context compression
LLM推理层
    Prompt template + tool calling
    支持 function calling / ReAct
工具执行层
	外部API调用 / 数据库 / 企业系统
输出层
	文本输出 / TTS语音输出
```



###### Q2：为什么不直接用纯向量检索？

```markdown
纯向量检索在“语义相似但关键词缺失”的情况下容易召回错误内容
在企业知识库中存在大量：
    产品型号
    编号
    规范条款
		→ 这些对关键词非常敏感，纯 embedding 会丢失精确匹配能力
```



###### Q3：稀疏检索在你们场景中解决了什么问题？

```markdown
BM25用于保证“精确匹配召回”
特别是在：
    工单号
    设备编号
    法规条款
提升 recall@k 的稳定性
```



###### Q4：Milvus在你们系统里承担什么角色？它的瓶颈怎么处理？

```markdown
Milvus负责向量索引（HNSW/IVF_FLAT）
优化点：
    IVF参数调优（nlist / nprobe）
    分库分索引（按业务域拆分collection）
    热数据cache embedding
瓶颈主要在：
	高并发 query latency
解决方式：
    batch query
    embedding cache
    限制topK + rerank前置过滤
```



###### Q5：“忠实度”你是怎么定义和计算的？

```markdown
Faithfulness = 答案是否可以被检索上下文支持
方法：
    将答案拆成 claim-level statements
    用 NLI 或 LLM judge 判断每个claim是否被context entailed
```



###### Q6：这些指标是人工标注还是自动评估？如果是自动评估，如何保证可信度？

```markdown
采用“LLM-as-Judge + rule-based hybrid”
其中：
    LLM判断语义一致性
    rule用于实体/数值对齐
为降低偏差：
    引入少量人工gold set（约200条）
    用来校准LLM评分偏置
```



###### Q7：你说降低幻觉率19%，这个“幻觉率”具体怎么算？

```markdown
定义：
	hallucination = 无法被context支持的claim数 / 总claim数
在测试集上：
    baseline：28%
    优化后：9%
    → 下降约19%
```



###### Q8：线上有没有出现过RAG“检索正确但答案仍然错”的情况？怎么定位的？

```markdown
检索正确但答案错误的问题

确实出现过
root cause：
    context中信息冲突（多个版本知识）
    LLM没有做冲突消解
解决：
    加入 context ranking（时间+权重）
    prompt加入“conflict resolution instruction”
```



###### Q9：系统延迟怎么控制？Multi-Agent + RAG + rerank 不会很慢吗？

```markdown
延迟优化

原始链路：
	RAG + rerank + agent planning → 2.8s
优化后：
    embedding cache
    并行召回（BM25 + vector async）
    rerank top50→top10
    → 降到 1.4s
```



###### Q10：如果现在让你再优化一版，你会动哪一层？

```markdown
我会重点优化三点：
    Agent planner（减少不必要的多轮推理）
    reranker轻量化（蒸馏模型替代cross-encoder）
    引入 speculative retrieval（提前预取知识）
```



###### Q11：你们这个系统到底是“智能体平台”，还是只是 RAG + prompt + tool call 的工作流编排？和 LangChain / Dify 有什么本质区别？

```markdown
客观来说，我们的系统早期确实是 workflow-based agent（工作流型智能体），但在后续迭代中逐步增强了“决策层”。

1）基础层（确实类似 LangChain）
    RAG 检索（BM25 + 向量）
    Prompt 拼装
    Tool calling 执行
👉 这一层确实和 LangChain / Dify 很接近

2）增强层（我们做的差异点）
我们引入了任务级规划器（Planner Agent）：
不是简单顺序 workflow
而是：
    任务拆解（decomposition）
    子目标排序（ordering）
    是否调用工具的决策（tool policy）
👉 这一点是区别点之一

3）平台层差异（真正工程差异）
我们不是“单个 agent runtime”，而是：
    多 agent registry（不同业务 agent）
    动态 routing（根据任务选择 agent）
    RAG + tool + voice 多模态统一编排
    
⚖️客观承认边界
但我也必须说明：
    当前版本仍然不是完全 autonomous agent（没有长期自学习闭环）
```



###### Q12：BM25 和向量召回冲突怎么办？权重怎么定？是不是拍脑袋？

```markdown
1. 基于置信度/分数的加权融合
核心思想：不同召回通道对同一信息有不同置信度（如相似度分数、BM25分数、图谱关系强度）。优先采信置信度最高的结果。
2. 基于时间/版本的新鲜度优先
核心思想：冲突往往因信息过时引起，默认采用更新时间更近的答案。
3. 基于来源权威性的层级优先
核心思想：不同来源的可信度不同（如官方文档 > 社区博客 > 自动抽取）。
4. 基于一致性检验的共识提取
核心思想：如果大多数召回通路支持同一事实，该事实更可能正确。
5. LLM作为裁判的语义裁决
核心思想：将冲突片段及各自的来源、分数、时间等上下文提交给LLM，由LLM依据逻辑和常识判断哪个更合理。
6. 避免融合：分情景返回
核心思想：如果冲突无法可靠解决，不如坦陈不确定性，让下游或用户决策。
```

###### Q13：幻觉率这个指标是不是你自己定义的？LLM-as-Judge会不会自嗨？

```markdown
1）幻觉率定义是可解释的，而不是随意定义，我们定义的是：
	hallucination = claim-level unsupported ratio
2）LLM-as-Judge确实有bias问题
    （1）模型隔离
        Judge model ≠ generation model，使用不同模型（避免 self-bias）
    （2）规则约束辅助
        数值一致性 rule check，entity match check
    （3）人工校准集
        200条 gold set，用来校准评分分布（不是直接训练，只做 bias correction）
```

###### Q14：为什么不用 LangChain / Dify 直接做？

```markdown
1）企业级定制能力
我们不是做 demo agent，而是：
    多知识域隔离
    多租户隔离
    agent policy 可控
👉 Dify / LangChain 更偏通用框架

2）评测体系（核心差异点）
我们有完整 RAG evaluation pipeline：
    faithfulness
    recall
    entity accuracy
    answer relevance
👉 这一点是很多框架没有内建的

3）语音 + agent 融合链路
我们是：
	ASR → agent → RAG → TTS
而不是纯文本链路
```

###### Q15：你说延迟从 2.8s 降到 1.4s，你具体怎么 profiling 的？有没有证据链？

```markdown
1）我们先做了 tracing（不是猜）
	embedding latency
    retrieval latency
    rerank latency
    LLM latency
    tool latency

2）真实瓶颈定位结果
	rerank + context packing 占比最高（约 40%）
	
3）优化手段
（1）rerank优化
    top50 → top10 rerank
    轻量cross encoder替换
（2）并行化
    BM25 + vector async
    tool prefetch
（3）cache
	embedding cache（query-level + chunk-level）

4）关于“证据链”
    tracing log
    p95 latency before/after
    stage breakdown dashboard
不是单点优化，是可观测系统优化
```

###### Q16：



------



## 6.2 比赛经历

### 2026-FlagOS开放计算全球大赛

#### 简历版

```markdown
FlagOS开放计算全球大赛 | 国家级行业赛事 | 2026.06 | 队长 | 三等奖
•	基于Qwen3-4B模型与ICL范式，优化Nanobot智能体运行架构，设计自适应少样本遴选策略，解决长上下文场景资源浪费、标注精度偏低等问题；引入轻量化GRPO强化学习方案，零参数更新优化模型判别能力。
•	覆盖情绪检测、主题分类、百科问答、算子代码生成多类任务，搭建代码闭环核验体系与多线程标注框架，多维度优化算法策略并完成消融实验，沉淀可落地的长文本数据标注解决方案。
```



#### 模拟面试

```markdown
**赛题：[长上下文场景中大模型自动数据标注](https://flagos.io/RaceDetail?id=296fmsd8&lang=cn)（三等奖）**

**项目描述：**本次挑战赛旨在**探索大语言模型在超长上下文（如数万至数十万token）范式下的数据标注能力提升方法**，重点评估模型在复杂、长序列标注场景中的稳定性、泛化性与实用性。


**技术难点**：
1. 在超长上下文场景下，如何设计有效的模型指令与提示策略，引导 LLM 稳定、高质量地完成数据标注任务？
2. 当可用标注示例数量显著超过模型上下文容量时，如何为待标注数据构造信息密集、结构合理的超长上下文输入？
3. 在自动多轮对话或持续交互场景中，如何高效利用超长上下文，实现一致性与可扩展性兼顾的数据标注？
```



##### 项目名称

###### 长上下文场景中大模型自动数据标注

**技术栈：**

Qwen3-4B、Context Engineering、Prompt Optimization、ICL、Multi-class GRPO、NanoBot、LLM Evaluation、Python

------

##### 项目背景

针对长上下文（32K~128K Token）自动数据标注场景，传统 Few-shot Prompt 存在上下文利用率低、类别边界模糊、Prompt 依赖人工调优等问题。

项目从：

- Context Engineering（上下文工程）
- Prompt Optimization（提示词优化）
- RL-based Boundary Learning（强化学习边界学习）

三个层面构建自动标注优化框架，实现复杂长序列场景下的高质量自动标注。

最终获得长上下文场景中大模型自动数据标注挑战赛全国三等奖。

------

##### 核心工作

###### 一、构建任务自适应上下文组织框架

针对长上下文场景中信息密度低、注意力分散的问题，设计任务级 Context Pipeline。

**上下文组成**

```
System Instruction

↓
Task Definition Card

↓
Boundary Definition

↓
Dynamic Few-shot Examples

↓
Output Constraint

↓
Target Sample
```

其中：

**Task Definition Card**

定义：

- 任务目标
- 标签含义
- 判断原则
- 特殊规则

降低模型理解成本。

------

**Dynamic Few-shot Retrieval**

设计动态示例检索机制：

固定 Few-shot：

```
每次固定选择5个示例
```

存在：

```
冗余样本过多
边界样本不足
Token浪费
```

问题。

因此设计：

```
Boundary-first Retrieval
```

优先召回：

- 类别边界样本
- 历史误分类样本
- 高置信代表样本

动态决定：

```
k ∈ [3,20]
```

而非固定数量。

实现有限 Token Budget 下的信息密度最大化。

------

###### **二、构建 Prompt 自动优化闭环**

传统 Prompt 调优依赖人工经验：

```
修改 Prompt
↓
跑测试集
↓
看结果
↓
继续修改
```

效率极低。

------

基于 NanoBot 构建 Prompt Optimization Agent：

**Error Collection**

收集：

- 错误样本
- 边界样本
- 低置信样本

------

**Error Analysis**

自动分析：

```
规则缺失
类别描述模糊
示例覆盖不足
```

等问题。

------

**Prompt Rewrite**

自动生成：

- 新规则
- 新约束
- 新示例

形成候选 Prompt。

------

**Regression Evaluation**

自动执行：

```
Prompt A/B Test
```

评估：

- Accuracy
- Macro-F1
- Consistency

指标变化。

形成：

```
Prompt
↓
Evaluate
↓
Analyze
↓
Rewrite
↓
Re-Evaluate
```

优化闭环。

------

###### 三、利用 GRPO 学习分类边界

项目最核心创新点。

------

**问题**

长上下文分类任务中：

很多错误来自：

```
类别定义不清晰
类别边界重叠
Prompt描述不充分
```

例如：

```
故障处理
VS
故障定位

投诉
VS
咨询

需求
VS
建议
```

边界非常模糊。

------

**数据构建**

将误分类样本自动收集：

```
Prediction ≠ Label
```

构建：

```
Hard Example Dataset
```

------

**Multi-class GRPO训练**

奖励函数设计：

正确分类：

```
+1
```

错误分类：

```
-1
```

边界样本正确分类：

```
+2
```

重点强化边界学习能力。

------

**Boundary Knowledge Extraction**

GRPO训练后：

分析模型学习到的：

```
类别区分特征
类别判定规则
边界描述
```

自动生成：

```
Boundary Definition Card
```

例如：

```
咨询：
以获取信息为目的

投诉：
以表达不满并要求处理为目的
```

------

**Prompt回灌**

将边界知识重新注入：

```
Prompt
Few-shot
Rule Card
```

实现：

```
RL Learning
↓
Boundary Discovery
↓
Prompt Enhancement
↓
Inference Improvement
```

闭环优化。

------

##### 技术亮点

###### Context Engineering

- 动态上下文编排
- Boundary-aware Few-shot Retrieval
- Token Budget Optimization
- 长上下文信息密度提升

------

###### Prompt Optimization

- NanoBot 自动调优
- 错误驱动 Prompt 演化
- Prompt A/B Evaluation

------

###### RL-based Boundary Learning

- Multi-class GRPO
- Hard Example Mining
- Boundary Knowledge Distillation
- Boundary Definition Generation

------

##### 面试中的一句话总结

> 这个项目本质上是在长上下文自动标注场景下，构建了一套“Context Engineering + Prompt Optimization + RL-based Boundary Learning”的闭环优化系统。通过动态上下文组织提升信息利用率，通过NanoBot实现Prompt自动优化，再利用Multi-class GRPO从误分类样本中学习类别边界知识并回灌Prompt，最终实现自动标注能力的持续迭代与自进化。



#### 模拟问题

##### Q1：先简单介绍一下你认为最有代表性的大模型项目，你解决的核心问题是什么？

```markdown
这个项目是一个长上下文（32K~128K Token）场景下的大模型自动数据标注系统，核心目标是在不使用 embedding、reranker 等外部神经网络模型的前提下，仅依赖 Qwen3-4B，通过 Prompt + Context Engineering + RL 来提升标注质量。

传统 Few-shot 标注在长上下文中存在三个问题：

上下文过长导致注意力分散
类别边界模糊导致误判
Prompt 依赖人工调优，迭代效率低

因此我们从三个方向优化：

Context Engineering：提升信息密度
Prompt Optimization：自动化迭代 Prompt
GRPO：学习类别边界

最终在比赛中获得全国三等奖。
```

##### Q2：为什么长上下文场景下 Few-shot 效果会变差？

```markdown
第一：长上下文存在明显的 Attention Dilution（注意力稀释）。
第二：固定 Few-shot 很难覆盖边界场景。
第三：长上下文会出现 Lost-in-the-Middle(开头/结尾,信息利用率最高)。
```

##### Q3：Lost-in-the-Middle你们是怎么验证的？

```markdown
我们做过位置消融实验。
固定 Prompt 内容：
	Prompt内容完全一致
仅调整 Few-shot 位置：
    Head
    Middle
    Tail
```

##### Q4：为什么不用 RAG？

```markdown
# 不让使用embedding，无法获取语义相似度
```

##### Q5：你们怎么做“检索 few-shot 示例”的？完全不用向量模型吗？

```markdown
Rule-based + Label-aware Selection

具体做法：
    基于标签分桶（label bucket）
    基于历史误分类样本池（error memory）
    基于规则匹配（关键词 + 模板匹配）
   	Boundary-first sampling（优先选“模型曾经错过的样本”。）
```

##### Q6：没有 embedding，你怎么判断“边界样本”？不会很主观吗？

```markdown
我们不是语义相似度判断，而是用行为信号定义边界样本。
边界样本定义为：

模型历史上：
    1. 预测错误
    2. 或置信度低（logprob proxy）
    3. 或多次预测不稳定
```

##### Q7：你说你做了 Context Pipeline，这一套到底解决了什么问题？为什么普通 Prompt 不行？

```markdown
普通 Prompt 在长上下文中有两个核心问题：
1️⃣ 信息衰减
    early token 被弱化
    instruction 被覆盖
2️⃣ 结构混乱
	few-shot 示例和规则混在一起，模型无法区分优先级

我们 Context Pipeline 做了结构化：
    System
    ↓
    Task Card
    ↓
    Boundary Definition
    ↓
    Few-shot
    ↓
    Constraint
    ↓
    Query
让模型“按层理解任务”，而不是混读
```

##### Q8：为什么 Task Definition Card 能提升效果？它本质不还是 prompt 吗？

```markdown
👉 降低认知负担（Cognitive Compression）

我们做了三件事：
    把任务拆成固定 schema
    每个 schema 是短句 + 强约束
    避免自由描述
```

##### Q9：Token Budget Optimization，具体怎么控制 k∈[3,20]？

```markdown
我们不是固定 k，而是动态计算：
	k = min(max_score_samples + boundary_samples, budget / avg_token_length)

策略：
    budget = 模型最大 context - system - query
    boundary samples 优先占位
    剩余填高置信样本
```

##### Q10：你说你做了 Multi-Class GRPO，它和普通 GRPO 本质区别是什么？

```markdown
普通 GRPO 是在单样本维度做 group relative optimization：
    同一个 prompt
    多个 sampling trajectory
    做 reward 相对比较

👉 本质是：sample-level ranking learning
而我们提出的 Multi-Class GRPO 是：
👉 batch-level + multi-class constraint 的 GRPO 变体

核心变化有三点：
1️⃣ batch 内包含多个类别（不是单任务 group）
Batch = {class1, class2, class3 ...}

而不是：
Batch = same prompt multiple outputs
2️⃣ 引入跨类别边界竞争机制
我们不只是比较“同类输出好坏”，而是：
👉 跨类别之间也参与 reward competition

例如：
“投诉 vs 咨询”
“故障定位 vs 故障处理”
模型必须在 batch 内做结构性区分。

3️⃣ reward 控制“可更新经验数量”
我们设计了一个关键机制：
    reward ↑  → 可更新经验 ↓
    reward ↓  → 可更新经验 ↑

直觉是：
    高 reward：说明当前 batch 已经稳定 → 不需要大幅更新
    低 reward：说明边界不清 → 加大经验更新力度
```

##### Q11：你这个“reward 控制更新经验数量”具体是什么意思？听起来不像标准 RL

```markdown
确实不是标准 RL 设计，我们做的是：

👉 把 GRPO 从“优化目标函数”改成“控制学习节奏的机制”
具体实现是：
Step1：计算 batch reward score
	R_batch = mean(correct samples) - mean(error samples)
Step2：映射 update quota
	update_ratio = sigmoid(α - R_batch)
Step3：控制 replay buffer 更新量
	reward 高 → 只更新少量 hard samples
	reward 低 → 扩大 error samples 回灌

👉 本质是：
reward 不只是“优化信号”，还是“经验流动控制器”
```

##### Q12：你说 batch 内跨类别比较，但不同类别本身语义不同，怎么保证可比性？

```markdown
👉 比较“分类决策一致性”而不是语义相似性
跨类别比较的不是“内容”，而是：
👉 分类边界清晰度

例如：
    咨询 vs 投诉
    是一个 decision boundary problem
    
👉 Multi-Class GRPO 本质不是跨语义比较
👉 而是跨 decision boundary 的结构优化
```

##### Q13：你说 batch reward 控制经验更新，那如果 batch 全对或者全错怎么办？

```markdown
GRPO 只在“部分正确但存在边界模糊”时才有效
```

##### Q14：你这个机制听起来像“训练控制策略”，那它和 curriculum learning 有什么区别？

```markdown
确实有相似性，但有本质区别：
Curriculum Learning：
    预定义难度顺序
    静态 schedule

Multi-Class GRPO：
👉 动态结构学习
区别在三点：
1️⃣ difficulty 不是预设的
而是：
difficulty = model error + boundary instability

2️⃣ batch 内是竞争结构
不是“简单排序”，而是：
👉 class-level competition

3️⃣ reward 控制的是“经验流动”
不是 sample selection，而是：
experience update rate
```

##### Q15：那你这个方法稳定吗？会不会出现训练震荡？

```markdown
❌ 问题1：reward oscillation
batch reward 在：
高正确率 ↔ 突然崩溃
解决：
	EMA smoothing reward：
	由于 batch 内跨类别竞争，reward 波动比较大，所以我们引入 EMA 对 reward 进行平滑，减少单 batch 偶然性对训练的影响。
	clip update ratio：
	为了避免 reward 过低时导致策略更新过激，我们对 update ratio 做了 clip，限制每次 policy 更新的最大幅度。

❌ 问题2：class dominance
某些类别长期高 reward
解决：
	class-balanced replay buffer：
	由于不同类别难度不同，容易出现 class dominance，我们用 replay buffer 做类别均衡采样，防止模型偏向高频或容易类别。
```

Q16：



### **2023-全国先进计算技术创新大赛**

#### 简历版

```
全国先进计算技术创新大赛 | 国家级行业赛事 | 2023.12 | 算法核心成员 | 一等奖
•	依托曙光国产软硬件平台，完成ChatGLM2-6B大模型在DCU硬件与Slurm集群环境下的迁移部署；运用P-tuning v2微调策略优化模型，模型整体性能达赛道SOTA级别。
•	完成多场景数据集推理评测，量化分析模型推理指标偏差原因，沉淀国产硬件下大模型部署、调优实战经验。
```



#### 模拟面试

```markdown
**赛题：[2023D3-基于计算硬件平台的AI大模型迁移](http://www.ac-innovation.com/#/home)（一等奖）**
**项目描述：**基于曙光计算服务平台为基础，进行大模型迁移或研发。在研发过程中需基于曙光自研的软件栈（配套有成熟软件库、编译器等），也使用的其他相关软件实现。
**核心工作**：
1、基于国产ChatGLM2-6B中文大模型，完成了在Slurm调度系统+DCU硬件环境下的环境适配，并进行P-tuning v2调优，模型效果达到SOTA水平。
2、在推理方面，模型性能在验证数据集为0.27，在测试数据集为0.70，相比于SOTA（ChatGLM2-6B,0.50）模型存在一定差距，通过代码初步分析了相关原因。
```



#### 模拟问题

##### Q1：你们迁移过程中最大的技术难点是什么？

```markdown
CUDA生态向DCU生态迁移。
原始代码大量依赖：
    CUDA
    NCCL
    Apex
    DeepSpeed部分CUDA Kernel

而DCU使用的是：ROCm生态。
因此存在大量兼容性问题。
```

##### Q2：具体遇到了哪些兼容问题？

```markdown
1 ROCm版本兼容
2 CUDA API不兼容
	torch.cuda.amp --> torch.autocast
3 NCCL通信问题
	backend='nccl' --> backend='hccl'
4 自定义Kernel失效
	部分FlashAttention实现依赖CUDA Kernel。需要退化到普通Attention实现。
```

##### Q3：FlashAttention失效后性能下降多少？

```markdown
单卡推理吞吐下降约20%~30%。
无法利用FlashAttention的显存优化和Kernel融合能力。
```

##### Q4：为什么没有自己重写Kernel？

```markdown
比赛周期有限。
重写FlashAttention涉及：
    HIP编程
    Kernel优化
    Warp级同步
工作量较大。
```

##### Q5：为什么选择P-Tuning v2？

```markdown
P-Tuning v2对显存最友好。
```

##### Q6：P-Tuning v2原理是什么？

```markdown
不更新LLM参数。
只训练连续Prompt Embedding。

原输入：
	Question
变成：
	[P1][P2]...[Pn] Question
	
其中：P1~Pn 是可学习参数。
训练过程中仅更新Prompt参数。
模型主体冻结。
```

##### Q7：Prompt长度设置多少？

```
实验对比过：128
```



Q8：
