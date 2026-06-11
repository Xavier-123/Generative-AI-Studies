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
- **Braintrust / Maxim AI**
  - **特点**：企业级端到端评估和仿真平台，支持将生产环境中的真实数据回流转化为测试集，实现闭环评测。



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

## 工作

### **马上消费金融有限公司**（2025.10-至今）

#### 项目一：[对话数据标注自进化Agent系统](https://chatgpt.com/c/6a2a4aff-22ac-83ea-bdf4-3631c4dbf48d)

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



**面试时的口语化讲法：**
 我做的是一个面向对话语料标注的自进化智能体系统。这个系统的目标是解决人工标注贵、慢，而且遇到新业务和新表达时很难快速适应的问题。整体上分成三层：先用通用粗分类做初始标注，再根据易错类别做自进化优化，最后通过细分类模块对高混淆样本逐级校验。系统还会自动优化提示词，让大模型在不训练参数的情况下持续提升标注效果。



##### 🎯 面试问题预测

###### 🔧 核心技术选型

**Q1：为什么选择"无训练"方案而不是直接微调模型？**

> **回答要点**：微调成本高（算力、数据标注）、周期长、难以快速迭代；无训练方案通过推理阶段积累经验，零算力成本、实时可更新，适合资源受限的业务场景。可对比 RAG（检索外部知识）与本方案（自生成内部经验）的差异。

**Q2：为什么选择 PromptWizard 和 TextGrad 两套框架？它们的核心差异是什么？**

> **回答要点**：PromptWizard 是微软提出的示例驱动搜索方案，适合有标注数据的分类/问答场景；TextGrad 将"文本梯度"概念引入优化，适合需要多跳推理的复杂任务。两者互补，统一 Config 层降低切换成本。

------

###### 🔥 核心挑战与解决方案

**Q3：经验库如何避免"噪声经验"污染后续推理？**

> **回答要点**：通过两阶段机制：① 单 Batch 内先生成 candidate_experiences 而非直接更新，避免单条坏样本直接写入；② Epoch 结束后的压缩阶段通过 LLM 二次审查合并冗余、淘汰低质量条目。可进一步讨论设置置信度阈值等改进方向。

**Q4：多线程并发调用 LLM API 时，如何处理限速（Rate Limit）问题？**

> **回答要点**：通过 `max_workers` 参数控制并发数，根据 API 配额动态调整；可引申讨论指数退避重试策略、Token 用量监控、批量请求 vs 并发请求的权衡。

**Q5：经验的"增删改合并"操作由 LLM 自身决策，如何保证 JSON 格式输出的鲁棒性？**

> **回答要点**：代码中使用 `split('```json')[-1].split('```')[0]` 做格式抽取，外层 try/except 兜底，失败时跳过而非崩溃（continue）。可进一步讨论引入 Pydantic 校验、Few-shot 格式示例强化输出稳定性。

------

###### 💡 技术难点深挖

**Q6：GRPO 算法的核心思想是什么？你的 Multi-Class GRPO 与原版 GRPO 有何本质区别？**

> **回答要点**：原版 GRPO 通过分组采样计算相对奖励并反向传播更新模型权重；本方案"借用"分组采样的思路，但奖励信号转化为文本经验而非梯度，用 Prompt 上下文替代参数更新，实现 inference-time learning。



**Q7：如何评估先验经验的质量与有效性？**

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



**Q8：新沉淀的经验与既有经验产生逻辑冲突，如何解决？**

> 1. **语义冲突检测（Conflict Detection）**：
>    - **方法**：利用专有的“审计 Agent”或符号化规则，对新生成的 JSON 经验与既有经验进行两两比对。
>    - **评估案例**：若经验 A 指出“‘退款’关键词属于‘售后服务’”，而新经验 B 指出“‘退款’且伴随抱怨语时应归入‘投诉’”，系统需评估两者的覆盖边界。
>    - **解决机制**：若判定为冲突，则触发“规则融合（Merge）”或“条件细化（Refinement）”，通过补充前置条件（如 Context-specific constraints）来消解冲突。
> 2. **语义冗余度与合并（Redundancy & Deduplication）**：
>    - **方法**：计算新经验与旧经验在向量空间（Embedding）中的余弦相似度。
>    - **评估指标**：当相似度超过设定阈值（如 0.85）时，判定为表述冗余。系统将自动调用大模型进行“规则归并”，用更具泛化性的表述合并这两条经验。



**Q9：如何提升跨样本的泛化能力？**

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



**Q8：`temperature=1.0` 的设计意图是什么？**

> **回答要点**：高温度增加输出多样性，保证 N 条采样轨迹覆盖不同推理路径（正确与错误），为经验提取提供更丰富的对比素材。若 temperature 过低，N 条输出高度相似，经验提取价值有限。



------

###### 📊 业务视角

**Q9：这套系统如何从实验原型落地到生产环境？你会做哪些工程化改造？**

> **回答要点**：① 经验库改为向量数据库（如 Chroma，代码中已有依赖）实现语义检索而非全量注入；② 增加经验版本管理与回滚机制；③ 异步任务队列处理大批量数据；④ 监控 API 调用成本与经验库增长曲线。



**Q10：如果业务方反馈模型在新数据上效果下降，你会如何排查？**

> **回答要点**：① 检查经验库是否存在过拟合训练分布的噪声经验；② 对比加/不加经验的推理结果，定位经验质量问题；③ 检查数据分布漂移（新数据与训练数据的 prompt 模式差异）；④ 必要时清空经验库重新从新数据中提取。

------

#### 项目二：零幻觉推理研究

一个基于 LangGraph 的 Agentic RAG 原型项目，用来演示如何通过查询处理、多源检索、证据精炼和反思推理来降低问答幻觉，并输出可追溯的证据。

##### 项目特点

- 基于 `LangGraph` 搭建工作流图，按状态驱动模块流转。
- 包含完整的 Agentic RAG 核心链路：
  - Query Processor
  - Retriever
  - Evidence Refiner
  - Reflective Reasoner
  - Orchestrator
- 提供简单的评测示例，便于验证答案与证据质量。
- 代码结构清晰，适合继续扩展为真实检索系统。





### **浪潮通信**（2021.07-2025.10）

#### **项目名称：九天全能小助手**

**技术栈：**Python/PyTorch/LLM/RAG/Prompt/Docker/爬虫/数据库

**项目描述：**旨在制作一款私有的、安全性高的智能问答类业务助手，辅助完成日常工作，在这个项目中，我主要负责后端LLM应用架构的设计与实现。主要包括（**行业研究助手、会议助手、视频助手、数据分析助手、Agent助手**、营销文案助手、招投标助手、算网支撑助手、财务稽核助手九大模块）。

**核心工作：**

1.行业助手：主要工作是采用搜索引擎采取爬虫技术收集行业数据，基于知识库和大模型，使模型回答更专业、更准确；

2.会议助手：部署语音识别和语音合成模型，识别语音内容，并生成会议纪要、总结、发言稿等信息；

3.视频助手：与会议助手相似；

4.数据分析助手：针对sql和表格数据，分析及可视化需求，提供上传 excel 数据，通过指令使其图表自动绘制，以及配置连接数据库，可完成语言指令的相关sql操作；

5.Agent助手：通过后端开发者自己制作agent，使大模型完成一定程度的自主行动（可完成一些特定的下游任务）；

 

#### **项目名称：AI训推平台**

**技术栈：Python/CUDA/CANN/LLMs**

**项目描述：**简化AI大模型训练流程，提供从数据标注、模型训练到服务部署的一站式解决方案。

**核心工作：**

1、开发数据标注模块，根据需求生成问答对；开发数据蒸馏模块，从功能强大的教师模型蒸馏具有思维数据和专业性数据；

2、AI训练模块开发。提供PT、SFT、DPO、蒸馏训练等多种训练算法，以及full、lora和qlora多种微调算法；以及协助具体项目的大模型微调训练落地工作，比如通信类大模型微调等；

3、推理服务优化。集成多种推理引擎，支持一键部署微调模型，测试效果。

4、API与工具链开发。封装 FastAPI 提供RESTful接口，支持用户上传数据、触发训练任务；简化用户代码操作，通过docker或页面完成相关操作。



#### **项目名称：Omega智能体平台**

**技术栈****：**Python/Agent/RAGs/LLMs/milvus/语音

**项目描述****：**该项目面向企业级应用的自主智能体开发平台，支持大模型任务自动化与复杂决策。核心目标是利用大模型技术实现复杂任务的自动化和智能决策。我主要负责其中的两大核心模块：LLM能力增强和RAG评估体系的构建。

**核心工作****：**

1、LLM能力增强。集成 RAG（检索增强生成），使用多种检索技术、推理优化技术、生成与融合策略等技术，减少模型幻觉，提高回答准确率，提升用户满意度。

2、RAG评估功能开发。集成RAG性能评估功能，通过调用已有RAG系统获取‘回答’和‘上下文索引’，然后通过忠实性、答案相关性、上下文精度、上下文召回率、上下文实体召回率等RAG的评估算法进行评判RAG性能好坏，降低人工评估工作。

 

#### **项目名称：多平台大模型应用适配**

**技术栈：**Python/CUDA/CANN /LLM/Docker

**项目描述：**基于GPU/NPU/CPU多平台对大模型适配

**核心工作：** 

1、针对NPU，在国产昇腾910B平台的推理适配（mindie），以及在昇腾910B的微调开发工作；

2、开发管理平台，统一管理：语言大模型、多模态大模型、Embedding、Reranker、语音模型、视频模型等，减少模型管理混乱；

3、针对不同平台，不同系统架构，基于Docker开发大模型一键部署功能，减少部署成本，以及后期维护成本。

 



## 比赛

### **全国先进计算技术创新大赛**

**赛题：[2023D3-基于计算硬件平台的AI大模型迁移]()（一等奖）**

**项目描述：**基于曙光计算服务平台为基础，进行大模型迁移或研发。在研发过程中需基于曙光自研的软件栈（配套有成熟软件库、编译器等），也使用的其他相关软件实现。

**核心工作**：

1、基于国产ChatGLM2-6B中文大模型，完成了在Slurm调度系统+DCU硬件环境下的环境适配，并进行P-tuning v2调优，模型效果达到SOTA水平。

2、在推理方面，模型性能在验证数据集为0.27，在测试数据集为0.70，相比于SOTA（ChatGLM2-6B,0.50）模型存在一定差距，通过代码初步分析了相关原因。



### FlagOS开放计算全球大赛

**赛题：[长上下文场景中大模型自动数据标注](https://flagos.io/RaceDetail?id=296fmsd8&lang=cn)（三等奖）**

**项目描述：**本次挑战赛旨在**探索大语言模型在超长上下文（如数万至数十万token）范式下的数据标注能力提升方法**，重点评估模型在复杂、长序列标注场景中的稳定性、泛化性与实用性。

**核心工作**：

1. 每类任务用自己的“指令卡 + few-shot 示例 + 输出约束”
2. 用**规则式**挑选few-shot 示例 

算法：

1. Coverage-based Demonstration Selection（覆盖式示例选择）



数据集1：最小绝对差

```
方法1：Curriculum ICL
	排序后求差
方法2：Self-Consistency
	解法A：排序后求差
	解法B：暴力比较
	最终输出一致答案
```

数据集2：名词/动词计数

```
方法1：Coverage-based ICL
	不是选最相似样例，而是选覆盖不同语法现象的样例。
方法2：Iterative ICL
	第一轮：识别词性
	第二轮：统计数量
```

数据集3：Collatz变换

```
方法1：Many-shot ICL
	构造大量样例
方法2：Self-Consistency
	逐元素推理，整体推理，规则解释推理
```

数据集4：字符串拼接

```
方法1：Many-shot ICL
方法2：Curriculum ICL
```

数据集5：Sad/Not Sad

```
方法1：Coverage-based Demonstration Selection
方法2：Iterative Demonstration Selection
```



数据集6：文体匹配

```
方法1：Coverage-based ICL
方法2：Curriculum ICL
```

数据集7：Clue QA

```
方法1：Many-shot ICL
方法2：Self-Consistency
```

数据集8：Triton代码生成

```
方法1：Coverage-based ICL
方法2：Curriculum Demonstration Selection
```







**技术难点**：

1. **长上下文的高效利用挑战**：32K 上下文窗口容量有限，而部分任务（如 Task7、Task5）训练示例数量庞大（数千个），如何在窗口内筛选 “高价值示例”（提升标注准确性），同时避免信息冗余（浪费 token 资源）成为关键；
2. **任务多样性的适配挑战**：分类、生成、代码四大类任务的标注逻辑差异显著 —— 分类任务需明确标签边界，生成任务需保证输出完整性，代码任务需满足语法与数值正确性，需设计统一且可扩展的标注框架；
3. **标注一致性与准确性平衡挑战**：无人工干预场景下，模型易受输入表述、示例顺序等因素影响导致输出波动，需通过规则约束、示例筛选等方式强化一致性；同时部分任务（如 Task6）存在 “主题匹配” 与 “语义相似度” 的易混淆点，需通过指令设计降低误判率；
4. **特殊任务的正确性验证挑战**：Task8（代码生成）需同时满足 “语法正确”“结构合规”“数值一致” 三大要求，传统单轮标注无法保证正确性，需设计闭环验证机制。
5. 在超长上下文场景下，如何设计有效的模型指令与提示策略，引导 LLM 稳定、高质量地完成数据标注任务？
6. 当可用标注示例数量显著超过模型上下文容量时，如何为待标注数据构造信息密集、结构合理的超长上下文输入？
7. 在自动多轮对话或持续交互场景中，如何高效利用超长上下文，实现一致性与可扩展性兼顾的数据标注？
8. 禁止使用Qwen3-4B以外的模型



长上下文场景中LLM自动数据标注挑战赛基于**Qwen3-4B**大语言模型，采用上下文（In-context Learning, ICL）范式开展**自动化数据标注**任务研究。参赛团队必须使用组委会统一提供的数据集，围绕超长上下文场景设计有效的 ICL 标注方案，并在统一评测集上完成推理与评测。组委会将依据标准化评测结果，对参赛方案进行综合评估并确定最终排名。

赛题重点聚焦以下三个关键科学与工程问题：

1. 在超长上下文场景下，如何设计有效的模型指令与提示策略，引导 LLM 稳定、高质量地完成数据标注任务？
2. 当可用标注示例数量显著超过模型上下文容量时，如何为待标注数据构造信息密集、结构合理的超长上下文输入？
3. 在自动多轮对话或持续交互场景中，如何高效利用超长上下文，实现一致性与可扩展性兼顾的数据标注？

测试数据：

任务1：在本任务中，你将获得一个整数列表。你需要找出列表中任意两个整数之间的最小绝对差值。绝对差值指一个整数减去另一个整数后的绝对值。输出要求为一个整数，即所能得到的最小绝对差值。

任务2：在本任务中，你需要统计给定句子中的名词 / 动词数量。

任务3：在本任务中，你将得到一个整数列表。对于列表中的每一个元素：如果该元素是**偶数**，则将其除以二；如果该元素是**奇数**，则先乘以三再加一。输出应为按照上述逻辑处理输入列表后得到的整数列表。

任务4：在本任务中，你将获得一个字符串列表，需要将它们拼接合并在一起。

任务5：在本任务中，你会收到一条推文。你需要判断推文作者是否处于悲伤情绪。根据判断将样本标注为 **“悲伤（Sad）”或“不悲伤（Not sad）”。你可以参考话题标签和表情符号辅助判断，但不可仅依靠它们，同时也必须重点关注推文的文本内容。

任务6：在本任务中，你将得到两个句子（句子1和句子2）以及它们所属的文体类别。你的任务是判断这两个句子是否属于同一文体类别。分别用 **Y（是）** 和 **N（否）** 给出答案。

可用文体类别包括：**面对面交谈、政务文稿、公益信函、9·11事件、专栏文稿、电话对话、旅行指南、语言学实录、牛津大学出版物、虚构文学**。

各类别释义：
- **面对面交谈**：与日常会话、人物对话相关的文本
- **政务文稿**：政府官方网站发布的各类公开信息
- **公益信函**：用于公益募捐相关的各类书面文稿
- **9·11事件**：与9·11恐怖袭击相关的信息内容
- **牛津大学出版物**：包含纺织行业、儿童发展相关的非虚构作品
- **专栏文稿**：《石板》杂志中涉及各类文化主题的内容
- **电话对话**：电话沟通形式的对话文本
- **旅行指南**：旅游攻略、旅行指南类相关信息
- **语言学实录**：与语言学相关的简短短文帖子
- **虚构文学**：《暗藏杀机》等通俗虚构类文学作品

### 补充说明
1. 原文 `oup` 为 **Oxford University Press** 缩写，译作**牛津大学出版物**；
2. `slate` 特指美国知名文化杂志 *Slate*，统一译为**《石板》杂志**；
3. `verbatim` 结合语境译为**语言学实录**，贴合短文帖子的属性。

任务7：你将收到一条趣味知识线索及其所属类别。请给出**符合该类别**且**与线索描述匹配**的最佳答案。为统一格式，答案需全部使用**小写字母**。

任务8：使用Triton实现自定义算法或函数，并**保证块掩码与步长处理逻辑正确**，保障内存安全。