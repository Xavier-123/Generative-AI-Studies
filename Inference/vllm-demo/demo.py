'''

Level 1: 别名与配置映射（零代码 / 配置级适配）
适用场景：
模型本质上是已有支持架构（如 LLaMA、Mistral、Qwen），只是在微调或开源发布时修改了 HuggingFace config.json 中的 architectures 或 model_type 名称。
适配工作：
直接加载：在 vllm/model_executor/models/registry.py 中将新的 architectures 名字映射到已有的模型实现类（如 LlamaForCausalLM）。
或者直接在本地修改模型的 config.json，把 architectures 改为 vLLM 已支持的标准名称。
难度：⭐（几行代码或修改配置即可）

Level 2: 现有架构的局部变体（微调已有模型类）
适用场景：
模型骨干与已有模型高度重合，但有少量的细节差异（如：换了特殊的 RoPE 缩放公式、增加了 QK-Norm、改变了 LayerNorm/RMSNorm 的偏置项、Tie-Word-Embeddings 规则不同等）。
适配工作：
在已有的模型文件（如 llama.py）上做小幅修改，或者继承现有类。
在 __init__ 中增加对新超参数的解析与分支逻辑。
适配权重的命名映射（Weight Name Mapping），确保 HuggingFace 权重能正确加载到对应的张量并行层中。
难度：⭐⭐（熟悉模型结构和权重映射即可）

Level 3: 全新标准 Transformer 架构（复用现有并行算子）
适用场景：
常规的 Decoder-only 或 Encoder-Decoder Transformer，但层间连接方式、注意力与 MLP 的排列或前后 Norm 结构是全新的（如早期新增 ChatGLM、MiniCPM、Gemma 等）。
适配工作：
组装网络：使用 vLLM 封装好的高性能并行层（如 ColumnParallelLinear、RowParallelLinear、VocabParallelEmbedding、Attention 等）重新编写 DecoderLayer 和 Model 整体结构。
编写权重加载：实现 load_weights 函数，将 HuggingFace 导出的权重进行切分（Tensor Parallelism）并加载到对应 GPU。
模型注册：在 Model Registry 中注册新模型，并编写对应的测试用例。
难度：⭐⭐⭐（需要清晰理解张量并行切分逻辑与模型 forward 过程）

Level 4: 混合专家模型（MoE - Mixture-of-Experts）
适用场景：
包含 Router 和多个 Expert 的 MoE 架构（如 Mixtral、DeepSeek-V2/V3、Qwen-MoE 等）。
适配工作：
适配 MoE 算子：调用 vLLM 的 FusedMoE 算子，实现门控网络（Gate/Router）计算（如 Top-K routing、Softmax/Sigmoid 归一化）。
特殊结构处理：如果包含共享专家（Shared Experts），需要分别处理 Shared Expert 和 Routed Expert 的输出融合。
分布式与并行切分：处理 TP（Tensor Parallelism）和 EP（Expert Parallelism）的权重切分与映射，MoE 权重的重排和打包加载相对繁琐。
难度：⭐⭐⭐⭐（涉及 MoE 显存分布、专家并行与高效算子对接）

Level 5: 多模态大模型（VLM / Multi-Modal）
适用场景：
图文、音视频多模态模型（如 LLaVA、Qwen2-VL、InternVL、Phi-3-Vision）。
适配工作：
多模态接口继承：实现 SupportsMultiModal 和 SupportsPP 等接口。
多模态编码器（Encoder + Projector）：实现图像/音频特征提取，并将其映射为与文本维度一致的 Embedding。
Prompt Token 占位与展开：在 input processor 中将 <image> 等占位符展开为多模态特征，管理跨模态的位置编码。
Dummy Input 与 Profiling：配合 vLLM 的显存预估机制，提供多模态虚拟输入（Dummy inputs），确保 PagedAttention 的显存分配正常。
难度：⭐⭐⭐⭐（涉及输入预处理流水线改造与多模态数据流控制）

Level 6: 非传统 Transformer / 需自定义核心算子与缓存机制
适用场景：
新型注意力：如 DeepSeek 的 MLA（多头潜在注意力）需要特定的 KV 解压缩融合算子，或定制的稀疏/滑动窗口注意力。
非 Transformer 架构：状态空间模型（SSM，如 Mamba/Jamba）、线性注意力（RWKV）、循环神经网络类（RecurrentGemma）等。
适配工作：
手写底层算子：需要用 Triton 或 CUDA C++ 实现定制的高性能 Forward/Kernel。
改造缓存与调度机制：非标准模型通常无法直接使用传统的 KV Cache（例如 Mamba 需要维护 State Cache/SSM State），需要扩展或重构 vLLM 的 Cache Engine、Block Allocator 以及 Scheduler。
CUDA Graph 兼容：确保新算子和状态更新逻辑能被 CUDA Graph 正确捕获并加速推理。
难度：⭐⭐⭐⭐⭐（需要深入理解 CUDA/Triton 算子开发及 vLLM 核心显存调度架构）

'''