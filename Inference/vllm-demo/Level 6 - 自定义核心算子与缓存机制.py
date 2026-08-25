'''

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