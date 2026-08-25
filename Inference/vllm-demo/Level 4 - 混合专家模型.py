'''

Level 4: 混合专家模型（MoE - Mixture-of-Experts）
适用场景：
包含 Router 和多个 Expert 的 MoE 架构（如 Mixtral、DeepSeek-V2/V3、Qwen-MoE 等）。
适配工作：
适配 MoE 算子：调用 vLLM 的 FusedMoE 算子，实现门控网络（Gate/Router）计算（如 Top-K routing、Softmax/Sigmoid 归一化）。
特殊结构处理：如果包含共享专家（Shared Experts），需要分别处理 Shared Expert 和 Routed Expert 的输出融合。
分布式与并行切分：处理 TP（Tensor Parallelism）和 EP（Expert Parallelism）的权重切分与映射，MoE 权重的重排和打包加载相对繁琐。
难度：⭐⭐⭐⭐（涉及 MoE 显存分布、专家并行与高效算子对接）

'''