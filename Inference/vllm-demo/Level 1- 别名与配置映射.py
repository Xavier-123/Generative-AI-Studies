'''

Level 1: 别名与配置映射（零代码 / 配置级适配）
适用场景：
模型本质上是已有支持架构（如 LLaMA、Mistral、Qwen），只是在微调或开源发布时修改了 HuggingFace config.json 中的 architectures 或 model_type 名称。
适配工作：
直接加载：在 vllm/model_executor/models/registry.py 中将新的 architectures 名字映射到已有的模型实现类（如 LlamaForCausalLM）。
或者直接在本地修改模型的 config.json，把 architectures 改为 vLLM 已支持的标准名称。
难度：⭐（几行代码或修改配置即可）

'''

import os
from vllm import LLM, SamplingParams
from vllm.model_executor.models import ModelRegistry

# ==========================================
# 步骤 1: 注册自定义架构名称到现有的模型类
# ==========================================
# 假设你的微调模型在 config.json 里的 architectures 字段写的是 "MyCustomLlamaForCausalLM"
# 或者是某种换皮的 Llama 衍生模型名
CUSTOM_ARCH_NAME = "MyCustomLlamaForCausalLM"

# 将其映射到 vLLM 内置的 LlamaForCausalLM
# 注意：第二个参数传入 vLLM 内部已经实现的架构类名字符串
ModelRegistry.register_model(CUSTOM_ARCH_NAME, "LlamaForCausalLM")

print(f"✅ 成功将架构别名 [{CUSTOM_ARCH_NAME}] 映射到 [LlamaForCausalLM]！")

# ==========================================
# 步骤 2: 正常加载并使用 vLLM 进行推理
# ==========================================
model_path = "/path/to/your/custom_llama_model"  # 替换为你实际的模型本地路径或 HF ID

# 验证加载
llm = LLM(
    model=model_path,
    trust_remote_code=True,  # 如果模型带有自定义代码建议开启
    gpu_memory_utilization=0.8,
)

sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=128)

prompts = [
    "Hello, introduce yourself briefly:",
    "How does vLLM achieve high throughput?",
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated_text}")