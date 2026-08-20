# Qwen 普通 SFT、Agent SFT 与 DPO

训练入口是 `main.py`，训练模式在 `config.py` 中切换。

```python
TRAINING_ALGORITHM = 'sft'  # 普通 SFT 或 Agent SFT
TRAINING_ALGORITHM = 'dpo'  # 偏好优化
```

## 普通 SFT

```python
TRAINING_MODE = 'sft'
DATA_PATH = r'../data/sft_data_mixed_single.csv'
```

当前 `SFTTrainer` 使用教学用的显式训练路径：项目直接按
`(softmax(logits) - one_hot(label)) / 有效 token 数` 计算 loss 对 logits
的梯度，并从 logits 启动 Transformer 计算图的向量-雅可比积，不调用
`loss.backward()`。参数更新由项目内的 `ManualAdamW` 完成，包括一阶/二阶
动量、偏置修正、解耦权重衰减、FP32 master weights、动态 loss scaling
和全局梯度裁剪；SFT 路径不调用 `torch.optim.AdamW`。Transformer 各算子的
局部反向仍由 PyTorch autograd 执行，以兼容任意 Hugging Face CausalLM。

兼容原有 CSV：

```csv
history,q,a
[],你好,你好！有什么可以帮助你的？
```

也可以使用包含 `messages` 的 JSON/JSONL/CSV。普通 SFT 只对最后一条
`assistant` 消息计算 loss，行为与原有代码一致。

## Agent SFT

```python
TRAINING_MODE = 'agent_sft'
DATA_PATH = r'./examples/agent_sft_example.jsonl'
```

每条 JSONL 数据包含完整轨迹 `messages`，以及可选的 OpenAI function tools。
Agent SFT 会监督轨迹中的每一条 `assistant` 消息，包括工具调用和最终回答；
`system`、`user`、`tool` 返回值不会参与 loss。

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "查询城市天气",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }
  ],
  "messages": [
    {"role": "system", "content": "你是一个可以调用工具的助手。"},
    {"role": "user", "content": "北京天气怎么样？"},
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "id": "call_1",
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": {"city": "北京"}
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_1",
      "name": "get_weather",
      "content": "{\"temperature\": \"28°C\", \"condition\": \"晴\"}"
    },
    {"role": "assistant", "content": "北京当前晴，气温约 28°C。"}
  ]
}
```

`function.arguments` 既可以是 JSON 对象，也可以是 JSON 字符串；加载时会统一
转换为 Qwen 聊天模板需要的对象。旧式 `assistant.function_call` 和
`role=function` 也会自动转换。

当前模型必须提供支持 `tools`、`assistant.tool_calls` 和 `tool` 消息的
`chat_template`。项目默认的 Qwen2.5-Instruct 模板满足这个要求。

这里实现的是 **Agent 轨迹监督训练**。训练完成后的在线工具注册、工具调用解析、
实际执行工具和把 observation 回填给模型，属于推理侧 Agent loop，需要在部署代码中
另外实现。

## DPO

DPO 默认从已经完成 SFT 的 checkpoint 继续训练，并使用同一个 SFT checkpoint
作为冻结 reference policy：

```python
TRAINING_ALGORITHM = 'dpo'
DATA_PATH = r'data/rl/train_dpo.jsonl'

DPO_POLICY_PATH = './output/final'
DPO_REFERENCE_PATH = './output/final'
DPO_BETA = 0.1
DPO_LABEL_SMOOTHING = 0.0
DPO_MAX_LENGTH = 2048
DPO_NORMALIZE_LOGP_BY_LENGTH = False
```

偏好数据支持 CSV、JSON 和 JSONL，每条数据固定包含三个未模板化文本字段：

```json
{
  "prompt": "北京今天的天气怎么样？",
  "chosen": "北京今天晴，气温约 28°C。",
  "rejected": "我不知道，而且无法提供任何帮助。"
}
```

代码会将 `prompt` 渲染为 system/user/assistant generation prompt，只对
`chosen` 和 `rejected` completion 计算序列 log-prob。Agent DPO 也使用相同
三字段格式，把 `<tool_call>`、工具结果与最终回答提前序列化进 chosen/rejected；
DPO 训练过程不会解析或执行工具。可参考 `examples/dpo_example.jsonl`。

DPO checkpoint 保存到 `output/dpo-final`、`output/dpo-best` 和
`output/dpo-checkpoint-*`，其中只包含 policy；`dpo_config.json` 会记录 reference
路径和 DPO 超参数。若 policy 是 LoRA adapter，代码会自动加载其基础模型再继续
训练 adapter；设置 `USE_LORA=False` 时会先合并已有 adapter，再解冻 policy 做
全参数 DPO。reference 始终保持冻结。

## 运行

```powershell
python main.py
```
## Calibra 模块化结构（新版）

项目已迁移为配置驱动的包结构：`configs/` 存放 YAML/JSON 实验配置，`calibra/data` 负责统一数据加载与 SFT/Agent-SFT/DPO formatter，`calibra/models` 提供模型 registry 和 LoRA loader，`calibra/trainers` 提供 Base/SFT/DPO 训练器，并预留 PPO、GRPO、Agent-RL 接口；`calibra/envs`、`tools`、`rewards`、`rollout` 分别用于 Agent 交互、工具注册、奖励组合和采样后端。

重构前的单文件实现已迁移到 [`legacy/`](legacy)，仅作为行为对照和回滚参考，不再是推荐运行路径。

推荐使用统一入口：

```powershell
pip install -e ".[train,dev]"
python scripts/train.py --config configs/sft_config.yaml
python scripts/train.py --config configs/dpo_config.yaml
python scripts/evaluate.py --config configs/sft_config.yaml --checkpoint output/sft/final
```

## C-Eval / CMMLU 评测

`scripts/evaluate_benchmarks.py` 支持 C-Eval 和 CMMLU 的选择题评测。脚本对
`A/B/C/D` 四个选项分别计算条件 log 概率，选择得分最高者，并输出总体准确率和
分科目准确率。支持官方常见的 `val/*.csv` 布局，也支持 CSV、JSON、JSONL 文件；
文件需要包含 `question,A,B,C,D,answer` 字段（`test` 数据没有 `answer` 时仍可
生成预测）。

```powershell
python scripts/evaluate_benchmarks.py `
  --config configs/sft_config.yaml --checkpoint output/sft/final `
  --benchmark ceval --data-dir data/ceval --split val `
  --output output/ceval_val.json --predictions output/ceval_predictions.jsonl

python scripts/evaluate_benchmarks.py `
  --config configs/sft_config.yaml --benchmark cmmlu --data-dir data/cmmlu

# data/benchmarks/ceval 和 data/benchmarks/cmmlu 同时存在时
python scripts/evaluate_benchmarks.py `
  --config configs/sft_config.yaml --benchmark both --data-dir data/benchmarks
```

可用 `--subjects computer` 限定科目，`--max-length` 覆盖配置中的上下文长度。
运行前安装训练依赖：`pip install -e ".[train,dev]"`。

支持 `training.num_epochs=1` 形式的命令行覆盖。根目录 `main.py` 保留为兼容启动器；原有根目录模块仍保留，现有 checkpoint 和旧脚本无需立即改动。
