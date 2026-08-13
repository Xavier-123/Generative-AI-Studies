# Qwen 普通 SFT 与 Agent SFT

训练入口是 `main.py`，训练模式在 `config.py` 中切换。

## 普通 SFT

```python
TRAINING_MODE = 'sft'
DATA_PATH = r'../data/sft_data_mixed_single.csv'
```

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

## 运行

```powershell
python main.py
```
