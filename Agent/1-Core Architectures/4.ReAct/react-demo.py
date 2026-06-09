# 导入依赖库
from openai import OpenAI

# ===================== 配置区（请修改这里） =====================
OPENAI_API_KEY = "ms-5e8ac3d3-3104-47b9-bf52-e96706571e23"
OPENAI_BASE_URL = "https://api-inference.modelscope.cn/v1"
# MODEL_NAME = "Qwen/Qwen3.5-35B-A3B"
MODEL_NAME = "Qwen/Qwen3.5-397B-A17B"
# ==============================================================

# 初始化OpenAI客户端
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


# ===================== ReAct工具库（极简版） =====================
# 定义2个最基础的工具，供大模型调用
def calculator(expression: str) -> str:
    """
    简易计算器工具：计算数学表达式
    :param expression: 数学表达式字符串（如 "10+5*2"）
    :return: 计算结果
    """
    try:
        # 安全计算简单表达式
        result = eval(expression)
        return f"计算结果：{result}"
    except:
        return "计算失败，请输入合法的数学表达式"


def mock_search(query: str) -> str:
    """
    模拟搜索工具：返回固定常识（极简版，真实场景可替换为搜索引擎API）
    :param query: 搜索问题
    :return: 搜索结果
    """
    knowledge_base = {
        "地球半径": "地球的平均半径约为6371千米",
        "光速": "真空中的光速约为30万公里/秒",
        "一年天数": "平年365天，闰年366天"
    }
    # 匹配知识库，无结果返回默认值
    return knowledge_base.get(query, "未找到相关信息")


# 工具映射表：模型调用工具时，匹配对应的函数
TOOL_MAP = {
    "calculator": calculator,
    "search": mock_search
}

# ===================== ReAct核心提示词 =====================
# 告诉大模型ReAct规则：如何思考、如何行动、如何结束
REACT_PROMPT = """
你是一个遵循ReAct框架的智能助手，严格按照以下格式执行任务：
1. 先思考问题需要什么操作（Thought）
2. 调用工具（Action）：只能调用 calculator(表达式) 或 search(关键词)
3. 得到工具结果后继续推理，直到得出答案
4. 最终答案用 Final Answer: 开头输出

可用工具：
- calculator(数学表达式)：用于数学计算
- search(关键词)：用于查询常识
"""


# ===================== ReAct主循环函数 =====================
def react_run(question: str) -> str:
    """
    执行ReAct推理+行动循环
    :param question: 用户问题
    :return: 最终答案
    """
    # 初始化对话历史：系统提示 + 用户问题
    messages = [
        {"role": "system", "content": REACT_PROMPT},
        {"role": "user", "content": question}
    ]

    # 最大循环次数（防止无限循环）
    max_round = 5

    for round in range(max_round):
        print(f"\n===== 第{round + 1}轮推理 =====")

        # 1. 调用OpenAI大模型，获取思考+行动
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
            stream=False,
        )
        # 提取模型回复
        model_reply = response.choices[0].message.content
        print(f"模型回复：\n{model_reply}")

        # 2. 判断是否已经得出最终答案
        if "Final Answer:" in model_reply:
            # 提取最终答案并返回
            final_answer = model_reply.split("Final Answer:")[-1].strip()
            return final_answer

        # 3. 解析模型的行动指令（极简解析：匹配工具名和参数）
        action = ""
        args = ""
        if "calculator(" in model_reply:
            action = "calculator"
            args = model_reply.split("calculator(")[-1].split(")")[0].strip()
        elif "search(" in model_reply:
            action = "search"
            args = model_reply.split("search(")[-1].split(")")[0].strip()
        else:
            # 无合法行动，强制结束
            return "无法识别行动指令"

        # 4. 执行工具，获取观察结果
        tool_result = TOOL_MAP[action](args)
        print(f"工具执行结果：{tool_result}")

        # 5. 将模型回复+工具结果加入对话历史，继续循环
        messages.append({"role": "assistant", "content": model_reply})
        messages.append({"role": "user", "content": f"Observation: {tool_result}"})

    # 超过最大循环次数
    return "已达到最大推理次数，未找到答案"


# ===================== 测试入口 =====================
if __name__ == "__main__":
    # 测试问题1：纯计算
    # user_question = "10加5乘以2等于多少？"
    # 测试问题2：纯搜索
    # user_question = "地球的半径是多少？"
    # 测试问题3：组合问题
    user_question = "地球半径是6371千米，换算成米是多少？"

    print(f"用户问题：{user_question}")
    answer = react_run(user_question)
    print(f"\n===== 最终结果 =====")
    print(answer)
