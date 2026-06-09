from openai import OpenAI

# ===================== 配置区（请修改这里） =====================
OPENAI_API_KEY = "ms-5e8ac3d3-3104-47b9-bf52-e96706571e23"
OPENAI_BASE_URL = "https://api-inference.modelscope.cn/v1"
MODEL_NAME = "Qwen/Qwen3.5-397B-A17B"

# 2. 初始化OpenAI客户端
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL,)

# ------------------- 核心：CoT思维链提示词 -------------------
# 这是最简单的CoT实现：仅通过指令让模型「一步步思考」，无需额外示例
# Zero-Shot CoT 核心：强制模型先推理、再给答案
cot_prompt = """
问题：小明有5个苹果，给了小红2个，又买了3个，现在有多少个苹果？
要求：请**一步步思考**，写出详细的推理过程，最后给出最终答案。
"""


# -----------------------------------------------------------

# 3. 定义调用大模型的函数
def cot_simple_call():
    """最简单的CoT调用函数：让大模型分步思考并回答"""
    try:
        # 调用OpenAI Chat API
        response = client.chat.completions.create(
            model=MODEL_NAME,  # 用性价比最高的模型，也可以换gpt-4
            messages=[
                {"role": "user", "content": cot_prompt}  # 传入CoT提示词
            ],
            temperature=0.7  # 随机性：0=最确定，1=最发散
        )

        # 提取并返回模型的回答
        answer = response.choices[0].message.content
        return answer

    except Exception as e:
        # 异常处理（网络错误、API密钥错误等）
        return f"调用失败：{str(e)}"


# 4. 主程序：执行调用并打印结果
if __name__ == "__main__":
    print("=== 最简单CoT调用结果 ===")
    result = cot_simple_call()
    print(result)