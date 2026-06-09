import os
from openai import OpenAI
import dotenv

# ===================== 配置区（请修改这里） =====================
dotenv.load_dotenv("../../.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")

# 初始化OpenAI客户端
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


# ===================== 基础工具函数 =====================
def call_llm(prompt: str) -> str:
    """
    封装OpenAI调用：传入提示词，返回模型的纯文本回答
    :param prompt: 给大模型的指令/问题
    :return: 模型回答字符串
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        extra_body={
            "enable_thinking": False,
        }
    )
    # 提取并返回模型的回答内容
    return response.choices[0].message.content.strip()


# ===================== 核心：Self-Ask（自问自答）函数 =====================
def self_ask(main_question: str) -> str:
    """
    Self-Ask 最简实现：
    1. 第一步：模型【自问】——判断是否需要拆分子问题，并生成子问题
    2. 第二步：模型【自答】——回答生成的子问题
    3. 第三步：整合信息，输出【最终答案】
    :param main_question: 用户的原始问题
    :return: 最终答案
    """
    # ---------------------- 1. 第一步：让模型自己提出子问题（自问） ----------------------
    ask_prompt = f"""
    问题：{main_question}
    规则：
    1. 如果你需要分步推理才能回答，请输出：需要子问题：[你的子问题]
    2. 如果你可以直接回答，请输出：不需要子问题
    3. 禁止输出多余文字！
    """
    # 获取模型的自问结果
    ask_result = call_llm(ask_prompt)

    # ---------------------- 2. 第二步：让模型回答子问题（自答） ----------------------
    sub_question, sub_answer = "", ""
    if "需要子问题：" in ask_result:
        # 提取模型生成的子问题
        sub_question = ask_result.split("需要子问题：")[1].strip()
        print(f"[模型自问] {sub_question}")

        # 让模型回答这个子问题
        sub_answer = call_llm(f"请简洁回答：{sub_question}")
        print(f"[模型自答] {sub_answer}")

    # ---------------------- 3. 第三步：整合信息，生成最终答案 ----------------------
    final_prompt = f"""
    原始问题：{main_question}
    子问题：{sub_question}
    子问题答案：{sub_answer}
    请根据以上信息，直接给出原始问题的最终答案。
    """
    final_answer = call_llm(final_prompt)
    return final_answer


# ===================== 测试运行 =====================
if __name__ == "__main__":
    # 测试用的问题（选了一个需要分步计算的问题，完美演示Self-Ask）
    test_q = "地球到月球的平均距离乘以2等于多少千米？"
    print(f"[原始问题] {test_q}")

    # 执行Self-Ask
    result = self_ask(test_q)

    # 打印最终结果
    print(f"[最终答案] {result}")
