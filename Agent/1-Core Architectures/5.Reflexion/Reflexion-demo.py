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


# ==============================================================

# ===================== Reflexion 核心实现 =====================
def llm_call(prompt: str) -> str:
    """
    封装 OpenAI 大模型调用函数（通用工具）
    :param prompt: 给大模型的提示词
    :return: 大模型的文本回答
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        extra_body={
            "enable_thinking": False,
        }
    )
    # 返回大模型的回答内容
    return response.choices[0].message.content.strip()


def reflexion_agent(question: str) -> str:
    """
    最简单的 Reflexion 智能体：生成 → 反思 → 修正（核心逻辑）
    :param question: 用户的问题
    :return: 最终优化后的答案
    """
    # ========== 步骤1：大模型生成【初步答案】 ==========
    print("=" * 50)
    print("步骤1：生成初步答案")
    initial_prompt = f"请回答问题：{question}，直接给出答案和简要过程"
    initial_answer = llm_call(initial_prompt)
    print(f"初步答案：\n{initial_answer}\n")

    # ========== 步骤2：大模型【自我反思】 ==========
    print("步骤2：自我反思（检查错误/漏洞/不严谨之处）")
    reflect_prompt = f"""
    问题：{question}
    初步答案：{initial_answer}
    请你严格检查这个答案的错误、逻辑漏洞、计算错误或表述不严谨的地方，只输出反思结果，不要修改答案。
    """
    reflect_result = llm_call(reflect_prompt)
    print(f"反思结果：\n{reflect_result}\n")

    # ========== 步骤3：基于反思【修正答案】 ==========
    print("步骤3：根据反思修正答案")
    refine_prompt = f"""
    问题：{question}
    原答案：{initial_answer}
    反思意见：{reflect_result}
    请根据反思意见，修正得到最终正确答案，简洁明了。
    """
    final_answer = llm_call(refine_prompt)
    print(f"最终答案：\n{final_answer}")
    print("=" * 50)

    return final_answer


# ===================== 测试运行 =====================
if __name__ == "__main__":
    # 测试问题（选了一个容易出错的数学题，能明显看到 Reflexion 的效果）
    user_question = "100以内最大的质数是多少？请给出推理过程"
    # 调用 Reflexion 智能体
    final_result = reflexion_agent(user_question)
