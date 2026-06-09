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


def plan_and_solve(question: str) -> str:
    """
    核心函数：Plan & Solve 流程
    :param question: 用户的问题
    :return: 最终解决后的答案
    """
    # ------------------- 第一步：Plan（制定解决计划）-------------------
    print("=== 正在生成解决计划 ===")
    plan_prompt = f"""
    请为解决以下问题，制定**简单、清晰、分步**的执行计划，只列步骤，不要多余内容：
    问题：{question}
    """
    # 调用大模型生成计划
    plan_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": plan_prompt}],
        temperature=0,
        extra_body={
            "enable_thinking": False,
        }
    )
    # 提取计划内容
    solve_plan = plan_response.choices[0].message.content.strip()
    print(f"解决计划：\n{solve_plan}\n")

    # ------------------- 第二步：Solve（按照计划解决问题）-------------------
    print("=== 正在按照计划解决问题 ===")
    solve_prompt = f"""
    请严格按照以下计划，逐步解决问题，给出最终答案：
    问题：{question}
    解决计划：{solve_plan}
    """
    # 调用大模型执行解决
    solve_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": solve_prompt}]
    )
    # 提取最终答案
    final_answer = solve_response.choices[0].message.content.strip()

    return final_answer


# ===================== 测试运行 =====================
if __name__ == "__main__":
    # 测试问题（可以替换成任意问题：数学、逻辑、文案、编程等）
    user_question = "一个长方形长5米，宽3米，求周长和面积分别是多少？"

    # 执行 Plan & Solve
    result = plan_and_solve(user_question)

    # 输出最终结果
    print("\n=== 最终答案 ===")
    print(result)
