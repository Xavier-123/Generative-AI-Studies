import os

from openai import OpenAI
import dotenv

# ===================== 配置区（请修改这里） =====================
dotenv.load_dotenv("../../.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")

# 2. 初始化OpenAI客户端
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


# --------------------------
# GoT 核心函数：极简图思维实现
# 功能：生成多分支思考 → 整合分支 → 输出最终答案
# --------------------------
def simple_got(question: str) -> str:
    """
    最简单的GoT（图思维）调用函数
    :param question: 用户的问题
    :return: 经过GoT思考后的最终答案
    """
    # ====================== GoT 第一步：生成【多个思考分支】（树/图的节点）
    # 给大模型指令：生成3个不同角度的思考方案（GoT的核心：多分支探索）
    prompt_branch = f"""
    问题：{question}
    要求：生成3个不同角度的解决方案/思考方向，简洁明了，不要多余内容
    """
    # 调用大模型生成分支
    response_branch = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt_branch}],
        stream=False,
    )
    branches = response_branch.choices[0].message.content  # 获取多分支结果

    # ====================== GoT 第二步：整合【所有思考分支】（图的合并优化）
    # 给大模型指令：汇总所有分支，输出最优、最简洁的最终答案
    prompt_merge = f"""
    以下是针对问题的多个思考分支：
    {branches}
    要求：整合所有分支的优点，输出最终、最完善的答案
    """
    # 调用大模型整合答案
    response_merge = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt_merge}],
        stream=False,
    )
    final_answer = response_merge.choices[0].message.content  # 最终答案

    return final_answer


# --------------------------
# 主程序：测试GoT功能
# --------------------------
if __name__ == "__main__":
    # 测试问题（可随意修改）
    user_question = "如何快速学会Python编程？"
    print("=" * 50)
    print(f"用户问题：{user_question}")
    print("=" * 50)

    # 调用GoT函数
    answer = simple_got(user_question)

    # 打印结果
    print("GoT最终答案：\n", answer)
