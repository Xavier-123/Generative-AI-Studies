"""基于 PyTorch + Transformers 的 Qwen SFT 微调训练入口。

数据来源为 `1-sft-data_proccess.py` 生成的 CSV 文件（列: history, q, a）。
训练时使用 Qwen 的对话模板拼接 system / history / user / assistant，
并且只对 assistant 的回答部分计算 loss（prompt 部分的 label 置为 -100）。

不使用 transformers 的 Trainer：前向、loss、反向、梯度累积、梯度裁剪、
学习率调度、评估与断点保存全部用原生 PyTorch 手写实现。

支持两种训练方式：
    - LoRA 微调（USE_LORA = True，显存友好，推荐单卡使用）
    - 全参数微调（USE_LORA = False）

模块划分：
    config.py        超参数与运行环境（设备、精度、随机种子）
    data.py          CSV -> 对话模板 -> token 编码 -> 动态 padding
    modeling.py      模型 / tokenizer 加载，LoRA 注入，梯度检查点
    loss.py          显式的 SFT 交叉熵 loss
    optimization.py  优化器、学习率调度、GradScaler
    trainer.py       训练循环、评估、断点保存

运行:
    python main.py
"""

from config import SEED, set_seed
from data import load_and_prepare_dataset
from modeling import load_model_and_tokenizer
from trainer import train


def main():
    set_seed(SEED)
    model, tokenizer = load_model_and_tokenizer()
    train_dataset, val_dataset = load_and_prepare_dataset(tokenizer)
    train(model, tokenizer, train_dataset, val_dataset)


if __name__ == '__main__':
    main()
