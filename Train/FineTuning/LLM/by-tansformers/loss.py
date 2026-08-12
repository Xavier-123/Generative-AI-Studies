"""SFT 训练 loss。

显式实现交叉熵，不依赖模型 forward 内部的隐式 loss，
便于后续改成 label smoothing、token 加权、focal loss 等变体。
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn

from config import IGNORE_INDEX


def compute_loss(model, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, int]:
    """显式计算 SFT 交叉熵 loss，返回 (loss, 参与计算的 token 数)。"""
    outputs = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
    )

    # 自回归错位：用前 t 个 token 预测第 t+1 个
    shift_logits = outputs.logits[:, :-1, :].contiguous()
    shift_labels = batch['labels'][:, 1:].contiguous()

    loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    # 在 fp32 下算 softmax/交叉熵，避免低精度累加带来的数值误差
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1).to(shift_logits.device),
    )
    num_tokens = int((shift_labels != IGNORE_INDEX).sum().item())
    return loss, num_tokens
