"""SFT 训练 loss。

交叉熵为手写实现，既不依赖模型 forward 内部的隐式 loss，
也不依赖 nn.CrossEntropyLoss，便于后续改成 label smoothing、
token 加权、focal loss 等变体。
"""

from typing import Dict, Tuple

import torch

from config import IGNORE_INDEX


def cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """手写交叉熵，等价于 nn.CrossEntropyLoss(ignore_index=..., reduction='mean')。

    交叉熵即目标类别的负对数概率 -log softmax(z)[t]，展开为：
        loss = log(Σ_i exp(z_i)) - z_t

    Args:
        logits:  (N, V) 未归一化得分。
        targets: (N,) 类别下标，等于 ignore_index 的位置不参与计算。

    Returns:
        有效位置上的平均 loss；若全部被忽略则返回 0（而非 nn 版本的 nan）。
    """
    # log-sum-exp 技巧：先减去每行最大值再取 exp，防止 exp 溢出为 inf。
    # 该平移不改变 softmax 的结果，对梯度也没有影响，故可直接 detach。
    max_logits = logits.max(dim=-1, keepdim=True).values.detach()
    shifted = logits - max_logits

    # log(Σ_i exp(z_i - max))
    log_sum_exp = shifted.exp().sum(dim=-1).log()

    valid = targets != ignore_index
    # 被忽略的位置填 0 占位，避免 gather 拿到 -100 这样的非法下标
    safe_targets = targets.masked_fill(~valid, 0)
    target_logits = shifted.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)

    nll = log_sum_exp - target_logits
    nll = nll.masked_fill(~valid, 0.0)

    # reduction='mean' 的语义：只在有效 token 上做平均
    return nll.sum() / valid.sum().clamp(min=1)


def compute_loss(model, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, int]:
    """显式计算 SFT 交叉熵 loss，返回 (loss, 参与计算的 token 数)。"""
    outputs = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
    )

    # 自回归错位：用前 t 个 token 预测第 t+1 个
    shift_logits = outputs.logits[:, :-1, :].contiguous()
    shift_labels = batch['labels'][:, 1:].contiguous()

    # 在 fp32 下算 softmax/交叉熵，避免低精度累加带来的数值误差
    loss = cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1).to(shift_logits.device),
    )
    num_tokens = int((shift_labels != IGNORE_INDEX).sum().item())
    return loss, num_tokens
