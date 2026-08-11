# import math
# import struct
# import inspect
# import time
#
# # import LMConfig
# from typing import Any, Optional, Tuple
# import numpy as np
import torch
import torch.nn.functional as F
# from torch import nn
# from transformers import PreTrainedModel
# from transformers.modeling_outputs import CausalLMOutputWithPast


def func1():
    import numpy as np

    # 定义三个权重矩阵和对应的 top-k 矩阵
    # 非常均衡、但是每个专家对每个token的处理高度类似
    # 没有实现专家的特异化
    weights_1 = np.array([
        [0.34, 0.34, 0.32],
        [0.34, 0.32, 0.34],
        [0.34, 0.34, 0.32],
        [0.32, 0.34, 0.34]
    ])

    topk_1 = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [1, 1, 0],
        [0, 1, 1]
    ])

    # 每个专家都有被用到，也实现了一定的特异化
    weights_2 = np.array([
        [0.6, 0.25, 0.15],
        [0.4, 0.4, 0.2],
        [0.3, 0.6, 0.1],
        [0.04, 0.06, 0.9]
    ])

    topk_2 = np.array([
        [1, 1, 0],
        [1, 1, 0],
        [1, 1, 0],
        [0, 1, 1]
    ])

    # 不均衡，只有专家1和专家2被使用
    weights_3 = np.array([
        [0.6, 0.25, 0.15],
        [0.4, 0.4, 0.2],
        [0.3, 0.6, 0.1],
        [0.9, 0.06, 0.04]
    ])

    topk_3 = np.array([
        [1, 1, 0],
        [1, 1, 0],
        [1, 1, 0],
        [1, 1, 0]
    ])

    # 定义计算 Pi * fi 的函数
    def calculate_aux_loss(weights, topk):
        Pi = weights.mean(axis=0)  # 计算纵向均值 Pi
        fi = topk.mean(axis=0)     # 计算纵向均值 fi
        Pi_fi = Pi * fi            # 计算 Pi * fi
        return Pi_fi.sum()         # 返回 aux_loss

    # 计算三个 aux_loss
    aux_loss_1 = calculate_aux_loss(weights_1, topk_1)
    aux_loss_2 = calculate_aux_loss(weights_2, topk_2)
    aux_loss_3 = calculate_aux_loss(weights_3, topk_3)

    # aux_loss_1, aux_loss_2, aux_loss_3


def func2():
    bsz = 3
    seq_len = 10

    #从所有专家那里获得的输出结果
    hidden_states = torch.randn(size=(30,512))

    #权重的初始化参数
    self_weight = torch.randn(size=(6,512))
    # self_weight = torch.randn(size=(512,4))

    #利用线性层将二者相连，构建每个token上每个专家的权重
    weights = F.linear(hidden_states, self_weight, None)

    print(weights.shape) #每个token在每个专家上的得分

    weights = weights.softmax(-1)  # 6个专家的得分总和被归一化到[0,1]之间

    topk_weight, topk_idx = torch.topk(weights, k=2, dim=-1, sorted=False)

    topk_idx_for_aux_loss = topk_idx.view(3, -1)
    mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=6)
    print(mask_ce)

if __name__ == '__main__':
    func2()