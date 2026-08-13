"""训练超参数与运行环境配置。

所有可调参数集中在此处，其余模块只从这里读取，不再重复定义。
注意：CUDA_VISIBLE_DEVICES 必须在 torch 初始化 CUDA 之前设置，
因此本模块应当是最先被导入的模块。
"""

import os
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import torch
import transformers

# =============================================================================
# 路径与数据
# =============================================================================
MODEL_PATH = r'E:\models\Qwen\Qwen2.5-0.5B-Instruct'   # 基础模型路径
DATA_PATH = r'../data/sft_data_mixed_single.csv'        # CSV / JSON / JSONL
OUTPUT_DIR = './output'                                 # 输出目录
SYSTEM_PROMPT = 'You are a helpful assistant.'          # 系统提示词

# 训练目标：
#   sft       - 普通 SFT；兼容 history/q/a，并且只监督最后一轮 assistant。
#   agent_sft - Agent 轨迹 SFT；输入必须包含 messages，监督轨迹中每一轮 assistant
#               （包括 tool_calls 和最终回答），不监督 system/user/tool。
TRAINING_MODE = 'sft'

# messages 数据没有 system 消息时，是否自动补上 SYSTEM_PROMPT。
ADD_SYSTEM_PROMPT_IF_MISSING = True

MAX_LENGTH = 2048        # 单条样本最大 token 数，超出截断
VAL_RATIO = 0.02         # 验证集比例
SEED = 42

# 忽略 loss 计算的 label 占位值
IGNORE_INDEX = -100

# =============================================================================
# LoRA 配置（USE_LORA=False 时切换为全参数微调）
# =============================================================================
USE_LORA = True
LORA_RANK = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    'q_proj', 'k_proj', 'v_proj', 'o_proj',
    'gate_proj', 'up_proj', 'down_proj',
]

# =============================================================================
# 训练超参数（替代 TrainingArguments）
# =============================================================================
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4            # 等效 batch = TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS
LEARNING_RATE = 1e-4 if USE_LORA else 2e-5
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.05
MAX_GRAD_NORM = 1.0
GRADIENT_CHECKPOINTING = True
NUM_WORKERS = 0

LOGGING_STEPS = 5               # 每多少个优化步打一次日志
EVAL_STEPS = 100                # 每多少个优化步跑一次验证
SAVE_STEPS = 100                # 每多少个优化步存一次断点
SAVE_TOTAL_LIMIT = 2            # 最多保留的断点数量

# =============================================================================
# 运行环境探测
# =============================================================================
# 设备与混合精度：优先 bf16，其次 fp16（需要 GradScaler），CPU 上关闭 AMP
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = DEVICE.type == 'cuda'
BF16_SUPPORTED = USE_AMP and torch.cuda.is_bf16_supported()
AMP_DTYPE = torch.bfloat16 if BF16_SUPPORTED else torch.float16

# transformers 5.x 起 from_pretrained 的 torch_dtype 参数更名为 dtype
DTYPE_KWARG = 'dtype' if int(transformers.__version__.split('.')[0]) >= 5 else 'torch_dtype'


def precision_name() -> str:
    return 'bf16' if BF16_SUPPORTED else ('fp16' if USE_AMP else 'fp32')


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
