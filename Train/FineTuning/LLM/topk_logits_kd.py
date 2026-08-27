"""
文件 1: topk_logits_kd.py
功能: Top-K Logits 离线流式量化缓存与学生端 Gather 蒸馏训练

核心机制：
教师端 Top-K 截断与 INT8 动态量化存储（大幅降低 I/O 开销）。
流式加载与反量化重构。
学生端仅 Gather 对应 Top-K 索引计算 KL 散度与 CE 损失。
"""


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================
# 1. 简易模型定义 (模拟 Teacher & Student)
# ============================
class SimpleLLM(nn.Module):
    def __init__(self, vocab_size=1000, hidden_dim=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=256, batch_first=True),
            num_layers=num_layers
        )
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids):
        # 期望 input_ids 维度: [Batch, Seq_Len]
        x = self.embedding(input_ids)  # [Batch, Seq_Len, hidden_dim]
        x = self.encoder(x)
        logits = self.lm_head(x)  # [Batch, Seq_Len, vocab_size]
        return logits


# ============================
# 2. 离线 Top-K 提取与 INT8 动态量化缓存
# ============================
def quantize_logits_to_int8(topk_logits):
    """
    对 Top-K logits 进行动态 Min-Max 8-bit 量化
    """
    min_val = topk_logits.min(dim=-1, keepdim=True)[0]
    max_val = topk_logits.max(dim=-1, keepdim=True)[0]
    scale = (max_val - min_val) / 255.0 + 1e-8
    q_logits = torch.clamp((topk_logits - min_val) / scale, 0, 255).to(torch.uint8)
    return q_logits, min_val, scale


def dequantize_logits(q_logits, min_val, scale):
    """
    反量化回浮点数
    """
    return q_logits.float() * scale + min_val


def generate_and_cache_topk_logits(teacher_model, dataloader, cache_file_path, top_k=16, temperature=1.5):
    """
    模拟 Teacher 模型生成离线 Top-K Logits 缓存（按单样本拆解存储）
    """
    teacher_model.eval()
    cached_records = []
    device = next(teacher_model.parameters()).device
    print(f"[Teacher] 开始在 [{device}] 生成离线 Top-{top_k} 压缩缓存...")

    with torch.no_grad():
        for batch_input_ids in dataloader:
            batch_input_ids = batch_input_ids.to(device)
            # Teacher 前向推理
            logits = teacher_model(batch_input_ids) / temperature
            # 截取 Top-K
            topk_vals, topk_indices = torch.topk(logits, k=top_k, dim=-1)

            # 动态量化
            q_vals, min_val, scale = quantize_logits_to_int8(topk_vals)

            # 【关键修复】按样本维度拆解保存，避免后续 DataLoader 叠加维度
            batch_size = batch_input_ids.size(0)
            for b in range(batch_size):
                cached_records.append({
                    "input_ids": batch_input_ids[b].cpu(),  # [Seq_Len]
                    "topk_indices": topk_indices[b].to(torch.int32).cpu(),  # [Seq_Len, K]
                    "q_vals": q_vals[b].cpu(),  # [Seq_Len, K]
                    "min_val": min_val[b].to(torch.float16).cpu(),  # [Seq_Len, 1]
                    "scale": scale[b].to(torch.float16).cpu()  # [Seq_Len, 1]
                })

    torch.save(cached_records, cache_file_path)
    print(f"[Teacher] 离线缓存构建完成，共计 {len(cached_records)} 条样本，已存入: {cache_file_path}")


# ============================
# 3. 流式读取 Dataset 与 Top-K KD 损失函数
# ============================
class CachedTopKDataset(Dataset):
    def __init__(self, cache_file_path):
        self.records = torch.load(cache_file_path, weights_only=False)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        input_ids = item["input_ids"]  # [Seq_Len]
        topk_indices = item["topk_indices"].to(torch.long)  # [Seq_Len, K]

        # 运行时流式反量化
        topk_logits = dequantize_logits(
            item["q_vals"],
            item["min_val"],
            item["scale"]
        )  # [Seq_Len, K]
        return input_ids, topk_indices, topk_logits


class TopKDistillationLoss(nn.Module):
    def __init__(self, alpha=0.6, temperature=1.5):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, student_logits, teacher_topk_logits, teacher_topk_indices, targets):
        """
        student_logits: [B, L, Vocab]
        teacher_topk_logits: [B, L, K]
        teacher_topk_indices: [B, L, K] (dtype=torch.long)
        targets: [B, L] (真实硬标签)
        """
        B, L, V = student_logits.shape

        # 1. 计算常规交叉熵损失 (Next-Token Prediction)
        shift_student_logits = student_logits[:, :-1, :].contiguous().view(-1, V)
        shift_targets = targets[:, 1:].contiguous().view(-1)
        loss_ce = self.ce_loss_fn(shift_student_logits, shift_targets)

        # 2. 计算 Top-K KL 散度蒸馏损失
        # Student 在 Teacher 的 Top-K 索引位置 Gather logits
        s_logits_scaled = student_logits / self.temperature
        student_topk_logits = torch.gather(s_logits_scaled, dim=-1, index=teacher_topk_indices)

        # 概率重归一化 (Softmax over Top-K subset)
        # p_teacher = F.softmax(teacher_topk_logits, dim=-1)
        p_teacher = F.softmax(teacher_topk_logits / self.temperature, dim=-1)
        log_p_student = F.log_softmax(student_topk_logits, dim=-1)

        # KL 散度 (Batch 平均)
        loss_kd = F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (self.temperature ** 2)

        # 3. 联合加权损失
        total_loss = (1.0 - self.alpha) * loss_ce + self.alpha * loss_kd
        return total_loss, loss_ce.item(), loss_kd.item()


# ============================
# 4. 执行完整 Demo 流程
# ============================
if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VOCAB_SIZE = 500
    SEQ_LEN = 32
    NUM_SAMPLES = 64
    CACHE_PATH = "./teacher_topk_cache.pt"

    print(f"当前运行设备: {device}")

    # 1. 模拟通信数据 (例如 5GC 信令与拓扑排障 Token)
    dummy_data = torch.randint(0, VOCAB_SIZE, (NUM_SAMPLES, SEQ_LEN))
    offline_loader = DataLoader(dummy_data, batch_size=4, shuffle=False)

    # 2. 模拟 Teacher 生成 Top-K 缓存
    teacher_model = SimpleLLM(vocab_size=VOCAB_SIZE, hidden_dim=256, num_layers=4).to(device)
    generate_and_cache_topk_logits(teacher_model, offline_loader, CACHE_PATH, top_k=16, temperature=1.5)

    # 3. 训练 Student 模型
    student_model = SimpleLLM(vocab_size=VOCAB_SIZE, hidden_dim=128, num_layers=2).to(device)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-3)
    kd_loss_fn = TopKDistillationLoss(alpha=0.6, temperature=1.5)

    dataset = CachedTopKDataset(CACHE_PATH)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    print("\n[Student] 开始基于 Top-K 离线缓存进行蒸馏训练...")
    student_model.train()
    for epoch in range(3):
        epoch_total_loss = 0.0
        for input_ids, topk_indices, teacher_topk_logits in train_loader:
            input_ids = input_ids.to(device)
            topk_indices = topk_indices.to(device)
            teacher_topk_logits = teacher_topk_logits.to(device)

            optimizer.zero_grad()
            student_logits = student_model(input_ids)
            loss, ce, kd = kd_loss_fn(student_logits, teacher_topk_logits, topk_indices, targets=input_ids)

            loss.backward()
            optimizer.step()
            epoch_total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/3 | Total Loss: {epoch_total_loss / len(train_loader):.4f} (CE: {ce:.4f}, KD: {kd:.4f})")

    # 清理生成的临时缓存文件
    if os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)
    print("\nTop-K Logits 离线流式蒸馏流程成功执行完毕！")