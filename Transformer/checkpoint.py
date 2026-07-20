import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


# 创建一个稍深的模型，包含激活函数
class LargeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # 堆叠多层，并使用 ReLU。ReLU 的中间状态正常情况下需要消耗显存
        self.layers = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)

class MyModel1(nn.Module):
    def __init__(self):
        super(MyModel1, self).__init__()
        self.layer1 = nn.Linear(512, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 512)

    def forward(self, x, use_checkpoint = False):
        if use_checkpoint:
            # 设置检查点
            x = checkpoint(self.layer1, x, use_reentrant=False)
            x = checkpoint(self.layer2, x, use_reentrant=False)
            x = checkpoint(self.layer3, x, use_reentrant=False)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        return x

# 创建模型和输入
model = MyModel1().cuda()
input_data = torch.randn(1000, 512).cuda()


# class MyModel2(nn.Module):
#     def __init__(self):
#         super(MyModel2, self).__init__()
#         self.block1 = LargeBlock()
#         self.block2 = LargeBlock()
#
#     def forward(self, x, use_checkpoint=False):
#         if use_checkpoint:
#             # 将整个 block 作为 checkpoint 的单位，而不是单个 Linear
#             x = checkpoint(self.block1, x, use_reentrant=False)
#             x = checkpoint(self.block2, x, use_reentrant=False)
#         else:
#             x = self.block1(x)
#             x = self.block2(x)
#         return x
#
# # 增大输入尺寸以使激活值占用更明显
# model = MyModel2().cuda()
# input_data = torch.randn(4000, 1024).cuda()


# 显存测量函数
def measure_memory(model, input_data, use_checkpoint=False):
    torch.cuda.empty_cache()  # 清空显存缓存
    torch.cuda.reset_peak_memory_stats()  # 重置显存统计

    torch.cuda.synchronize()  # 同步 GPU 计算，帮助torch.cuda的监控结果与nvidia-smi保持一致
    output = model(input_data, use_checkpoint=use_checkpoint)
    loss = output.sum()
    loss.backward()
    torch.cuda.synchronize()  # 确保反向传播完成

    # 获取内存分配情况
    allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
    reserved = torch.cuda.max_memory_reserved() / 1024**2  # MB
    return allocated, reserved

# 测量未使用梯度检查点的显存
allocated_no_checkpoint, reserved_no_checkpoint = measure_memory(model, input_data, use_checkpoint=False)
print(f"[No Checkpoint] Allocated: {allocated_no_checkpoint:.2f} MB, Reserved: {reserved_no_checkpoint:.2f} MB")

# 测量使用梯度检查点的显存
allocated_with_checkpoint, reserved_with_checkpoint = measure_memory(model, input_data, use_checkpoint=True)
print(f"[With Checkpoint] Allocated: {allocated_with_checkpoint:.2f} MB, Reserved: {reserved_with_checkpoint:.2f} MB")

# 显存节约对比
allocated_saving = (allocated_no_checkpoint - allocated_with_checkpoint) / allocated_no_checkpoint * 100
print(f"Memory saving (allocated): {allocated_saving:.2f}%")


