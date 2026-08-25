import torch
import torch.nn as nn
import torch.optim as optim

# 1. 准备极简数据: x 是输入, y 是真实标签 (y = 2x)
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# 2. 定义模型、损失函数、优化器
model = nn.Linear(1, 1)  # 最简单的线性模型: y = w*x + b
criterion = nn.MSELoss()  # 均方误差损失 (Mean Squared Error)
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 梯度下降优化器

# 3. 训练循环 (只演示核心的4步)
for epoch in range(200):
    # ① 前向传播 (Forward)
    predictions = model(x)

    # ② 计算损失 (Loss)
    loss = criterion(predictions, y)

    # ③ 梯度清零 + 反向传播 (Backward)
    optimizer.zero_grad()  # 清空上一步的残余梯度
    loss.backward()  # 自动计算梯度

    # ④ 梯度优化/更新权重 (Optimization)
    optimizer.step()  # 根据梯度更新参数: w = w - lr * grad

    # 每50轮打印一次损失
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/200], Loss: {loss.item():.4f}")

# 4. 测试模型预测
test_x = torch.tensor([[5.0]])
print(f"\n测试输入 5.0，模型预测结果: {model(test_x).item():.2f} (真实值应该是 10.0)")