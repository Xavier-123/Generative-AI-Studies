import torch


# ==========================================
# 1. 手动实现线性模型（包含 forward、backward 和 parameters）
# ==========================================
class MyLinear:
    def __init__(self, in_features, out_features):
        # 随机初始化权重 w 和偏置 b
        self.w = torch.randn(in_features, out_features)
        self.b = torch.zeros(out_features)

        # 初始化梯度为 None
        self.w.grad = None
        self.b.grad = None

        # 缓存输入，反向传播求导时需要用到 x
        self.x = None

    def forward(self, x):
        self.x = x
        # 前向计算: y = x * w + b
        return torch.matmul(x, self.w) + self.b

    def backward(self, grad_output):
        """
        grad_output 即为下游(Loss层)传回来的 dL/d(y_hat)
        """
        # 手动计算 dL/dw = x^T * grad_output
        self.w.grad = torch.matmul(self.x.t(), grad_output)

        # 手动计算 dL/db = sum(grad_output)
        self.b.grad = torch.sum(grad_output, dim=0)

    def parameters(self):
        """返回所有需要优化的参数列表"""
        return [self.w, self.b]


# ==========================================
# 2. 手动实现 MSE 损失函数（包含 forward 和 backward）
# ==========================================
class MyMSELoss:
    def __init__(self):
        self.pred = None
        self.target = None

    def forward(self, pred, target):
        # 缓存预测值和真实值
        self.pred = pred
        self.target = target
        # 前向计算: MSE = 1/N * sum((pred - target)^2)
        return torch.mean((pred - target) ** 2)

    def backward(self):
        """
        手动实现 MSE 对预测值 pred (y_hat) 的求导:
        公式: dL/d(pred) = 2/N * (pred - target)
        """
        N = self.pred.shape[0]  # 样本数量
        grad_output = (2.0 / N) * (self.pred - self.target)
        return grad_output


# ==========================================
# 3. 手动实现 SGD 优化器
# ==========================================
class MySGD:
    def __init__(self, params, lr=0.01):
        """
        :param params: 模型参数的可迭代对象 (例如 model.parameters())
        :param lr: 学习率 (learning rate)
        """
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        """清空所有参数的梯度，防止梯度在每个 step 中累加"""
        for p in self.params:
            p.grad = None

    def step(self):
        """
        执行一步参数更新
        公式: p = p - lr * grad
        """
        for p in self.params:
            if p.grad is not None:
                # 使用原地减法更新张量的值
                p -= self.lr * p.grad


# ==========================================
# 4. 准备数据
# ==========================================
# 4个样本，特征维度为1
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# 实例化模型、损失函数和优化器
model = MyLinear(in_features=1, out_features=1)
criterion = MyMSELoss()
optimizer = MySGD(model.parameters(), lr=0.01)

# ==========================================
# 5. 标准训练循环 (标准 5 步流程)
# ==========================================
for epoch in range(200):
    # ① 清空梯度 (Zero Grad)
    optimizer.zero_grad()

    # ② 模型前向传播 (Forward)
    predictions = model.forward(x)

    # ③ Loss 前向计算 (Loss Forward)
    loss = criterion.forward(predictions, y)

    # ④ 手动反向传播计算梯度 (Backward)
    grad_loss = criterion.backward()
    model.backward(grad_loss)

    # ⑤ 优化器更新参数 (Optimizer Step)
    optimizer.step()

    # 打印训练过程
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/200], Loss: {loss.item():.4f}")

# 6. 测试效果
test_x = torch.tensor([[5.0]])
test_pred = model.forward(test_x)
print(f"\n训练结束！")
print(f"学到的权重 w: {model.w.item():.4f} (真实目标 ≈ 2.0)")
print(f"学到的偏置 b: {model.b.item():.4f} (真实目标 ≈ 0.0)")
print(f"测试输入 5.0 的预测结果: {test_pred.item():.2f}")