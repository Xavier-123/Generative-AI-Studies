import os
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss

# 1. 准备演示数据 (Query - Positive)
# 在实际应用中，您应该准备几千到上万条类似的领域相关数据
train_data = {
    "query": [
        "如何吃得健康？",
        "什么是人工智能？",
        "北京有哪些著名景点？",
        "如何学习Python编程？",
        "红烧肉怎么做？"
    ],
    "positive": [
        "保持饮食均衡，多吃蔬菜水果，减少高糖和高脂肪食物的摄入是健康饮食的关键。",
        "人工智能是计算机科学的一个分支，旨在创造能够模拟人类智能行为的系统。",
        "北京的著名景点包括故宫、天坛、颐和园以及八达岭长城等历史文化遗迹。",
        "学习Python建议从基础语法开始，多写代码，并结合实际项目进行练习。",
        "做红烧肉需要将五花肉切块，焯水后加入冰糖、酱油、八角等调料慢火炖煮。"
    ]
}

# 转换为 Hugging Face Dataset 格式
# 注意：列名需要对应模型期望的输入，一般为 'anchor' (即query) 和 'positive'
dataset = Dataset.from_dict({
    "anchor": train_data["query"],
    "positive": train_data["positive"]
})

# 2. 加载 BGE 基础模型
# model_id = "BAAI/bge-small-zh-v1.5"
model_id = r"E:\models\BAAI\bge-small-zh-v1___5"
model = SentenceTransformer(model_id)

# 3. 定义损失函数
# MultipleNegativesRankingLoss 是微调检索/重排模型常用的损失函数
train_loss = MultipleNegativesRankingLoss(model)

# 4. 设置训练参数
# 可根据显存大小和实际需求调整 batch_size 和 epochs
training_args = SentenceTransformerTrainingArguments(
    output_dir="./bge-small-zh-v1.5-fine-tuned",  # 模型保存路径
    num_train_epochs=3,                          # 训练轮数
    per_device_train_batch_size=2,               # 批次大小 (Demo使用较小值)
    learning_rate=2e-5,                          # 学习率
    warmup_ratio=0.1,                            # 预热比例
    weight_decay=0.01,                           # 权重衰减
    logging_steps=1,                             # 打印日志的步数
    save_strategy="no",                          # 演示暂不中途保存
    fp16=False,                                  # 如果GPU支持可开启混合精度 (设置为 True)
)

# 5. 初始化 Trainer
trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    loss=train_loss,
)

# 6. 开始训练
print("开始微调模型...")
trainer.train()

# 7. 保存微调后的模型
model.save_pretrained("./bge-small-zh-v1.5-fine-tuned")
print("模型微调完成并已保存至 ./bge-small-zh-v1.5-fine-tuned")