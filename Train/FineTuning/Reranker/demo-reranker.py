import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

# 1. 基础配置
model_id = r"E:\models\Qwen\Qwen3-Reranker-0.6B"
# 提示：请确保 transformers>=4.51.0
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # CausalLM 训练建议右填充

# 2. 准备数据
train_data = {
    "query": [
        "如何吃得健康？",
        "如何吃得健康？",
        "什么是人工智能？",
        "什么是人工智能？"
    ],
    "document": [
        "保持饮食均衡，多吃蔬菜水果，减少高糖高脂摄入是健康饮食的关键。",
        "今天天气真好，特别适合去公园散步和慢跑。",
        "人工智能是计算机科学的一个分支，旨在模拟人类的智能行为和思维过程。",
        "小明今天早上在早餐店吃了一个煎饼果子。"
    ],
    "label": [1.0, 0.0, 1.0, 0.0]  # 1.0 对应 yes，0.0 对应 no
}
dataset = Dataset.from_dict(train_data)


# 3. 构造 Qwen3 Reranker 官方 Prompt 模板
def format_prompt(query, document):
    instruction = "Given a web search query, retrieve relevant passages that answer the query"
    return (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
        "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
        "<|im_start|>user\n"
        f"<Instruct>: {instruction}\n"
        f"<Query>: {query}\n"
        f"<Document>: {document}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# 4. 数据分词与 Label 掩码（只对 yes/no 计算 Loss）
def preprocess_function(examples):
    input_ids_batch = []
    labels_batch = []

    for query, doc, label in zip(examples["query"], examples["document"], examples["label"]):
        prompt = format_prompt(query, doc)
        target = "yes" if label == 1.0 else "no"

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = tokenizer.encode(target, add_special_tokens=False) + [tokenizer.eos_token_id]

        input_ids = prompt_ids + target_ids
        # 将 Prompt 部分的 Label 设为 -100，避免其参与梯度计算
        labels = [-100] * len(prompt_ids) + target_ids

        # 截断超长文本
        if len(input_ids) > 512:
            prompt_ids = prompt_ids[:512 - len(target_ids)]
            input_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids

        input_ids_batch.append(input_ids)
        labels_batch.append(labels)

    return {"input_ids": input_ids_batch, "labels": labels_batch}


tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

# 5. 加载 CausalLM 模型
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# 6. 设置训练参数
training_args = TrainingArguments(
    output_dir="./qwen3-reranker-causal-sft",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=1,
    save_strategy="no",
    fp16=torch.cuda.is_available(),
)

# 7. 序列到序列数据整理器
data_collator = DataCollatorWithPadding_or_Seq2Seq = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100
)

# 8. 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("开始微调生成式重排模型...")
trainer.train()

# 9. 保存
model.save_pretrained("./qwen3-reranker-causal-sft")
tokenizer.save_pretrained("./qwen3-reranker-causal-sft")
print("保存成功")