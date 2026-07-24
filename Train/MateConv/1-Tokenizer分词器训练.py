
'''
1. 准备训练数据
'''
import random
import pandas as pd
csv_paths = ["E:/dataset/AI-ModelScope/alpaca-gpt4-data-zh/train.csv", "E:/dataset/AI-ModelScope/alpaca-gpt4-data-en/train.csv"]
texts = []

for csv_path in csv_paths:
    df = pd.read_csv(csv_path, encoding="utf-8")

    # 2. 提取目标字段，把"字段名"替换成csv里真实列名
    instruction = list(df["instruction"])
    input = list(df["input"])
    output = list(df["output"])

    for text in instruction:
        if text.strip() and isinstance(text, str):
            texts.append(text)

    for text in input:
        if text.strip() and isinstance(text, str):
            texts.append(text)

    for text in output:
        if text.strip() and isinstance(text, str):
            texts.append(text)

random.seed(41)
random.shuffle(texts)

with open("corpus.txt", "w", encoding="utf-8") as f:
    for line in texts:
        f.write(line.lstrip("\n").rstrip("\n") + "\n")

'''
2. 训练分词器代码实例
'''
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


def train_tokenizer(corpus_file, vocab_size: int =5000):
    # 1. 初始化 BPE 模型
    # unk_token 是当遇到完全无法识别的内容时的占位符（在 Byte-level BPE 中其实很少见）
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    # 2. 配置预分词器 (Pre-tokenizer)
    # ByteLevel 会把文本拆分成字节，并处理空格。这是目前 Llama/GPT 常用方式。
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    # 3. 配置训练器
    # 设置词表大小和特殊符号
    # 特殊符号的顺序很重要，通常 index 0 是 bos, 1 是 eos 等
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,  # 词频小于2的会被忽略
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],    # 定义特殊token
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet()
    )

    # 4. 开始训练
    files = [corpus_file]
    tokenizer.train(files, trainer)

    # 5. 配置后处理器 (Post-processor)
    # 负责在序列前后添加 <s> 和 </s> 等
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=True)

    # 6. 配置解码器
    tokenizer.decoder = ByteLevelDecoder()

    # 7. 保存分词器
    tokenizer.save("my_tokenizer.json")
    print("训练完成，分词器已保存为 my_tokenizer.json")


# 执行训练
# train_tokenizer("tokenizer_train.jsonl")
train_tokenizer("corpus.txt")


'''
3. 加载并使用分词器
'''
from tokenizers import Tokenizer

# 加载
tokenizer = Tokenizer.from_file("my_tokenizer.json")
# tokenizer = Tokenizer.from_file("E:/models/Qwen/Qwen3-0.6B/tokenizer.json")

# 编码测试
text = "你好，LLM！"
# text = "hello，人工智能！"
encoded = tokenizer.encode(text)

print(f"原始文本: {text}")
print(f"分词 ID: {encoded.ids}")
print(f"分词 Token: {encoded.tokens}")

# 解码测试
decoded = tokenizer.decode(encoded.ids)
print(f"解码文本: {decoded}")


'''
4. 在 Transformers 库中使用
'''
from transformers import PreTrainedTokenizerFast

fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="my_tokenizer.json",
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    mask_token="<mask>"
)

# 这样就可以像使用标准 LlamaTokenizer 一样使用了
print(fast_tokenizer("你好，世界")["input_ids"])


# import json
# def read_texts_from_jsonl(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             data = json.loads(line)
#             yield data['text']
#
# # 测试读取数据，你可以更换成你建立的目录
# data_path = './tokenizer_train.jsonl'
# texts = read_texts_from_jsonl(data_path)
# for text in texts:
#     print(text)