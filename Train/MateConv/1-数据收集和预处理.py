import json

# 文件路径
file_paths = [
    # r"./sft_data_zh.jsonl",
    r"./tokenizer_train.jsonl",
    # r"./mobvoi_seq_monkey_general_open_corpus.jsonl",
]

# 读取并展示文件内容
def read_and_display_samples(file_path, num_samples=2):
    print(f"Reading from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                data = json.loads(line.strip())
                print(f"Sample {i+1}: {json.dumps(data, indent=4, ensure_ascii=False)}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    print("\n ============================= \n")

# 遍历文件并读取样本
for path in file_paths:
    read_and_display_samples(path)