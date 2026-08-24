import jieba
import numpy as np
from datasketch import MinHash, MinHashLSH
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def get_minhash(text: str, num_perm: int = 128) -> MinHash:
    """计算文本的 MinHash 对象（支持中文分词 / n-gram）"""
    m = MinHash(num_perm=num_perm)
    # 中文建议按词分词，也可以按 2-gram / 3-gram 切分
    tokens = list(jieba.cut(text))
    # 为防止词过少，加入 2-gram
    shingles = tokens + [tokens[i] + tokens[i + 1] for i in range(len(tokens) - 1)]
    for s in shingles:
        m.update(s.encode('utf8'))
    return m


class DualDeduplicator:
    def __init__(
            self,
            minhash_threshold: float = 0.7,  # MinHash 相似度阈值
            embedding_threshold: float = 0.85,  # 语义向量相似度阈值
            model_name: str = "BAAI/bge-small-zh-v1.5",  # 中文小模型，英文可用 all-MiniLM-L6-v2
            num_perm: int = 128
    ):
        self.minhash_threshold = minhash_threshold
        self.embedding_threshold = embedding_threshold
        self.num_perm = num_perm

        # 初始化 LSH 索引
        self.lsh = MinHashLSH(threshold=minhash_threshold, num_perm=num_perm)

        # 加载 Embedding 模型
        print(f"正在加载语义模型: {model_name} ...")
        self.model = SentenceTransformer(model_name)

    def deduplicate(self, corpus: list[str]) -> list[dict]:
        """
        双重去重主流程：
        返回保留的文档列表及处理日志
        """
        print(f"\n[开始去重] 原始文档数量: {len(corpus)}")

        # -------------------------------------------------------------
        # Stage 1: MinHash LSH 快速字面粗筛 (O(N) 级别)
        # -------------------------------------------------------------
        stage1_kept_indices = []
        minhash_objects = []

        for idx, text in enumerate(corpus):
            mh = get_minhash(text, self.num_perm)
            minhash_objects.append(mh)

            # 查询是否与已有文档在 LSH 桶中碰撞且相似度 > 阈值
            candidates = self.lsh.query(mh)

            if not candidates:
                # 没有找到高度相似的字面重复项，保留
                self.lsh.insert(f"doc_{idx}", mh)
                stage1_kept_indices.append(idx)
            else:
                # 命中了字面相似
                print(f"[-] Doc {idx} 被 MinHash 过滤 (与 Doc {candidates[0]} 字面高度相似)")

        print(f"[Stage 1 完成] MinHash 过滤后剩余: {len(stage1_kept_indices)} 篇")

        # -------------------------------------------------------------
        # Stage 2: Embedding 语义向量精筛 (只计算通过 Stage 1 的样本)
        # -------------------------------------------------------------
        stage1_texts = [corpus[i] for i in stage1_kept_indices]

        # 批量计算剩余文档的语义向量
        embeddings = self.model.encode(stage1_texts, normalize_embeddings=True, show_progress_bar=False)

        final_kept_indices = []
        kept_embeddings = []

        for i, original_idx in enumerate(stage1_kept_indices):
            current_emb = embeddings[i].reshape(1, -1)

            if not kept_embeddings:
                final_kept_indices.append(original_idx)
                kept_embeddings.append(current_emb)
                continue

            # 计算当前文档与已保留文档的 Cosine 相似度最大值
            sim_matrix = cosine_similarity(current_emb, np.vstack(kept_embeddings))
            max_sim = np.max(sim_matrix)
            matched_idx = np.argmax(sim_matrix)

            if max_sim > self.embedding_threshold:
                print(
                    f"[-] Doc {original_idx} 被 Embedding 过滤 (与 Doc {final_kept_indices[matched_idx]} 语义相似度: {max_sim:.4f})")
            else:
                final_kept_indices.append(original_idx)
                kept_embeddings.append(current_emb)

        print(f"[Stage 2 完成] Embedding 过滤后最终保留: {len(final_kept_indices)} 篇\n")

        return [
            {"doc_id": idx, "text": corpus[idx]}
            for idx in final_kept_indices
        ]


# -------------------------------------------------------------
# 测试样例
# -------------------------------------------------------------
if __name__ == "__main__":
    test_corpus = [
        # 0. 原始基准文本 1
        "人工智能正在迅速改变人类社会的生产生活方式，深度学习是其核心驱动力。",

        # 1. 字面高相似（增删几个虚词） -> 预期被 MinHash 过滤
        "人工智能正在改变人类社会的生产与生活方式，深度学习是其核心的驱动力。",

        # 2. 语义同义改写（字面重合低，但意思完全一样） -> 预期通过 MinHash，被 Embedding 过滤
        "深度学习作为关键的推动力量，AI技术正在对人类日常工作与生活产生深刻的变革。",

        # 3. 独立不相关文本 1
        "今天北京的天气非常好，阳光明媚，适合去奥林匹克森林公园跑步运动。",

        # 4. 独立不相关文本 2（领域相似但语义不同）
        "量子计算利用量子叠加态进行高速信息处理，未来可能在密码学领域带来颠覆性影响。"
    ]

    deduplicator = DualDeduplicator(
        minhash_threshold=0.75,  # MinHash 阈值（0.7-0.85 适中）
        embedding_threshold=0.82,  # 语义相似度阈值（通常 0.8-0.9 判定为改写重复）
        model_name=r"E:\models\BAAI\bge-small-zh-v1___5"
    )

    results = deduplicator.deduplicate(test_corpus)

    print("=" * 20 + " 最终去重结果 " + "=" * 20)
    for item in results:
        print(f"[保留 ID: {item['doc_id']}] {item['text']}")