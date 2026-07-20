import torch
import torch.nn as nn
embedding_dim = 512
num_heads = 8
seq_len = 10
batch_size = 4


multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=0.1)

query = torch.randn(seq_len, batch_size, embedding_dim)
key = torch.randn(seq_len, batch_size, embedding_dim)
value = torch.randn(seq_len, batch_size, embedding_dim)

attn_output, attn_weights = multihead_attn(query, key, value, average_attn_weights=False)

print("Attention Output Shape:", attn_output.shape)  # 输出形状: (sequence_length, batch_size, embedding_dim)