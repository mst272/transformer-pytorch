import torch
from torch import nn


# step 2、 接下来进行位置编码
# 位置编码就是用论文中的两个公式进行计算
# 在上一步token embedding中，生成的矩阵size是 [len_size,dim_vector]，由于会padding，故len_size即为max_len
class PositionalEmbedding(nn.Module):
    def __init__(self, dim_vector, max_len):
        super().__init__()

        self.dim_vector = dim_vector
        self.max_len = max_len

        pe = torch.zeros(max_len, dim_vector)
        for pos in range(max_len):
            for i in range(0, dim_vector, 2):
                pe[pos, i] = torch.sin(pos / (10000 ** ((2 * i) / dim_vector)))
                pe[pos, i+1] = torch.cos(pos / (10000**((2 * i) / dim_vector)))

        # 此时的pe矩阵维度为 [len_size, dim_vector]
        print(f"pe矩阵的维度为：{pe.size()}")

        # 需要进行一个维度的变换, 下面代码变换后得到的维度为 [1, len_size, dim_vector]
        pe = pe.unsqueeze(0)

        # 注册缓冲区，表示这个参数不更新。 Tips：注册缓冲区后相当于在__init__中定义了self.pe，但是参数不会更新。
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 假设 pe为 [max_len = 512, d_model = 512]

        # 位置编码的输入X一般是[batch_size, seq_len],其中seq_len是句子的长度，即不加padding的句子长度
        batch_size, seq_len = x.size()
        return self.pe[:seq_len, :]





