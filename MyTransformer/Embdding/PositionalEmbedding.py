import torch
from torch import nn


# step 2、 接下来进行位置编码
# 位置编码就是用论文中的两个公式进行计算
class PositionalEmbedding(nn.Module):
    def __init__(self, dim_vector, max_len):
        super().__init__()

        self.dim_vector = dim_vector
        self.max_len = max_len

        pe =
