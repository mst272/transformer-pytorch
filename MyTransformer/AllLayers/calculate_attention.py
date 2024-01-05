"""
此部分是 step4、构建多头注意力层的子部分，即注意力(attention)计算的实现模块
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# class CalculateAttention(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self):

def calculate_attention(q, k, v, mask=None):
    # 输入的是四维矩阵
    batch_size, n_head, seq_len, dim_split = q.size()

    # 1、开始计算
    k_t = k.transpose(2, 3)
    attention = torch.matmul(q, k_t) / math.sqrt(dim_split)

    # 2、选择是否mask
    if mask is not None:
        attention = attention.masked_fill(mask == 0, -10000)

    # 3、 进行softmax
    attention = F.softmax(attention, dim=-1)  # dim=-1表示在最后一维即dim_split维进行softmax



