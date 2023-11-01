import math

import torch
import torch.nn as nn
from torch import Tensor


class TokenEmbedding(nn.Module):
    """
    d_model: 代表权重矩阵的列数
    vocab: 表示词表的大小，即编码的词表总大小。
    """

    def __int__(self, d_model, vocab):
        super(TokenEmbedding, self).__int__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab, d_model)  # 就是要将词向量的维度从vocab编码到d_model

    def forward(self, token: Tensor):
        return self.embedding(token.long()) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __int__(self, d_model, dropout=0.1, max_length=5000):
        super(PositionalEncoding, self).__int__()
        pe = torch.zeros(max_length, d_model)
        self.dropout = dropout
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        # position = torch.arange(0,max_length,dtype=torch.float).reshape(max_length,1) 等同于上述操作
        '''
        这里的unsqueeze函数：就是在指定的位置插入一个维度，有两个参数，input是输入的tensor,dim是要插到的维度。
        例子：
        <<<
        a = torch.arange(0, 5)
        b = torch.arange(0,5).unsqueeze(1)
        print(a.size())
        print(b.size())
        <<<
        输出：
        torch.Size([5])
        torch.Size([5, 1])
        '''
