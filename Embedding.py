import math
import torch
import torch.nn as nn
from torch import Tensor


class TokenEmbedding(nn.Module):
    """
    d_model: 代表权重矩阵的列数
    vocab: 表示词表的大小，即编码的词表总大小。
    """

    def __init__(self, d_model, vocab):
        super(TokenEmbedding, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab, d_model)  # 就是要将词向量的维度从vocab编码到d_model

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_length=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_length, d_model)
        self.dropout = nn.Dropout(p=dropout)
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
        # Position Encoding. Transformer位置无关，需要加上位置编码。 i in [0,emb_size/2)
        # PE(pos,2i) = sin(pos/10000^(2i/d)) # 偶数位置
        # PE(pos,2i+1) = cos(pos/10000^(2i/d)) # 奇数位置
        # 对 pos/10000^(2i/d) 取log就是下面的东西
        two_i = torch.arange(0, d_model, 2, dtype=torch.float32)
        div_term = torch.exp(two_i * -(math.log(1000) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 上行代码输出后pe：[max_length*d_model],还需做一个维度变换
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)  # 可简单理解为这个参数不更新

    def forward(self, x: Tensor):
        """
            :param x: [x_len, batch_size, emb_size]
            :return: [x_len, batch_size, emb_size]
            :pe: [max_len, 1, d_model]
        """
        x += self.pe[:x.size(0), :]  # [x_len, batch_size, d_model]
        return self.dropout(x)


if __name__ == '__main__':
    x = torch.tensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]], dtype=torch.long)
    x = x.reshape(5, 2)
    print(f"x size:{x.size()}")
    token_embedding = TokenEmbedding(d_model=512, vocab=11)
    x = token_embedding(tokens=x)
    print(f"X after token embedding size:{x.size()}")
    pos_embedding = PositionalEncoding(d_model=512)
    x = pos_embedding(x=x)
    print(f"After position embedding size:{x.shape}")
