# step 3、 将token embedding 和 positional embedding相加

import torch
import torch.nn as nn
from MyTransformer.Embedding.PositionalEmbedding import PositionalEmbedding
from MyTransformer.Embedding.TokenEmbedding import TokenEmbedding


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, dim_vector, max_len, drop_out):
        super().__init__()
        self.token_embd = TokenEmbedding(vocab_size, dim_vector)
        self.position_embd = PositionalEmbedding(dim_vector, max_len)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x):
        # 输入x 的维度 [batch_size, seq_len]
        batch_size, seq_len = x.size()
        token_embd = self.token_embd(x)  # 此部分输出是[batch_size, seq_len, dim_vector]
        position_embd = self.position_embd(x)  # 此部分输出是[seq_len, dim_vector]

        return self.dropout(token_embd + position_embd)


# ------------------------测试例子----------------------------------------------

if __name__ == '__main__':

    # 整体的词表是0-9共10个词，下面的input输入了两个句子，每个句子的长度均为4个词。
    # Pytorch是进行矩阵运算的，故输入的tensor必须是矩阵，所以就有了后面的padding（补全）操作。
    input_word = torch.LongTensor([[1, 0, 4, 5], [4, 3, 2, 9]])
    print(input_word.size())
    embedding = TransformerEmbedding(10, 6, 10, 0.1)
    # 输出的output即为token embedding后的矩阵，矩阵的每一行代表一个单词表示。
    output = embedding(input_word)
    print(output.size())
    print(output)