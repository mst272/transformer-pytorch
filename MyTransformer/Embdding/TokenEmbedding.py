# step 1、第一步是TokenEmbedding，即词向量编码

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    词向量编码将每个词用一个向量表示


    :param vocab_size: size of vocabulary 词表的大小
    :param dim_vector: 每个词向量的长度
    """

    def __init__(self, vocab_size, dim_vector):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, dim_vector)

    def forward(self, x):
        return self.token_embed(x)


# ------------------------测试例子----------------------------------------------

if __name__ == '__main__':

    # 整体的词表是0-9共10个词，下面的input输入了两个句子，每个句子的长度均为4个词。
    # Pytorch是进行矩阵运算的，故输入的tensor必须是矩阵，所以就有了后面的padding（补全）操作。
    input_word = torch.LongTensor([[1, 0, 4, 5], [4, 3, 2, 9]])
    print(input_word.size())
    embedding = TokenEmbedding(10, 6)
    # 输出的output即为token embedding后的矩阵，矩阵的每一行代表一个单词表示。
    output = embedding(input_word)
    print(output.size())
    print(output)












